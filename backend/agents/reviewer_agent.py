# =============================================================================
#  backend/agents/reviewer_agent.py
#  Reviewer Agent (Critic / Validator).
#  Sits between specialist agents and the Signal Agent.
#  Applies validation checks to each AgentOutput and assigns a verdict:
#    PASS  → forward normally (weight = 1.0)
#    WARN  → forward with reduced weight (weight = 0.5)
#    FAIL  → skip this agent's data (weight = 0.0)
#  This prevents bad / stale / anomalous data from corrupting final signals.
# =============================================================================

import logging
from datetime import datetime
from typing import Optional

from .base_agent import AgentOutput, BaseAgent, Verdict
from config.settings import settings

logger = logging.getLogger(__name__)


class ReviewerAgent:
    """
    The Reviewer Agent is not a specialist — it's a meta-agent.
    It receives the dict of all specialist outputs and validates each one.
    It does NOT generate its own score; it gates and weights others' scores.

    Checks performed:
      1. Schema check      — required fields present, correct types
      2. Confidence gate   — score confidence above minimum threshold
      3. Staleness check   — news data not too old
      4. Outlier guard     — score didn't jump suspiciously far
      5. Coverage gate     — enough articles / data points
      6. Cross-agent check — detect bullish TA + very negative sentiment
    """

    def __init__(self):
        self.name   = "ReviewerAgent"
        self.logger = logging.getLogger("agents.reviewer")
        # Track previous scores to detect jumps (in-memory per session)
        self._previous_scores: dict[str, dict[str, float]] = {}

    # ── Main review method ────────────────────────────────────────────────────
    def review(
        self,
        ticker:   str,
        outputs:  dict[str, AgentOutput],
    ) -> dict[str, AgentOutput]:
        """
        Reviews all agent outputs for a ticker.

        Args:
            ticker:  stock ticker
            outputs: dict of agent_name → AgentOutput

        Returns:
            Same dict with verdict, review_notes, and weight set on each output.
        """
        self.logger.info(f"[Reviewer] Reviewing {len(outputs)} agents for {ticker}")
        reviewed = {}

        for agent_name, output in outputs.items():
            reviewed_output = self._review_single(agent_name, output, outputs)
            reviewed[agent_name] = reviewed_output
            self.logger.info(
                f"[Reviewer] {agent_name} → {reviewed_output.verdict.value} "
                f"(weight={reviewed_output.weight})"
            )

        # Cross-agent conflict check (runs after all individual checks)
        reviewed = self._check_cross_agent_conflicts(reviewed)

        # Store scores for next run's outlier detection
        self._store_scores(ticker, reviewed)

        # Summary log
        verdicts = [o.verdict.value for o in reviewed.values()]
        self.logger.info(
            f"[Reviewer] Summary for {ticker}: "
            f"PASS={verdicts.count('PASS')}, "
            f"WARN={verdicts.count('WARN')}, "
            f"FAIL={verdicts.count('FAIL')}"
        )

        return reviewed

    # ── Single agent review ───────────────────────────────────────────────────
    def _review_single(
        self,
        agent_name: str,
        output:     AgentOutput,
        all_outputs: dict[str, AgentOutput],
    ) -> AgentOutput:
        """Run all checks on one agent's output. Modifies output in place."""
        notes    = []
        verdict  = Verdict.PASS
        weight   = 1.0
        severity = 0   # 0=pass, 1=warn, 2=fail

        # ── Check 1: Did the agent succeed at all? ─────────────────────────
        if not output.success:
            notes.append(f"Agent failed: {output.error_message}")
            severity = max(severity, 2)

        # ── Check 2: Schema — required fields present ─────────────────────
        schema_issues = self._check_schema(output)
        if schema_issues:
            notes.extend(schema_issues)
            severity = max(severity, 1)   # missing fields → WARN not FAIL

        # ── Check 3: Confidence gate ──────────────────────────────────────
        if output.confidence is not None:
            if output.confidence < settings.REVIEWER_MIN_CONFIDENCE:
                notes.append(
                    f"Low confidence: {output.confidence:.2f} "
                    f"(threshold: {settings.REVIEWER_MIN_CONFIDENCE})"
                )
                severity = max(severity, 1)

        # ── Check 4: Staleness (sentiment / news agents) ──────────────────
        if agent_name in ("SentimentAnalysis", "NewsFreshness"):
            stale_issue = self._check_staleness(output)
            if stale_issue:
                notes.append(stale_issue)
                severity = max(severity, 1)

        # ── Check 5: Outlier guard ────────────────────────────────────────
        outlier_issue = self._check_outlier(agent_name, output)
        if outlier_issue:
            notes.append(outlier_issue)
            severity = max(severity, 1)

        # ── Check 6: Minimum data coverage ───────────────────────────────
        coverage_issue = self._check_coverage(agent_name, output)
        if coverage_issue:
            notes.append(coverage_issue)
            severity = max(severity, 1)

        # ── Assign verdict and weight based on severity ───────────────────
        if severity == 0:
            verdict = Verdict.PASS
            weight  = 1.0
        elif severity == 1:
            verdict = Verdict.WARN
            weight  = 0.5
        else:
            verdict = Verdict.FAIL
            weight  = 0.0

        output.verdict      = verdict
        output.review_notes = notes
        output.weight       = weight
        return output

    # ── Individual checks ────────────────────────────────────────────────────

    def _check_schema(self, output: AgentOutput) -> list[str]:
        """Verify required fields are present and have correct types."""
        issues = []
        if output.score is None and output.success:
            issues.append("Missing 'score' field in output")
        if output.label is None and output.success:
            issues.append("Missing 'label' field in output")
        if output.score is not None and not isinstance(output.score, (int, float)):
            issues.append(f"Invalid score type: {type(output.score)}")
        if output.score is not None and (output.score < -1.5 or output.score > 105):
            issues.append(f"Score out of expected range: {output.score}")
        return issues

    def _check_staleness(self, output: AgentOutput) -> Optional[str]:
        """Check if news data is too old to be reliable."""
        meta  = output.metadata or {}
        data  = output.data or {}

        # Check article count
        article_count = data.get("article_count", None)
        if article_count is not None and article_count < settings.REVIEWER_MIN_HEADLINES:
            return (
                f"Too few articles ({article_count}), "
                f"minimum {settings.REVIEWER_MIN_HEADLINES} required"
            )

        # Check if news agent flagged staleness
        metrics = data.get("metrics", {})
        if metrics.get("is_stale"):
            avg_age = metrics.get("avg_age_hours", 0)
            return f"News is stale — average age {avg_age:.0f}h exceeds {settings.REVIEWER_MAX_NEWS_AGE_HOURS}h limit"

        return None

    def _check_outlier(self, agent_name: str, output: AgentOutput) -> Optional[str]:
        """
        Compare current score to previous run's score.
        A jump > MAX_SCORE_JUMP in one run suggests a data feed error.
        """
        if output.score is None:
            return None

        prev_scores = self._previous_scores.get(output.ticker, {})
        prev_score  = prev_scores.get(agent_name)

        if prev_score is not None:
            jump = abs(output.score - prev_score)
            if jump > settings.REVIEWER_MAX_SCORE_JUMP:
                return (
                    f"Outlier detected: score jumped {jump:.1f} points "
                    f"({prev_score:.1f} → {output.score:.1f})"
                )
        return None

    def _check_coverage(self, agent_name: str, output: AgentOutput) -> Optional[str]:
        """Verify the agent had enough data to produce a reliable result."""
        meta = output.metadata or {}
        data = output.data or {}

        if agent_name == "TechnicalAnalysis":
            bars = meta.get("bars", 0)
            if bars < 30:
                return f"Insufficient price bars: {bars} (minimum 30)"
            signals = data.get("detail", {})
            total   = signals.get("total", 0)
            if total < 5:
                return f"Too few TA signals computed: {total} (minimum 5)"

        elif agent_name == "VolumeAnalysis":
            bars = meta.get("bars", 0)
            if bars < 20:
                return f"Insufficient volume data: {bars} bars (minimum 20)"

        return None

    def _check_cross_agent_conflicts(
        self, outputs: dict[str, AgentOutput]
    ) -> dict[str, AgentOutput]:
        """
        Detect significant disagreement between agents.
        Example: TA says Strong Buy (score > 75) but Sentiment is Very Negative (score < -0.5).
        Adds a WARN note to both but doesn't change individual verdicts.
        """
        ta_out   = outputs.get("TechnicalAnalysis")
        sent_out = outputs.get("SentimentAnalysis")

        if ta_out and sent_out and ta_out.score and sent_out.score:
            ta_norm   = ta_out.score        # 0–100
            sent_norm = sent_out.score      # –1 to +1

            # Strong bullish TA + strongly negative sentiment → conflict
            if ta_norm > 75 and sent_norm < -0.3:
                msg = (
                    f"Cross-agent conflict: TA={ta_norm:.0f} (bullish) "
                    f"but Sentiment={sent_norm:.2f} (negative)"
                )
                ta_out.review_notes.append(msg)
                sent_out.review_notes.append(msg)
                # Escalate to WARN if currently PASS
                if ta_out.verdict == Verdict.PASS:
                    ta_out.verdict = Verdict.WARN
                    ta_out.weight  = 0.7

            # Strong bearish TA + strongly positive sentiment → also flag
            elif ta_norm < 25 and sent_norm > 0.3:
                msg = (
                    f"Cross-agent conflict: TA={ta_norm:.0f} (bearish) "
                    f"but Sentiment={sent_norm:.2f} (positive)"
                )
                ta_out.review_notes.append(msg)
                sent_out.review_notes.append(msg)
                if sent_out.verdict == Verdict.PASS:
                    sent_out.verdict = Verdict.WARN
                    sent_out.weight  = 0.7

        return outputs

    def _store_scores(self, ticker: str, outputs: dict[str, AgentOutput]) -> None:
        """Persist current scores for next run's outlier detection."""
        self._previous_scores[ticker] = {
            name: output.score
            for name, output in outputs.items()
            if output.score is not None
        }

    # ── Summary helper for the UI ─────────────────────────────────────────────
    def get_review_summary(self, outputs: dict[str, AgentOutput]) -> dict:
        """Returns a summary dict for rendering in the Flask API response."""
        return {
            "total_agents":  len(outputs),
            "passed":        sum(1 for o in outputs.values() if o.verdict == Verdict.PASS),
            "warned":        sum(1 for o in outputs.values() if o.verdict == Verdict.WARN),
            "failed":        sum(1 for o in outputs.values() if o.verdict == Verdict.FAIL),
            "overall_health": self._overall_health(outputs),
            "per_agent": {
                name: {
                    "verdict":      out.verdict.value,
                    "weight":       out.weight,
                    "notes":        out.review_notes,
                }
                for name, out in outputs.items()
            },
        }

    def _overall_health(self, outputs: dict[str, AgentOutput]) -> str:
        failed = sum(1 for o in outputs.values() if o.verdict == Verdict.FAIL)
        warned = sum(1 for o in outputs.values() if o.verdict == Verdict.WARN)
        if failed >= 2:       return "CRITICAL"
        elif failed == 1:     return "DEGRADED"
        elif warned >= 2:     return "WARNING"
        elif warned == 1:     return "CAUTION"
        else:                 return "HEALTHY"

# =============================================================================
#  backend/orchestrator.py
#  Pipeline Orchestrator.
#  Coordinates agent execution, manages shared state, and returns
#  a unified result object for each ticker run.
#  Implements a LangGraph-inspired state machine pattern using Python.
# =============================================================================

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from backend.agents import (
    AgentOutput, Verdict,
    TechnicalAnalysisAgent,
    VolumeAgent,
    SentimentAgent,
    NewsAgent,
    SignalAgent,
    ReviewerAgent,
)
from config.settings import settings

logger = logging.getLogger(__name__)


# ── Pipeline state object (LangGraph-inspired typed state) ────────────────────
@dataclass
class PipelineState:
    """
    Shared state passed through the agent graph.
    The orchestrator reads and writes this object at each node.
    """
    ticker:          str
    started_at:      datetime              = field(default_factory=datetime.utcnow)
    completed_at:    Optional[datetime]    = None

    # Raw agent outputs (before review)
    raw_outputs:     dict[str, AgentOutput] = field(default_factory=dict)

    # Reviewed outputs (after Reviewer Agent)
    reviewed_outputs: dict[str, AgentOutput] = field(default_factory=dict)

    # Final signal
    signal_output:   Optional[AgentOutput]  = None

    # Reviewer summary
    review_summary:  dict                    = field(default_factory=dict)

    # Pipeline metadata
    duration_seconds: Optional[float]        = None
    agents_run:       list[str]              = field(default_factory=list)
    error:            Optional[str]          = None

    def to_dict(self) -> dict:
        return {
            "ticker":          self.ticker,
            "started_at":      self.started_at.isoformat(),
            "completed_at":    self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "agents_run":      self.agents_run,
            "error":           self.error,
            "review_summary":  self.review_summary,
            "raw_outputs": {
                k: v.to_dict() for k, v in self.raw_outputs.items()
            },
            "reviewed_outputs": {
                k: v.to_dict() for k, v in self.reviewed_outputs.items()
            },
            "signal": self.signal_output.to_dict() if self.signal_output else None,
        }


# ── Main orchestrator class ───────────────────────────────────────────────────
class StockSenseOrchestrator:
    """
    Central orchestrator for the multi-agent pipeline.

    Execution graph:
      [Data Fetch] → [TA + Volume + News] → [Sentiment] → [Reviewer] → [Signal]
                         ↑ (parallel)            ↑
                    These run in parallel to minimise latency.

    Supports:
      - Full pipeline run (all agents)
      - Partial runs (specific agents only)
      - Parallel execution within each stage
    """

    def __init__(self):
        self.ta_agent        = TechnicalAnalysisAgent()
        self.volume_agent    = VolumeAgent()
        self.sentiment_agent = SentimentAgent()
        self.news_agent      = NewsAgent()
        self.signal_agent    = SignalAgent()
        self.reviewer        = ReviewerAgent()

        # In-memory result cache: ticker → PipelineState
        self._cache: dict[str, PipelineState] = {}

        logger.info("StockSenseOrchestrator initialised with all agents")

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def run_full_pipeline(
        self,
        ticker:    str,
        use_cache: bool = False,
    ) -> PipelineState:
        """
        Run the complete multi-agent pipeline for a single ticker.

        Args:
            ticker:    Stock ticker (e.g. "RELIANCE.NS")
            use_cache: Return cached result if available

        Returns:
            PipelineState with all outputs populated
        """
        if use_cache and ticker in self._cache:
            logger.info(f"[Orchestrator] Cache hit for {ticker}")
            return self._cache[ticker]

        state = PipelineState(ticker=ticker)
        t0    = time.time()

        logger.info(f"[Orchestrator] Starting full pipeline for {ticker}")

        try:
            # ── Stage 1: Run TA, Volume, and News agents in parallel ──────
            state = self._stage_parallel_analysis(state)

            # ── Stage 2: Sentiment agent (slightly slower — I/O bound) ────
            state = self._stage_sentiment(state)

            # ── Stage 3: Reviewer validates all outputs ───────────────────
            state = self._stage_review(state)

            # ── Stage 4: Signal agent aggregates into final signal ────────
            state = self._stage_signal(state)

        except Exception as exc:
            logger.error(f"[Orchestrator] Pipeline failed for {ticker}: {exc}", exc_info=True)
            state.error = str(exc)

        finally:
            state.completed_at    = datetime.utcnow()
            state.duration_seconds = round(time.time() - t0, 2)
            logger.info(
                f"[Orchestrator] Pipeline for {ticker} completed in "
                f"{state.duration_seconds}s"
            )

        self._cache[ticker] = state
        return state

    # ── Stage implementations ─────────────────────────────────────────────────

    def _stage_parallel_analysis(self, state: PipelineState) -> PipelineState:
        """
        Stage 1: TA, Volume, and News agents run in parallel threads.
        These are independent and don't depend on each other's outputs.
        """
        logger.info(f"[Stage 1] Parallel analysis — TA, Volume, News")

        def run_agent(name, fn, ticker):
            return name, fn(ticker)

        tasks = {
            "TechnicalAnalysis": (self.ta_agent.execute,     state.ticker),
            "VolumeAnalysis":    (self.volume_agent.execute, state.ticker),
            "NewsFreshness":     (self.news_agent.execute,   state.ticker),
        }

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(fn, ticker): name
                for name, (fn, ticker) in tasks.items()
            }
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    output = future.result(timeout=30)
                    state.raw_outputs[agent_name] = output
                    state.agents_run.append(agent_name)
                    logger.info(f"[Stage 1] {agent_name} completed — score={output.score}")
                except Exception as e:
                    logger.error(f"[Stage 1] {agent_name} raised exception: {e}")
                    from backend.agents.base_agent import AgentOutput
                    state.raw_outputs[agent_name] = AgentOutput.failure(
                        agent_name, state.ticker, str(e)
                    )

        return state

    def _stage_sentiment(self, state: PipelineState) -> PipelineState:
        """Stage 2: Sentiment agent (network I/O — runs after parallel stage)."""
        logger.info("[Stage 2] Sentiment analysis")
        output = self.sentiment_agent.execute(state.ticker)
        state.raw_outputs["SentimentAnalysis"] = output
        state.agents_run.append("SentimentAnalysis")
        logger.info(f"[Stage 2] Sentiment completed — score={output.score}")
        return state

    def _stage_review(self, state: PipelineState) -> PipelineState:
        """Stage 3: Reviewer Agent validates all specialist outputs."""
        logger.info("[Stage 3] Reviewer validation")
        state.reviewed_outputs = self.reviewer.review(
            state.ticker, state.raw_outputs
        )
        state.review_summary = self.reviewer.get_review_summary(state.reviewed_outputs)
        logger.info(
            f"[Stage 3] Review complete — health={state.review_summary.get('overall_health')}"
        )
        return state

    def _stage_signal(self, state: PipelineState) -> PipelineState:
        """Stage 4: Signal Agent produces final trading recommendation."""
        logger.info("[Stage 4] Signal aggregation")
        state.signal_output = self.signal_agent.execute(
            state.ticker,
            reviewed_outputs=state.reviewed_outputs,
        )
        logger.info(
            f"[Stage 4] Signal — {state.signal_output.label} "
            f"(score={state.signal_output.score})"
        )
        return state

    # ── Partial pipeline runs (for individual agent access from UI) ───────────

    def run_ta_only(self, ticker: str) -> AgentOutput:
        """Run only the TA agent (user selects 'Technical Analysis' mode)."""
        return self.ta_agent.execute(ticker)

    def run_sentiment_only(self, ticker: str) -> AgentOutput:
        """Run only the sentiment agent."""
        return self.sentiment_agent.execute(ticker)

    def run_volume_only(self, ticker: str) -> AgentOutput:
        """Run only the volume agent."""
        return self.volume_agent.execute(ticker)

    # ── Screener: run pipeline for multiple tickers ───────────────────────────
    def run_screener(
        self,
        tickers:      list[str],
        strategy:     str = "all",
        top_n:        int = 10,
        max_workers:  int = 4,
    ) -> list[dict]:
        """
        Run the full pipeline on a list of tickers and return ranked results.

        Args:
            tickers:     List of tickers to screen
            strategy:    "long_term" | "swing_trading" | "short_term" | "all"
            top_n:       Number of top stocks to return
            max_workers: Parallel worker threads

        Returns:
            List of ranked result dicts, sorted by strategy score descending
        """
        logger.info(f"[Screener] Starting for {len(tickers)} tickers, strategy={strategy}")
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self.run_full_pipeline, ticker): ticker
                for ticker in tickers
            }
            for future in as_completed(future_map):
                ticker = future_map[future]
                try:
                    state = future.result(timeout=60)
                    if state.signal_output and state.signal_output.success:
                        results.append(self._format_screener_result(state, strategy))
                except Exception as e:
                    logger.error(f"[Screener] Failed for {ticker}: {e}")

        # Sort by strategy score (descending)
        score_key = "score" if strategy == "all" else f"{strategy}_score"
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:top_n]

    def _format_screener_result(self, state: PipelineState, strategy: str) -> dict:
        """Format a pipeline state into a screener result row."""
        sig       = state.signal_output
        strategies = sig.data.get("strategies", {}) if sig else {}
        price_data = {}

        ta_out = state.reviewed_outputs.get("TechnicalAnalysis")
        if ta_out and ta_out.data:
            price_data = ta_out.data.get("price", {})

        return {
            "ticker":               state.ticker,
            "signal":               sig.label if sig else "N/A",
            "score":                sig.score if sig else 0,
            "confidence":           sig.confidence if sig else 0,
            "price":                price_data.get("current", 0),
            "change_pct":           price_data.get("change_pct", 0),
            "ta_score":             state.reviewed_outputs.get("TechnicalAnalysis", AgentOutput("", "")).score,
            "sentiment_score":      state.reviewed_outputs.get("SentimentAnalysis", AgentOutput("", "")).score,
            "volume_score":         state.reviewed_outputs.get("VolumeAnalysis", AgentOutput("", "")).score,
            "long_term_score":      strategies.get("long_term", {}).get("score", 0),
            "swing_trading_score":  strategies.get("swing_trading", {}).get("score", 0),
            "short_term_score":     strategies.get("short_term", {}).get("score", 0),
            "review_health":        state.review_summary.get("overall_health", "N/A"),
            "duration_s":           state.duration_seconds,
        }

    def clear_cache(self) -> None:
        """Clear the in-memory result cache."""
        self._cache.clear()
        logger.info("[Orchestrator] Cache cleared")

    def get_cache_keys(self) -> list[str]:
        """Return list of tickers currently in cache."""
        return list(self._cache.keys())

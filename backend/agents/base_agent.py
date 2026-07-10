# =============================================================================
#  backend/agents/base_agent.py
#  Base class and shared data structures for all specialist agents.
#  Every agent must inherit BaseAgent and implement the run() method.
# =============================================================================

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Verdict enum used by the Reviewer Agent ───────────────────────────────────
class Verdict(str, Enum):
    PASS  = "PASS"    # output is reliable — forward normally
    WARN  = "WARN"    # output has issues — forward with lower weight
    FAIL  = "FAIL"    # output is unreliable — skip and flag in UI


# ── Standardised output returned by every agent ───────────────────────────────
@dataclass
class AgentOutput:
    """
    Every agent returns an AgentOutput. The Reviewer Agent reads this,
    applies its checks, and sets verdict + review_notes before the output
    reaches the Signal Agent.
    """
    agent_name:     str                         # which agent produced this
    ticker:         str                         # stock ticker (e.g. "RELIANCE.NS")
    timestamp:      datetime = field(default_factory=datetime.utcnow)
    success:        bool     = True             # did the agent complete without error?
    error_message:  Optional[str] = None        # error detail if success=False

    # ── Core output payload (each agent fills what's relevant) ────────────
    score:          Optional[float] = None      # primary numeric score (0–100 or –1 to +1)
    label:          Optional[str]   = None      # human-readable label (e.g. "Strong Buy")
    confidence:     Optional[float] = None      # model confidence 0–1
    data:           dict = field(default_factory=dict)   # agent-specific detail payload
    metadata:       dict = field(default_factory=dict)   # timestamps, counts, sources

    # ── Set by Reviewer Agent after validation ────────────────────────────
    verdict:        Verdict          = Verdict.PASS
    review_notes:   list[str]        = field(default_factory=list)
    weight:         float            = 1.0      # reduced by WARN, zeroed by FAIL

    def to_dict(self) -> dict:
        """Serialise to JSON-safe dict for Flask API responses."""
        return {
            "agent_name":    self.agent_name,
            "ticker":        self.ticker,
            "timestamp":     self.timestamp.isoformat(),
            "success":       self.success,
            "error_message": self.error_message,
            "score":         round(self.score, 4) if self.score is not None else None,
            "label":         self.label,
            "confidence":    round(self.confidence, 4) if self.confidence is not None else None,
            "data":          self.data,
            "metadata":      self.metadata,
            "verdict":       self.verdict.value,
            "review_notes":  self.review_notes,
            "weight":        self.weight,
        }

    @classmethod
    def failure(cls, agent_name: str, ticker: str, error: str) -> "AgentOutput":
        """Factory method for a failed agent run — keeps error handling concise."""
        return cls(
            agent_name    = agent_name,
            ticker        = ticker,
            success       = False,
            error_message = error,
            verdict       = Verdict.FAIL,
        )


# ── Abstract base class all agents inherit ────────────────────────────────────
class BaseAgent(ABC):
    """
    All specialist agents inherit this class.
    Provides logging, error handling, and enforces the run() interface.
    """

    def __init__(self, name: str):
        self.name   = name
        self.logger = logging.getLogger(f"agents.{name}")

    def execute(self, ticker: str, **kwargs) -> AgentOutput:
        """
        Public entry point called by the orchestrator.
        Wraps run() with error catching so one agent failure never
        crashes the full pipeline.
        """
        self.logger.info(f"[{self.name}] Starting for ticker: {ticker}")
        try:
            output = self.run(ticker, **kwargs)
            self.logger.info(
                f"[{self.name}] Completed — score={output.score}, label={output.label}"
            )
            return output
        except Exception as exc:
            self.logger.error(f"[{self.name}] Failed for {ticker}: {exc}", exc_info=True)
            return AgentOutput.failure(self.name, ticker, str(exc))

    @abstractmethod
    def run(self, ticker: str, **kwargs) -> AgentOutput:
        """
        Subclasses implement their domain logic here.
        Must always return an AgentOutput, even on partial failure.
        """
        raise NotImplementedError

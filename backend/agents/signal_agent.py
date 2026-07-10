# =============================================================================
#  backend/agents/signal_agent.py
#  Signal Aggregation Agent.
#  Takes reviewed outputs from all specialist agents and produces:
#  — Final trading signal (Strong Buy / Buy / Hold / Sell / Strong Sell)
#  — Confidence score
#  — Strategy-specific recommendations (long-term / swing / short-term)
#  Uses a LightGBM model if trained; falls back to weighted rule fusion.
# =============================================================================

import os
import pickle
import numpy as np
import logging
from typing import Optional

from .base_agent import AgentOutput, BaseAgent, Verdict
from config.settings import settings

logger = logging.getLogger(__name__)


class SignalAgent(BaseAgent):
    """
    Final signal aggregator.
    Combines TA, Volume, Sentiment, and News scores (weighted by Reviewer verdicts)
    into a unified trading recommendation.

    Strategies served:
        - Long-term (fundamental + sentiment momentum)
        - Swing trading (TA + volume confluence)
        - Short-term (momentum + sentiment burst)
    """

    # Signal thresholds for each output label
    THRESHOLDS = {
        "Strong Buy":  80,
        "Buy":         62,
        "Hold":        42,
        "Sell":        25,
        "Strong Sell": 0,
    }

    def __init__(self):
        super().__init__("SignalAggregation")
        self.model = self._load_model()

    def _load_model(self) -> Optional[object]:
        """Load pre-trained LightGBM model if it exists."""
        model_path = settings.LIGHTGBM_MODEL_PATH
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                self.logger.info(f"LightGBM model loaded from {model_path}")
                return model
            except Exception as e:
                self.logger.warning(f"Could not load model: {e}. Using rule-based fusion.")
        return None

    def run(self, ticker: str, reviewed_outputs: dict[str, AgentOutput] = None, **kwargs) -> AgentOutput:
        if not reviewed_outputs:
            return AgentOutput.failure(self.name, ticker, "No agent outputs provided")

        # 1. Extract weighted scores from reviewed outputs
        feature_vector = self._extract_features(reviewed_outputs)

        # 2. Generate signal (model or rule-based)
        if self.model:
            score, confidence = self._model_predict(feature_vector)
        else:
            score, confidence = self._rule_fusion(feature_vector, reviewed_outputs)

        # 3. Derive label and strategy signals
        label     = self._score_to_label(score)
        strategies = self._strategy_breakdown(score, feature_vector, reviewed_outputs)

        return AgentOutput(
            agent_name = self.name,
            ticker     = ticker,
            score      = score,
            label      = label,
            confidence = round(confidence, 4),
            data = {
                "signal":           label,
                "score_0_100":      round(score, 2),
                "features":         feature_vector,
                "strategies":       strategies,
                "agent_weights":    {
                    name: out.weight
                    for name, out in reviewed_outputs.items()
                },
            },
            metadata={"fusion_method": "lightgbm" if self.model else "rule_based"},
        )

    # ── Feature extraction ────────────────────────────────────────────────────
    def _extract_features(self, outputs: dict[str, AgentOutput]) -> dict:
        """
        Pull scalar features from each agent output.
        Missing / FAIL agents get neutral fallback values.
        """
        ta   = outputs.get("TechnicalAnalysis")
        vol  = outputs.get("VolumeAnalysis")
        sent = outputs.get("SentimentAnalysis")
        news = outputs.get("NewsFreshness")

        def weighted_score(out: Optional[AgentOutput], default: float, scale: float = 1.0) -> float:
            """Apply reviewer weight to score; return default if FAIL."""
            if out is None or out.verdict == Verdict.FAIL or out.score is None:
                return default
            return float(out.score) * out.weight * scale

        return {
            # TA features (0–100 → keep as-is)
            "ta_score":        weighted_score(ta, 50.0),
            "ta_confidence":   float(ta.confidence or 0.5) if ta else 0.5,
            "rsi":             float(ta.data.get("indicators", {}).get("rsi", 50)) if ta else 50,
            "macd_hist":       float(ta.data.get("indicators", {}).get("macd_hist", 0)) if ta else 0,
            "adx":             float(ta.data.get("indicators", {}).get("adx", 20)) if ta else 20,

            # Volume features (0–100)
            "volume_score":    weighted_score(vol, 50.0),
            "volume_ratio":    float(vol.data.get("volume_ratio", 1.0)) if vol else 1.0,
            "obv_rising":      1.0 if (vol and vol.data.get("metrics", {}).get("obv_trend") == "rising") else 0.0,
            "above_vwap":      1.0 if (vol and vol.data.get("metrics", {}).get("above_vwap")) else 0.0,

            # Sentiment features (–1 to +1 → normalise to 0–100)
            "sentiment_score_raw": float(sent.score or 0) if sent else 0,
            "sentiment_score_norm": weighted_score(sent, 50.0, scale=50.0) + 50 if sent else 50.0,
            "sentiment_confidence": float(sent.confidence or 0.5) if sent else 0.5,
            "article_count":   float(sent.data.get("article_count", 0)) if sent else 0,

            # News quality (0–100)
            "news_score":      weighted_score(news, 50.0),
            "news_freshness":  1.0 if (news and not news.data.get("metrics", {}).get("is_stale")) else 0.0,
        }

    # ── Rule-based fusion ─────────────────────────────────────────────────────
    def _rule_fusion(
        self,
        features: dict,
        outputs:  dict[str, AgentOutput],
    ) -> tuple[float, float]:
        """
        Weighted average of agent scores when no LightGBM model is available.
        Weights are tuned for Indian market dynamics.
        """
        # Strategy weights: TA and sentiment are primary signals
        weights = {
            "ta_score":             0.35,
            "sentiment_score_norm": 0.30,
            "volume_score":         0.20,
            "news_score":           0.15,
        }

        weighted_sum = sum(features.get(k, 50) * w for k, w in weights.items())
        score        = weighted_sum / sum(weights.values())

        # Bonus / penalty adjustments
        if features["rsi"] < 30:   score += 3    # oversold boost
        if features["rsi"] > 70:   score -= 3    # overbought penalty
        if features["obv_rising"]: score += 2    # volume confirms move
        if features["above_vwap"]: score += 2    # price above VWAP
        if features["volume_ratio"] > 2.5: score += 3  # major volume spike

        # Confidence = weighted average of agent confidences
        confidences = [
            out.confidence or 0.5
            for out in outputs.values()
            if out.verdict.value != "FAIL"
        ]
        confidence = sum(confidences) / len(confidences) if confidences else 0.5

        return round(float(np.clip(score, 0, 100)), 2), confidence

    def _model_predict(self, features: dict) -> tuple[float, float]:
        """Run LightGBM model for prediction."""
        try:
            import numpy as np
            feature_order = [
                "ta_score", "ta_confidence", "rsi", "macd_hist", "adx",
                "volume_score", "volume_ratio", "obv_rising", "above_vwap",
                "sentiment_score_raw", "sentiment_score_norm", "sentiment_confidence",
                "article_count", "news_score", "news_freshness",
            ]
            X = np.array([[features.get(k, 0) for k in feature_order]])
            pred       = self.model.predict(X)[0]
            pred_proba = self.model.predict_proba(X)[0]
            # Convert class (0=Sell, 1=Hold, 2=Buy) to 0–100 score
            score      = float(pred_proba[0]) * 10 + float(pred_proba[1]) * 50 + float(pred_proba[2]) * 90
            confidence = float(max(pred_proba))
            return round(score, 2), round(confidence, 4)
        except Exception as e:
            self.logger.warning(f"Model prediction failed: {e}. Using rule fusion.")
            return self._rule_fusion(features, {})

    # ── Label and strategy ────────────────────────────────────────────────────
    def _score_to_label(self, score: float) -> str:
        if score >= 80:   return "Strong Buy"
        elif score >= 62: return "Buy"
        elif score >= 42: return "Hold"
        elif score >= 25: return "Sell"
        else:             return "Strong Sell"

    def _strategy_breakdown(
        self,
        score:    float,
        features: dict,
        outputs:  dict[str, AgentOutput],
    ) -> dict:
        """
        Generate strategy-specific recommendations.
        Different weightings of the same data for different holding periods.
        """
        ta_score   = features.get("ta_score", 50)
        sent_score = features.get("sentiment_score_norm", 50)
        vol_score  = features.get("volume_score", 50)
        adx        = features.get("adx", 20)
        rsi        = features.get("rsi", 50)

        # Long-term: sentiment + fundamentals > TA noise
        lt_score = sent_score * 0.50 + ta_score * 0.30 + vol_score * 0.20
        lt_score = float(np.clip(lt_score, 0, 100))

        # Swing trading: TA confluence + volume confirmation
        swing_score = ta_score * 0.45 + vol_score * 0.35 + sent_score * 0.20
        swing_score = float(np.clip(swing_score, 0, 100))

        # Short-term: momentum + volume spike
        st_score = ta_score * 0.40 + vol_score * 0.40 + sent_score * 0.20
        # Boost if strong ADX (trending) and not overbought
        if adx > 30 and 30 < rsi < 65:
            st_score = min(100, st_score + 5)
        st_score = float(np.clip(st_score, 0, 100))

        return {
            "long_term": {
                "score":  round(lt_score, 2),
                "signal": self._score_to_label(lt_score),
                "rationale": "Based on sentiment strength and fundamental trend",
            },
            "swing_trading": {
                "score":  round(swing_score, 2),
                "signal": self._score_to_label(swing_score),
                "rationale": "Based on TA confluence and volume confirmation",
            },
            "short_term": {
                "score":  round(st_score, 2),
                "signal": self._score_to_label(st_score),
                "rationale": "Based on momentum and volume spike",
            },
        }

# =============================================================================
#  backend/agents/volume_agent.py
#  Volume Analysis Agent.
#  Detects unusual volume, computes OBV trend, VWAP deviation,
#  and delivery percentage signals specific to Indian markets (NSE).
# =============================================================================

import numpy as np
import pandas as pd
import yfinance as yf

from .base_agent import AgentOutput, BaseAgent


class VolumeAgent(BaseAgent):
    """
    Analyses trading volume to detect:
      - Unusual volume spikes (z-score method)
      - OBV (On-Balance Volume) trend
      - VWAP deviation (price vs volume-weighted average)
      - Volume trend (expanding or contracting)

    Score 0–100:
        > 70 → bullish volume confirmation
        30–70 → neutral
        < 30 → bearish volume divergence
    """

    def __init__(self):
        super().__init__("VolumeAnalysis")

    def run(self, ticker: str, period: str = "3mo") -> AgentOutput:

        df = self._download(ticker, period)
        if df is None or len(df) < 20:
            return AgentOutput.failure(self.name, ticker, "Insufficient volume data")

        metrics = self._compute_volume_metrics(df)
        score, label = self._score(metrics, df)

        return AgentOutput(
            agent_name = self.name,
            ticker     = ticker,
            score      = score,
            label      = label,
            confidence = 0.75,
            data = {
                "metrics": metrics,
                "current_volume":  int(df["Volume"].iloc[-1]),
                "avg_volume_20d":  int(df["Volume"].tail(20).mean()),
                "volume_ratio":    round(df["Volume"].iloc[-1] / df["Volume"].tail(20).mean(), 2),
            },
            metadata={"period": period, "bars": len(df)},
        )

    def _download(self, ticker: str, period: str) -> pd.DataFrame | None:
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df.dropna() if not df.empty else None
        except Exception as e:
            self.logger.error(f"Volume download failed: {e}")
            return None

    def _compute_volume_metrics(self, df: pd.DataFrame) -> dict:
        close  = df["Close"]
        volume = df["Volume"]
        high   = df["High"]
        low    = df["Low"]
        metrics = {}

        # ── Volume spike detection (z-score) ──────────────────────────────
        vol_mean = volume.tail(20).mean()
        vol_std  = volume.tail(20).std()
        vol_z    = (volume.iloc[-1] - vol_mean) / vol_std if vol_std > 0 else 0
        metrics["volume_z_score"]   = round(float(vol_z), 2)
        metrics["is_unusual_volume"] = bool(abs(vol_z) > 2)
        metrics["volume_spike_type"] = (
            "bullish_spike" if vol_z > 2 and close.iloc[-1] > close.iloc[-2]
            else "bearish_spike" if vol_z > 2
            else "normal"
        )

        # ── OBV (On-Balance Volume) ────────────────────────────────────────
        price_diff = close.diff()
        obv = (np.sign(price_diff) * volume).fillna(0).cumsum()
        obv_ema   = obv.ewm(span=10).mean()
        metrics["obv_trend"]     = "rising" if obv.iloc[-1] > obv_ema.iloc[-1] else "falling"
        metrics["obv_vs_ema"]    = round(float(obv.iloc[-1] - obv_ema.iloc[-1]), 0)

        # ── VWAP (Volume-Weighted Average Price) ──────────────────────────
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).sum() / volume.sum()
        metrics["vwap"]              = round(float(vwap), 2)
        metrics["price_vs_vwap_pct"] = round((float(close.iloc[-1]) - float(vwap)) / float(vwap) * 100, 2)
        metrics["above_vwap"]        = bool(close.iloc[-1] > vwap)

        # ── Volume trend (5d vs 20d average) ──────────────────────────────
        avg_5d  = volume.tail(5).mean()
        avg_20d = volume.tail(20).mean()
        metrics["volume_trend"]  = "expanding" if avg_5d > avg_20d else "contracting"
        metrics["vol_5d_20d_ratio"] = round(float(avg_5d / avg_20d), 2)

        # ── Price-Volume divergence ────────────────────────────────────────
        price_chg_5d  = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100
        volume_chg_5d = (avg_5d - avg_20d) / avg_20d * 100
        metrics["price_vol_divergence"] = bool(
            (price_chg_5d > 0 and volume_chg_5d < -20)
            or (price_chg_5d < 0 and volume_chg_5d < -20)
        )

        return metrics

    def _score(self, metrics: dict, df: pd.DataFrame) -> tuple[float, str]:
        """Combine volume signals into a 0–100 bullish/bearish score."""
        score = 50.0   # start neutral

        # OBV trend
        if metrics.get("obv_trend") == "rising":  score += 15
        else:                                      score -= 15

        # Above VWAP
        if metrics.get("above_vwap"):  score += 10
        else:                          score -= 10

        # Volume expansion
        if metrics.get("volume_trend") == "expanding":  score += 10
        else:                                            score -= 5

        # Bullish spike
        if metrics.get("volume_spike_type") == "bullish_spike":  score += 15
        elif metrics.get("volume_spike_type") == "bearish_spike": score -= 15

        # Divergence penalty
        if metrics.get("price_vol_divergence"):  score -= 10

        score = max(0, min(100, score))

        if score >= 70:   label = "Strong Volume"
        elif score >= 55: label = "Good Volume"
        elif score >= 40: label = "Neutral"
        elif score >= 25: label = "Weak Volume"
        else:             label = "Volume Warning"

        return round(score, 2), label

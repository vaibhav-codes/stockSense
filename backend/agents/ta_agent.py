# =============================================================================
#  backend/agents/ta_agent.py
#  Technical Analysis Agent.
#  Downloads OHLCV price data and computes a composite TA score (0–100)
#  using RSI, MACD, Bollinger Bands, ADX, Stochastic, and EMA crossovers.
#  Score logic mirrors TradingView's Buy/Sell rating methodology.
# =============================================================================

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

from .base_agent import AgentOutput, BaseAgent
from config.settings import settings


class TechnicalAnalysisAgent(BaseAgent):
    """
    Computes technical indicators and produces a composite score (0–100).

    Score interpretation:
        0  – 20   → Strong Sell
        20 – 40   → Sell
        40 – 60   → Neutral
        60 – 80   → Buy
        80 – 100  → Strong Buy
    """

    def __init__(self):
        super().__init__("TechnicalAnalysis")

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(self, ticker: str, period: str = "6mo", interval: str = "1d") -> AgentOutput:

        # 1. Download price data
        df = self._download(ticker, period, interval)
        if df is None or len(df) < 30:
            return AgentOutput.failure(self.name, ticker, "Insufficient price data")

        # 2. Compute indicators
        indicators = self._compute_indicators(df)
        if not indicators:
            return AgentOutput.failure(self.name, ticker, "Indicator computation failed")

        # 3. Generate buy/sell signals from each indicator
        signals = self._evaluate_signals(indicators, df)

        # 4. Aggregate into composite score 0–100
        score, label, detail = self._aggregate_score(signals)

        return AgentOutput(
            agent_name = self.name,
            ticker     = ticker,
            score      = score,
            label      = label,
            confidence = min(1.0, len(signals) / 12),   # confidence scales with indicator count
            data = {
                "indicators": indicators,
                "signals":    signals,
                "detail":     detail,
                "price":      {
                    "current":  round(float(df["Close"].iloc[-1]), 2),
                    "open":     round(float(df["Open"].iloc[-1]), 2),
                    "high":     round(float(df["High"].iloc[-1]), 2),
                    "low":      round(float(df["Low"].iloc[-1]), 2),
                    "volume":   int(df["Volume"].iloc[-1]),
                    "change_pct": round(
                        (df["Close"].iloc[-1] - df["Close"].iloc[-2])
                        / df["Close"].iloc[-2] * 100, 2
                    ),
                },
            },
            metadata = {
                "period":   period,
                "interval": interval,
                "bars":     len(df),
            },
        )

    # ── Data download ─────────────────────────────────────────────────────────
    def _download(self, ticker: str, period: str, interval: str) -> pd.DataFrame | None:
        """Download OHLCV from yfinance. Returns None on failure."""
        try:
            df = yf.download(
                ticker,
                period   = period,
                interval = interval,
                auto_adjust = True,
                progress    = False,
            )
            if df.empty:
                return None
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df.dropna()
        except Exception as e:
            self.logger.error(f"Download failed for {ticker}: {e}")
            return None

    # ── Indicator computation ─────────────────────────────────────────────────
    def _compute_indicators(self, df: pd.DataFrame) -> dict:
        """
        Computes all technical indicators.
        Falls back to manual numpy calculations if pandas_ta is unavailable.
        """
        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]
        ind    = {}

        try:
            if PANDAS_TA_AVAILABLE:
                # ── RSI (14) ──────────────────────────────────────────────
                rsi = ta.rsi(close, length=14)
                ind["rsi"] = round(float(rsi.iloc[-1]), 2) if rsi is not None else None

                # ── MACD (12, 26, 9) ──────────────────────────────────────
                macd_df = ta.macd(close, fast=12, slow=26, signal=9)
                if macd_df is not None:
                    ind["macd"]        = round(float(macd_df["MACD_12_26_9"].iloc[-1]), 4)
                    ind["macd_signal"] = round(float(macd_df["MACDs_12_26_9"].iloc[-1]), 4)
                    ind["macd_hist"]   = round(float(macd_df["MACDh_12_26_9"].iloc[-1]), 4)

                # ── Bollinger Bands (20, 2) ────────────────────────────────
                bb = ta.bbands(close, length=20, std=2)
                if bb is not None:
                    ind["bb_upper"]  = round(float(bb["BBU_20_2.0"].iloc[-1]), 2)
                    ind["bb_middle"] = round(float(bb["BBM_20_2.0"].iloc[-1]), 2)
                    ind["bb_lower"]  = round(float(bb["BBL_20_2.0"].iloc[-1]), 2)
                    ind["bb_width"]  = round(float(bb["BBB_20_2.0"].iloc[-1]), 4)

                # ── Stochastic (14, 3) ────────────────────────────────────
                stoch = ta.stoch(high, low, close, k=14, d=3)
                if stoch is not None:
                    ind["stoch_k"] = round(float(stoch["STOCHk_14_3_3"].iloc[-1]), 2)
                    ind["stoch_d"] = round(float(stoch["STOCHd_14_3_3"].iloc[-1]), 2)

                # ── ADX (14) ──────────────────────────────────────────────
                adx = ta.adx(high, low, close, length=14)
                if adx is not None:
                    ind["adx"]    = round(float(adx["ADX_14"].iloc[-1]), 2)
                    ind["di_pos"] = round(float(adx["DMP_14"].iloc[-1]), 2)
                    ind["di_neg"] = round(float(adx["DMN_14"].iloc[-1]), 2)

                # ── EMAs ──────────────────────────────────────────────────
                for period in [9, 20, 50, 200]:
                    ema = ta.ema(close, length=period)
                    if ema is not None:
                        ind[f"ema_{period}"] = round(float(ema.iloc[-1]), 2)

                # ── ATR (Average True Range — volatility) ─────────────────
                atr = ta.atr(high, low, close, length=14)
                if atr is not None:
                    ind["atr"] = round(float(atr.iloc[-1]), 4)

                # ── Williams %R ───────────────────────────────────────────
                willr = ta.willr(high, low, close, length=14)
                if willr is not None:
                    ind["williams_r"] = round(float(willr.iloc[-1]), 2)

            else:
                # Fallback: manual RSI calculation
                ind["rsi"] = float(self._manual_rsi(close))

            # ── 52-week high/low (always computed manually) ────────────────
            ind["week52_high"] = round(float(high.tail(252).max()), 2)
            ind["week52_low"]  = round(float(low.tail(252).min()), 2)
            ind["current"]     = round(float(close.iloc[-1]), 2)

        except Exception as e:
            self.logger.warning(f"Partial indicator failure: {e}")

        return {k: v for k, v in ind.items() if v is not None}

    def _manual_rsi(self, close: pd.Series, period: int = 14) -> float:
        """Simple RSI calculation as fallback."""
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(period).mean()
        loss   = (-delta.clip(upper=0)).rolling(period).mean()
        rs     = gain / loss
        return float(100 - (100 / (1 + rs.iloc[-1])))

    # ── Signal evaluation ─────────────────────────────────────────────────────
    def _evaluate_signals(self, ind: dict, df: pd.DataFrame) -> dict:
        """
        Each indicator votes +1 (buy), -1 (sell), or 0 (neutral).
        Returns a dict of signal_name → vote.
        """
        signals = {}
        close   = float(df["Close"].iloc[-1])

        # RSI
        if (rsi := ind.get("rsi")) is not None:
            if rsi < 30:   signals["rsi"] = 1    # oversold → buy
            elif rsi > 70: signals["rsi"] = -1   # overbought → sell
            else:          signals["rsi"] = 0

        # MACD crossover
        if (macd := ind.get("macd")) is not None and (sig := ind.get("macd_signal")) is not None:
            signals["macd"] = 1 if macd > sig else -1

        # MACD histogram momentum
        if (hist := ind.get("macd_hist")) is not None:
            signals["macd_momentum"] = 1 if hist > 0 else -1

        # Bollinger Bands
        if all(k in ind for k in ["bb_upper", "bb_lower", "bb_middle"]):
            if close < ind["bb_lower"]:        signals["bollinger"] = 1
            elif close > ind["bb_upper"]:      signals["bollinger"] = -1
            else:                              signals["bollinger"] = 0

        # Stochastic
        if (k := ind.get("stoch_k")) is not None and (d := ind.get("stoch_d")) is not None:
            if k < 20 and d < 20:   signals["stochastic"] = 1
            elif k > 80 and d > 80: signals["stochastic"] = -1
            else:                   signals["stochastic"] = 0

        # ADX trend strength + direction
        if (adx := ind.get("adx")) is not None and adx > 25:
            di_pos = ind.get("di_pos", 0)
            di_neg = ind.get("di_neg", 0)
            signals["adx_trend"] = 1 if di_pos > di_neg else -1

        # EMA crossovers
        if (ema9 := ind.get("ema_9")) and (ema20 := ind.get("ema_20")):
            signals["ema_9_20"] = 1 if ema9 > ema20 else -1

        if (ema20 := ind.get("ema_20")) and (ema50 := ind.get("ema_50")):
            signals["ema_20_50"] = 1 if ema20 > ema50 else -1

        if (ema50 := ind.get("ema_50")) and (ema200 := ind.get("ema_200")):
            signals["ema_50_200"] = 1 if ema50 > ema200 else -1   # golden/death cross

        # Price vs EMAs
        if (ema20 := ind.get("ema_20")):
            signals["price_vs_ema20"] = 1 if close > ema20 else -1

        # Williams %R
        if (willr := ind.get("williams_r")) is not None:
            if willr < -80:   signals["williams_r"] = 1
            elif willr > -20: signals["williams_r"] = -1
            else:             signals["williams_r"] = 0

        # 52-week proximity
        if all(k in ind for k in ["week52_high", "week52_low"]):
            rng = ind["week52_high"] - ind["week52_low"]
            if rng > 0:
                pos = (close - ind["week52_low"]) / rng
                if pos > 0.85:   signals["week52"] = -1   # near 52w high — stretched
                elif pos < 0.15: signals["week52"] = 1    # near 52w low — oversold
                else:            signals["week52"] = 0

        return signals

    # ── Score aggregation ─────────────────────────────────────────────────────
    def _aggregate_score(self, signals: dict) -> tuple[float, str, dict]:
        """
        Converts signal votes into a 0–100 composite score.
        buy_pct = proportion of signals that voted buy → mapped to 0–100.
        """
        if not signals:
            return 50.0, "Neutral", {}

        votes      = list(signals.values())
        buy_count  = sum(1 for v in votes if v == 1)
        sell_count = sum(1 for v in votes if v == -1)
        total      = len(votes)

        # Map to 0–100: 100% buy votes = 100, 100% sell votes = 0
        score = ((buy_count - sell_count) / total + 1) / 2 * 100

        # Rating label
        if score >= 80:   label = "Strong Buy"
        elif score >= 60: label = "Buy"
        elif score >= 40: label = "Neutral"
        elif score >= 20: label = "Sell"
        else:             label = "Strong Sell"

        detail = {
            "buy_signals":  buy_count,
            "sell_signals": sell_count,
            "neutral":      total - buy_count - sell_count,
            "total":        total,
        }
        return round(score, 2), label, detail

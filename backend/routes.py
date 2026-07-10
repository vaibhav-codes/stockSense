# =============================================================================
#  backend/routes.py
#  Flask Blueprint containing all API routes.
#  Every route delegates to the orchestrator and returns JSON.
# =============================================================================

import logging
import pandas as pd
from flask import Blueprint, jsonify, request, current_app

from config.settings import settings

logger    = logging.getLogger(__name__)
api       = Blueprint("api", __name__, url_prefix="/api")


def get_orchestrator():
    """Get orchestrator from Flask app context."""
    return current_app.config["ORCHESTRATOR"]


# ── Health check ──────────────────────────────────────────────────────────────
@api.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


# ── Run full pipeline for a single ticker ─────────────────────────────────────
@api.route("/pipeline/<ticker>", methods=["GET"])
def run_pipeline(ticker: str):
    """
    GET /api/pipeline/RELIANCE.NS
    Runs the full multi-agent pipeline and returns all agent outputs.
    """
    use_cache = request.args.get("cache", "false").lower() == "true"
    try:
        state = get_orchestrator().run_full_pipeline(ticker.upper(), use_cache=use_cache)
        return jsonify(state.to_dict())
    except Exception as e:
        logger.error(f"Pipeline error for {ticker}: {e}")
        return jsonify({"error": str(e)}), 500


# ── Screener endpoint ─────────────────────────────────────────────────────────
@api.route("/screener", methods=["POST"])
def screener():
    """
    POST /api/screener
    Body: {"tickers": [...], "strategy": "swing_trading", "top_n": 10}
    Runs pipeline on multiple tickers and returns ranked results.
    """
    body     = request.get_json(force=True) or {}
    tickers  = body.get("tickers", settings.NIFTY50_TICKERS[:20])
    strategy = body.get("strategy", "all")
    top_n    = int(body.get("top_n", 10))

    try:
        results = get_orchestrator().run_screener(tickers, strategy, top_n)
        return jsonify({"results": results, "count": len(results), "strategy": strategy})
    except Exception as e:
        logger.error(f"Screener error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Individual agent endpoints ────────────────────────────────────────────────
@api.route("/ta/<ticker>", methods=["GET"])
def technical_analysis(ticker: str):
    """GET /api/ta/RELIANCE.NS — TA agent only."""
    try:
        output = get_orchestrator().run_ta_only(ticker.upper())
        return jsonify(output.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api.route("/sentiment/<ticker>", methods=["GET"])
def sentiment(ticker: str):
    """GET /api/sentiment/RELIANCE.NS — Sentiment agent only."""
    try:
        output = get_orchestrator().run_sentiment_only(ticker.upper())
        return jsonify(output.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api.route("/volume/<ticker>", methods=["GET"])
def volume(ticker: str):
    """GET /api/volume/RELIANCE.NS — Volume agent only."""
    try:
        output = get_orchestrator().run_volume_only(ticker.upper())
        return jsonify(output.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Chart data endpoint ───────────────────────────────────────────────────────
@api.route("/chart/<ticker>", methods=["GET"])
def chart_data(ticker: str):
    """
    GET /api/chart/RELIANCE.NS?period=6mo&interval=1d
    Returns OHLCV data + TA indicators for the chart terminal.
    """
    period   = request.args.get("period", "6mo")
    interval = request.args.get("interval", "1d")

    try:
        import yfinance as yf
        import pandas_ta as ta

        df = yf.download(
            ticker.upper(), period=period, interval=interval,
            auto_adjust=True, progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return jsonify({"error": "No data found"}), 404

        # Build OHLCV candles
        candles = []
        for idx, row in df.iterrows():
            candles.append({
                "time":   idx.strftime("%Y-%m-%d"),
                "open":   round(float(row["Open"]), 2),
                "high":   round(float(row["High"]), 2),
                "low":    round(float(row["Low"]), 2),
                "close":  round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })

        # Overlay indicators
        close  = df["Close"]
        rsi    = ta.rsi(close, length=14) if ta else None
        ema20  = ta.ema(close, length=20) if ta else None
        ema50  = ta.ema(close, length=50) if ta else None

        overlays = {
            "ema20":  [{"time": df.index[i].strftime("%Y-%m-%d"), "value": round(float(v), 2)}
                       for i, v in enumerate(ema20) if ema20 is not None and not pd.isna(v)],
            "ema50":  [{"time": df.index[i].strftime("%Y-%m-%d"), "value": round(float(v), 2)}
                       for i, v in enumerate(ema50) if ema50 is not None and not pd.isna(v)],
            "rsi":    [{"time": df.index[i].strftime("%Y-%m-%d"), "value": round(float(v), 2)}
                       for i, v in enumerate(rsi) if rsi is not None and not pd.isna(v)],
        }

        return jsonify({
            "ticker":   ticker.upper(),
            "period":   period,
            "interval": interval,
            "candles":  candles,
            "overlays": overlays,
        })

    except Exception as e:
        logger.error(f"Chart data error for {ticker}: {e}")
        return jsonify({"error": str(e)}), 500


# ── Index constituents ────────────────────────────────────────────────────────
@api.route("/indices", methods=["GET"])
def indices():
    """GET /api/indices — returns available Indian indices and their tickers."""
    return jsonify({
        "indices": {
            "NIFTY50":    settings.NIFTY50_TICKERS,
            "DEFAULT":    settings.DEFAULT_TICKERS,
        }
    })


# ── Cache management ──────────────────────────────────────────────────────────
@api.route("/cache/clear", methods=["POST"])
def clear_cache():
    get_orchestrator().clear_cache()
    return jsonify({"status": "cleared"})


@api.route("/cache/keys", methods=["GET"])
def cache_keys():
    return jsonify({"cached_tickers": get_orchestrator().get_cache_keys()})




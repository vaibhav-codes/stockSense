# =============================================================================
#  config/settings.py
#  Central configuration loader.
#  Reads from .env file and exposes a typed Settings object used everywhere.
# =============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    """
    Single source of truth for all configuration.
    Access anywhere via: from config.settings import settings
    """

    # ── Project paths ──────────────────────────────────────────────────────
    BASE_DIR        = BASE_DIR
    BACKEND_DIR     = BASE_DIR / "backend"
    FRONTEND_DIR    = BASE_DIR / "frontend"
    LOGS_DIR        = BASE_DIR / "logs"
    EXPORTS_DIR     = BASE_DIR / "exports"

    # ── Flask ──────────────────────────────────────────────────────────────
    SECRET_KEY      = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
    FLASK_ENV       = os.getenv("FLASK_ENV", "development")
    DEBUG           = os.getenv("FLASK_DEBUG", "True") == "True"
    PORT            = int(os.getenv("FLASK_PORT", 5000))
    HOST            = os.getenv("FLASK_HOST", "0.0.0.0")

    # ── Groq LLM ───────────────────────────────────────────────────────────
    GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama3-70b-8192")

    # ── News APIs ──────────────────────────────────────────────────────────
    NEWS_API_KEY    = os.getenv("NEWS_API_KEY", "")
    GNEWS_API_KEY   = os.getenv("GNEWS_API_KEY", "")

    # ── Redis / Cache ──────────────────────────────────────────────────────
    REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    USE_REDIS       = os.getenv("USE_REDIS", "False") == "True"

    # ── Scheduler ──────────────────────────────────────────────────────────
    SCHEDULER_ENABLED           = os.getenv("SCHEDULER_ENABLED", "True") == "True"
    MARKET_OPEN_TIME            = os.getenv("MARKET_OPEN_TIME", "09:15")
    MARKET_CLOSE_TIME           = os.getenv("MARKET_CLOSE_TIME", "15:30")
    PIPELINE_INTERVAL_MINUTES   = int(os.getenv("PIPELINE_INTERVAL_MINUTES", 15))

    # ── Models ─────────────────────────────────────────────────────────────
    FINBERT_MODEL           = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
    LIGHTGBM_MODEL_PATH     = BASE_DIR / os.getenv("LIGHTGBM_MODEL_PATH", "backend/models/signal_model.pkl")

    # ── Indian Market Data ─────────────────────────────────────────────────
    DEFAULT_TICKERS = os.getenv(
        "DEFAULT_TICKERS",
        "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS"
    ).split(",")
    DEFAULT_INDEX           = os.getenv("DEFAULT_INDEX", "^NSEI")
    PRICE_HISTORY_PERIOD    = os.getenv("PRICE_HISTORY_PERIOD", "1y")

    # ── Nifty 50 constituent tickers ──────────────────────────────────────
    NIFTY50_TICKERS = [
        "RELIANCE.NS", "TCS.NS",       "HDFCBANK.NS",  "INFY.NS",      "ICICIBANK.NS",
        "HINDUNILVR.NS","ITC.NS",      "SBIN.NS",      "BHARTIARTL.NS","KOTAKBANK.NS",
        "LT.NS",       "BAJFINANCE.NS","HCLTECH.NS",   "WIPRO.NS",     "ASIANPAINT.NS",
        "AXISBANK.NS", "MARUTI.NS",    "SUNPHARMA.NS", "TITAN.NS",     "ULTRACEMCO.NS",
        "NESTLEIND.NS","BAJAJFINSV.NS","TECHM.NS",     "POWERGRID.NS", "NTPC.NS",
        "ONGC.NS",     "JSWSTEEL.NS",  "TATAMOTORS.NS","ADANIENT.NS",  "ADANIPORTS.NS",
        "CIPLA.NS",    "DRREDDY.NS",   "EICHERMOT.NS", "GRASIM.NS",    "HEROMOTOCO.NS",
        "HINDALCO.NS", "INDUSINDBK.NS","M&M.NS",       "SBILIFE.NS",   "TATACONSUM.NS",
        "TATASTEEL.NS","BRITANNIA.NS", "COALINDIA.NS", "DIVISLAB.NS",  "HDFCLIFE.NS",
        "BPCL.NS",     "APOLLOHOSP.NS","UPL.NS",       "SHREECEM.NS",  "BAJAJ-AUTO.NS",
    ]

    # ── Reviewer Agent thresholds ──────────────────────────────────────────
    REVIEWER_MIN_HEADLINES      = 2       # minimum headlines for sentiment to be trusted
    REVIEWER_MAX_SCORE_JUMP     = 30      # max allowed score change between runs
    REVIEWER_MAX_NEWS_AGE_HOURS = 48      # headlines older than this are stale
    REVIEWER_MIN_CONFIDENCE     = 0.45   # minimum model confidence to PASS


# Singleton instance imported everywhere
settings = Settings()

# Ensure required directories exist
for directory in [settings.LOGS_DIR, settings.EXPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

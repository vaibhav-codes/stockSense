# =============================================================================
#  app.py
#  StockSense — Main Flask Application Entry Point
#
#  Run with:  python app.py
#  The app will auto-open http://localhost:5000 in your browser.
#
#  Architecture:
#    Flask app → Blueprint(api) for REST endpoints
#               Blueprint(views) for page routes
#    Orchestrator is created once and stored in app.config["ORCHESTRATOR"]
#    APScheduler runs market-hours auto-refresh in background
# =============================================================================

import os
import sys
import logging
import threading
import webbrowser
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for

# ── Ensure project root is on Python path ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import settings
from backend.routes import api
from backend.orchestrator import StockSenseOrchestrator
from backend.data.scheduler import MarketScheduler


# ── Logging configuration ─────────────────────────────────────────────────────
def configure_logging():
    """Set up structured logging to both console and file."""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_level  = logging.DEBUG if settings.DEBUG else logging.INFO

    logging.basicConfig(
        level   = log_level,
        format  = log_format,
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(settings.LOGS_DIR / "stocksense.log", encoding="utf-8"),
        ],
    )
    # Quiet noisy third-party loggers
    for noisy in ["urllib3", "yfinance", "peewee", "apscheduler"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("app")


# ── Flask application factory ─────────────────────────────────────────────────
def create_app() -> Flask:
    """
    Application factory.
    Creates and configures the Flask app, registers blueprints,
    initialises the orchestrator, and wires up the scheduler.
    """
    app = Flask(
        __name__,
        template_folder = str(settings.FRONTEND_DIR / "templates"),
        static_folder   = str(settings.FRONTEND_DIR / "static"),
        static_url_path = "/static",
    )

    # ── Flask config ──────────────────────────────────────────────────────
    app.config["SECRET_KEY"]   = settings.SECRET_KEY
    app.config["DEBUG"]        = settings.DEBUG
    app.config["JSON_SORT_KEYS"] = False  # preserve field order in API responses

    # ── Initialise orchestrator (singleton for the app lifetime) ──────────
    logger.info("Initialising StockSense Orchestrator…")
    orchestrator = StockSenseOrchestrator()
    app.config["ORCHESTRATOR"] = orchestrator

    # ── Register API blueprint (/api/...) ─────────────────────────────────
    app.register_blueprint(api)

    # ── Register page-view routes ─────────────────────────────────────────
    register_views(app)

    # ── Global error handlers ─────────────────────────────────────────────
    register_error_handlers(app)

    # ── Start scheduler ───────────────────────────────────────────────────
    scheduler = MarketScheduler(orchestrator)
    scheduler.start()
    app.config["SCHEDULER"] = scheduler

    logger.info("Flask app created and configured")
    return app


# ── Page-view routes (render HTML templates) ─────────────────────────────────
def register_views(app: Flask):
    """
    Register all HTML page routes as a Blueprint.
    Each route maps a URL to its Jinja2 template.
    Query params (e.g. ?ticker=TCS.NS) are passed to the template
    so JS can auto-populate inputs on page load.
    """
    from flask import Blueprint
    views = Blueprint("views", __name__)

    @views.route("/")
    def dashboard():
        return render_template("dashboard.html")

    @views.route("/chart")
    def chart_terminal():
        # Optional ?ticker= param pre-fills the chart input
        ticker = request.args.get("ticker", "TCS.NS")
        return render_template("chart.html", prefill_ticker=ticker)

    @views.route("/screener")
    def screener():
        return render_template("screener.html")

    @views.route("/sentiment")
    def sentiment_view():
        ticker = request.args.get("ticker", "")
        return render_template("sentiment.html", prefill_ticker=ticker)

    @views.route("/signals")
    def signals():
        return render_template("signals.html")

    app.register_blueprint(views)


# ── Global error handlers ─────────────────────────────────────────────────────
def register_error_handlers(app: Flask):
    @app.errorhandler(404)
    def not_found(e):
        return {"error": "Not found", "status": 404}, 404

    @app.errorhandler(500)
    def server_error(e):
        return {"error": "Internal server error", "status": 500}, 500


# ── Auto-open browser ─────────────────────────────────────────────────────────
def open_browser(host: str, port: int, delay: float = 1.5):
    """
    Opens the default browser after a short delay (so Flask has time to start).
    Runs in a daemon thread so it never blocks the main process.
    """
    def _open():
        import time
        time.sleep(delay)
        url = f"http://localhost:{port}"
        logger.info(f"Opening browser → {url}")
        webbrowser.open(url)

    t = threading.Thread(target=_open, daemon=True)
    t.start()


# ── Startup banner ────────────────────────────────────────────────────────────
def print_banner(host: str, port: int):
    banner = f"""
╔══════════════════════════════════════════════════════════╗
║                  📈  StockSense AI                       ║
║         Multi-Agent Indian Equity Analysis               ║
╠══════════════════════════════════════════════════════════╣
║  Dashboard   →  http://localhost:{port}/                  ║
║  Chart       →  http://localhost:{port}/chart             ║
║  Screener    →  http://localhost:{port}/screener          ║
║  Sentiment   →  http://localhost:{port}/sentiment         ║
║  Signals     →  http://localhost:{port}/signals           ║
║  API Health  →  http://localhost:{port}/api/health        ║
╠══════════════════════════════════════════════════════════╣
║  Press Ctrl+C to stop                                    ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


# ── Main entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger = configure_logging()

    logger.info("=" * 60)
    logger.info("Starting StockSense AI — Multi-Agent Stock Analysis")
    logger.info("=" * 60)

    # Validate critical config
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your-groq-api-key-here":
        logger.warning("⚠  GROQ_API_KEY not set — sentiment fallback will be disabled")
    if not settings.NEWS_API_KEY or settings.NEWS_API_KEY == "your-newsapi-key-here":
        logger.warning("⚠  NEWS_API_KEY not set — news fetching will be disabled")

    # Create the Flask app
    app = create_app()

    # Print startup banner
    print_banner(settings.HOST, settings.PORT)

    # Auto-open browser (skip in testing / CI environments)
    if os.environ.get("NO_BROWSER") != "1":
        open_browser(settings.HOST, settings.PORT)

    # Run Flask
    app.run(
        host        = settings.HOST,
        port        = settings.PORT,
        debug       = settings.DEBUG,
        use_reloader = False,   # reloader conflicts with APScheduler background thread
    )
else:
    # When imported as a module (e.g. gunicorn), use module-level logger
    logger = logging.getLogger("app")
    app    = create_app()

# =============================================================================
#  backend/data/scheduler.py
#  APScheduler-based real-time pipeline scheduler.
#  Runs the full pipeline automatically during NSE market hours (IST).
#  Results are cached so the UI always shows fresh data.
# =============================================================================

import logging
import pytz
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import settings

logger = logging.getLogger(__name__)
IST    = pytz.timezone("Asia/Kolkata")


class MarketScheduler:
    """
    Manages scheduled pipeline runs.
    The orchestrator is injected to avoid circular imports.
    """

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.scheduler    = BackgroundScheduler(timezone=IST)
        self._is_running  = False

    def start(self):
        """Start the scheduler with market-hours jobs."""
        if not settings.SCHEDULER_ENABLED:
            logger.info("[Scheduler] Disabled via config")
            return

        # Run pipeline every N minutes during market hours
        self.scheduler.add_job(
            func    = self._scheduled_run,
            trigger = IntervalTrigger(minutes=settings.PIPELINE_INTERVAL_MINUTES),
            id      = "market_pipeline",
            name    = "Market pipeline refresh",
            replace_existing = True,
        )

        # Clear cache at market open (9:15 IST)
        self.scheduler.add_job(
            func    = self.orchestrator.clear_cache,
            trigger = CronTrigger(hour=9, minute=15, timezone=IST),
            id      = "cache_clear",
            name    = "Clear cache at market open",
        )

        self.scheduler.start()
        self._is_running = True
        logger.info(
            f"[Scheduler] Started — running every {settings.PIPELINE_INTERVAL_MINUTES} min"
        )

    def stop(self):
        if self._is_running:
            self.scheduler.shutdown(wait=False)
            self._is_running = False
            logger.info("[Scheduler] Stopped")

    def _scheduled_run(self):
        """Run pipeline for default tickers if market is open."""
        if not self._is_market_hours():
            return
        logger.info("[Scheduler] Triggered pipeline refresh")
        for ticker in settings.DEFAULT_TICKERS[:5]:   # top 5 to stay fast
            try:
                self.orchestrator.run_full_pipeline(ticker, use_cache=False)
            except Exception as e:
                logger.error(f"[Scheduler] Failed for {ticker}: {e}")

    def _is_market_hours(self) -> bool:
        """Return True if current IST time is within NSE trading hours."""
        now   = datetime.now(IST)
        open_h, open_m   = map(int, settings.MARKET_OPEN_TIME.split(":"))
        close_h, close_m = map(int, settings.MARKET_CLOSE_TIME.split(":"))
        market_open  = now.replace(hour=open_h,  minute=open_m,  second=0)
        market_close = now.replace(hour=close_h, minute=close_m, second=0)
        is_weekday   = now.weekday() < 5
        return is_weekday and market_open <= now <= market_close

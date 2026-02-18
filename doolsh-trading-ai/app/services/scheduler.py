"""APScheduler-based background job scheduler replacing Celery.

Runs the auto-trading cycle on a configurable interval during market hours.
Lightweight enough for Userland / Android.
"""

from __future__ import annotations

import logging
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.core.config import get_settings
from app.services.auto_trader import is_auto_trading_enabled, is_market_open, trading_cycle

logger = logging.getLogger(__name__)
settings = get_settings()

scheduler = AsyncIOScheduler()

# Runtime config (set via API)
_model_type = "rf"
_model_version = "v1"


def set_scheduler_model(model_type: str, version: str) -> None:
    global _model_type, _model_version
    _model_type = model_type
    _model_version = version


async def _scheduled_cycle() -> None:
    """Runs every scheduler_interval_seconds during market hours."""
    if not is_auto_trading_enabled():
        return
    if not is_market_open():
        return

    model_dir = Path(settings.model_save_dir)
    if _model_type == "rf":
        mp = str(model_dir / f"rf_{settings.watchlist_symbols[0]}_{_model_version}.pkl")
    else:
        mp = str(model_dir / f"lstm_{settings.watchlist_symbols[0]}_{_model_version}.pt")

    try:
        summary = await trading_cycle(model_path=mp, model_type=_model_type)
        orders = summary.get("orders", [])
        errors = summary.get("errors", [])
        if orders:
            logger.info("Cycle completed: %d orders placed", len(orders))
        if errors:
            logger.warning("Cycle errors: %s", errors)
    except Exception:
        logger.exception("Scheduled trading cycle failed")


def start_scheduler() -> None:
    if scheduler.running:
        return
    scheduler.add_job(
        _scheduled_cycle,
        "interval",
        seconds=settings.scheduler_interval_seconds,
        id="trading_cycle",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started (interval=%ds)", settings.scheduler_interval_seconds)


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

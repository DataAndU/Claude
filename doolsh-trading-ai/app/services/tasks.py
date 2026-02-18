"""Celery background tasks for model training and data refresh."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from app.core.celery_app import celery_app
from app.services.market_data import fetch_daily_prices
from ml.features.engineering import build_features
from ml.training.lstm_model import train_lstm
from ml.training.random_forest import train_random_forest

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from a sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def task_train_model(
    self,
    symbol: str,
    model_type: str = "rf",
    horizon: int = 5,
    version: str = "v1",
    hyperparameters: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Background task: fetch data, engineer features, train model."""
    hp = hyperparameters or {}
    try:
        df = _run_async(fetch_daily_prices(symbol, use_cache=False))
        featured = build_features(df)

        if model_type == "rf":
            result = train_random_forest(
                featured,
                symbol=symbol,
                horizon=horizon,
                n_estimators=hp.get("n_estimators", 200),
                max_depth=hp.get("max_depth", 12),
                version=version,
            )
        else:
            result = train_lstm(
                featured,
                symbol=symbol,
                horizon=horizon,
                seq_len=hp.get("seq_len", 30),
                hidden_size=hp.get("hidden_size", 128),
                num_layers=hp.get("num_layers", 2),
                epochs=hp.get("epochs", 50),
                batch_size=hp.get("batch_size", 64),
                lr=hp.get("lr", 1e-3),
                version=version,
            )
        logger.info("Training complete for %s/%s/%s", model_type, symbol, version)
        return result

    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def task_refresh_market_data(self, symbol: str) -> Dict[str, Any]:
    """Background task: refresh cached market data for a symbol."""
    try:
        df = _run_async(fetch_daily_prices(symbol, use_cache=False))
        return {"symbol": symbol, "rows": len(df)}
    except Exception as exc:
        logger.exception("Data refresh failed for %s: %s", symbol, exc)
        raise self.retry(exc=exc)

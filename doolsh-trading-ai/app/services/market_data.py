"""Market data service — Zerodha Kite historical + live data.

Replaces the old Alpha Vantage integration. Uses Kite Connect's
historical data API for OHLCV candles and the ticker for live quotes.
Falls back to in-memory cache (no Redis needed).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from app.core.config import get_settings
from app.core.kite import get_kite, is_logged_in
from app.core.redis import get_redis

logger = logging.getLogger(__name__)
settings = get_settings()

# NSE instrument token cache (loaded once)
_instrument_cache: dict[str, int] = {}


async def _load_instruments() -> None:
    """Cache NSE instrument tokens from Kite."""
    global _instrument_cache
    if _instrument_cache:
        return
    kite = get_kite()
    instruments = kite.instruments(settings.trading_exchange)
    _instrument_cache = {i["tradingsymbol"]: i["instrument_token"] for i in instruments}
    logger.info("Loaded %d %s instruments", len(_instrument_cache), settings.trading_exchange)


def get_instrument_token(symbol: str) -> int:
    token = _instrument_cache.get(symbol)
    if token is None:
        raise ValueError(f"Instrument not found: {symbol} on {settings.trading_exchange}")
    return token


async def fetch_daily_prices(
    symbol: str,
    days: int = 365,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily OHLCV candles from Kite historical data API."""
    cache_key = f"kite:daily:{symbol}:{days}"
    cache = await get_redis()

    if use_cache:
        cached = await cache.get(cache_key)
        if cached:
            return pd.DataFrame(json.loads(cached))

    if not is_logged_in():
        raise RuntimeError("Kite not logged in. Call /api/v1/kite/auto-login first.")

    await _load_instruments()
    token = get_instrument_token(symbol)
    kite = get_kite()

    to_date = datetime.now(timezone.utc)
    from_date = to_date - timedelta(days=days)

    candles = kite.historical_data(
        instrument_token=token,
        from_date=from_date.strftime("%Y-%m-%d"),
        to_date=to_date.strftime("%Y-%m-%d"),
        interval="day",
    )

    df = pd.DataFrame(candles)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    df.rename(columns={"date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    await cache.setex(cache_key, 300, df.to_json(orient="records", date_format="iso"))
    return df


async def fetch_intraday_prices(
    symbol: str,
    interval: str = "15minute",
    days: int = 5,
) -> pd.DataFrame:
    """Fetch intraday OHLCV candles from Kite."""
    if not is_logged_in():
        raise RuntimeError("Kite not logged in.")

    await _load_instruments()
    token = get_instrument_token(symbol)
    kite = get_kite()

    to_date = datetime.now(timezone.utc)
    from_date = to_date - timedelta(days=days)

    candles = kite.historical_data(
        instrument_token=token,
        from_date=from_date.strftime("%Y-%m-%d %H:%M:%S"),
        to_date=to_date.strftime("%Y-%m-%d %H:%M:%S"),
        interval=interval,
    )

    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


async def get_live_quote(symbol: str) -> dict:
    """Get the latest live quote via Kite."""
    if not is_logged_in():
        raise RuntimeError("Kite not logged in.")
    kite = get_kite()
    key = f"{settings.trading_exchange}:{symbol}"
    quotes = kite.quote([key])
    return quotes.get(key, {})


async def get_ltp(symbols: list[str]) -> dict[str, float]:
    """Get last traded price for multiple symbols."""
    if not is_logged_in():
        raise RuntimeError("Kite not logged in.")
    kite = get_kite()
    keys = [f"{settings.trading_exchange}:{s}" for s in symbols]
    data = kite.ltp(keys)
    return {
        s: data.get(f"{settings.trading_exchange}:{s}", {}).get("last_price", 0.0)
        for s in symbols
    }

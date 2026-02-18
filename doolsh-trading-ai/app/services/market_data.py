"""Market data fetching service (Alpha Vantage integration)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
import pandas as pd

from app.core.config import get_settings
from app.core.redis import get_redis

logger = logging.getLogger(__name__)
settings = get_settings()

ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"


async def fetch_daily_prices(
    symbol: str,
    outputsize: str = "full",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily OHLCV data from Alpha Vantage, with Redis caching."""
    cache_key = f"market:daily:{symbol}:{outputsize}"

    if use_cache:
        redis = await get_redis()
        cached = await redis.get(cache_key)
        if cached:
            logger.debug("Cache hit for %s", cache_key)
            data = json.loads(cached)
            return pd.DataFrame(data)

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": settings.alpha_vantage_api_key,
        "datatype": "json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(ALPHA_VANTAGE_BASE, params=params)
        resp.raise_for_status()
        payload = resp.json()

    ts_key = "Time Series (Daily)"
    if ts_key not in payload:
        error_msg = payload.get("Note") or payload.get("Error Message") or str(payload)
        raise ValueError(f"Alpha Vantage error for {symbol}: {error_msg}")

    records = []
    for date_str, ohlcv in payload[ts_key].items():
        records.append(
            {
                "date": date_str,
                "open": float(ohlcv["1. open"]),
                "high": float(ohlcv["2. high"]),
                "low": float(ohlcv["3. low"]),
                "close": float(ohlcv["4. close"]),
                "volume": int(ohlcv["5. volume"]),
            }
        )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if use_cache:
        redis = await get_redis()
        await redis.setex(
            cache_key,
            settings.data_cache_ttl_seconds,
            df.to_json(orient="records", date_format="iso"),
        )

    return df


async def fetch_intraday_prices(
    symbol: str,
    interval: str = "15min",
    outputsize: str = "full",
) -> pd.DataFrame:
    """Fetch intraday OHLCV data from Alpha Vantage."""
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": settings.alpha_vantage_api_key,
        "datatype": "json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(ALPHA_VANTAGE_BASE, params=params)
        resp.raise_for_status()
        payload = resp.json()

    ts_key = f"Time Series ({interval})"
    if ts_key not in payload:
        error_msg = payload.get("Note") or payload.get("Error Message") or str(payload)
        raise ValueError(f"Alpha Vantage error for {symbol}: {error_msg}")

    records = []
    for dt_str, ohlcv in payload[ts_key].items():
        records.append(
            {
                "datetime": dt_str,
                "open": float(ohlcv["1. open"]),
                "high": float(ohlcv["2. high"]),
                "low": float(ohlcv["3. low"]),
                "close": float(ohlcv["4. close"]),
                "volume": int(ohlcv["5. volume"]),
            }
        )

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

"""Market data engine — fetches OHLCV via ccxt (crypto) and yfinance (equities).

Supports multiple timeframes, local caching to SQLite, and synthetic data
for paper mode fallback. Designed for low-memory environments.
"""

from __future__ import annotations

import gc
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── ccxt (crypto) ────────────────────────────────────────────────────────────

_exchange = None


def _get_exchange(exchange_id: str = "binance"):
    """Lazy-load exchange to save memory."""
    global _exchange
    if _exchange is None:
        import ccxt
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Exchange '{exchange_id}' not found in ccxt")
        _exchange = exchange_class({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        logger.info("Initialized %s exchange", exchange_id)
    return _exchange


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 500,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """Fetch OHLCV candles from a crypto exchange via ccxt.

    Returns DataFrame with columns: [timestamp, open, high, low, close, volume].
    """
    try:
        ex = _get_exchange(exchange_id)
        since = None
        if limit > 0:
            tf_ms = _timeframe_to_ms(timeframe)
            since = int((datetime.now(timezone.utc) - timedelta(milliseconds=tf_ms * limit)).timestamp() * 1000)

        raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not raw:
            logger.warning("No data returned for %s %s", symbol, timeframe)
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")

        logger.info("Fetched %d candles for %s %s", len(df), symbol, timeframe)
        return df

    except Exception as e:
        logger.error("ccxt fetch failed for %s %s: %s", symbol, timeframe, e)
        return pd.DataFrame()


def fetch_multi_timeframe(
    symbol: str,
    timeframes: List[str],
    limit: int = 500,
    exchange_id: str = "binance",
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple timeframes."""
    result = {}
    for tf in timeframes:
        df = fetch_ohlcv(symbol, timeframe=tf, limit=limit, exchange_id=exchange_id)
        if not df.empty:
            result[tf] = df
        time.sleep(0.5)  # rate limit courtesy
    return result


def fetch_ticker(symbol: str, exchange_id: str = "binance") -> Dict:
    """Fetch current ticker (last price, bid, ask)."""
    try:
        ex = _get_exchange(exchange_id)
        return ex.fetch_ticker(symbol)
    except Exception as e:
        logger.error("Ticker fetch failed for %s: %s", symbol, e)
        return {}


# ─── yfinance (equities, optional) ────────────────────────────────────────────


def fetch_equity_data(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch equity OHLCV via yfinance. Optional — fails gracefully."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    except ImportError:
        logger.warning("yfinance not installed — equity data unavailable")
        return pd.DataFrame()
    except Exception as e:
        logger.error("yfinance fetch failed for %s: %s", symbol, e)
        return pd.DataFrame()


# ─── Synthetic data (paper mode fallback) ─────────────────────────────────────


def generate_synthetic(symbol: str = "SYN/USDT", bars: int = 500) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV for paper mode testing."""
    np.random.seed(hash(symbol) % 2**31)
    base_price = np.random.uniform(100, 50000)
    returns = np.random.normal(0, 0.015, bars)
    close = base_price * np.cumprod(1 + returns)

    timestamps = pd.date_range(end=datetime.now(timezone.utc), periods=bars, freq="h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": close * (1 + np.random.uniform(-0.005, 0.005, bars)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.01, bars))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.01, bars))),
        "close": close,
        "volume": np.random.uniform(100, 10000, bars),
    })
    return df


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _timeframe_to_ms(tf: str) -> int:
    """Convert timeframe string to milliseconds."""
    unit = tf[-1]
    val = int(tf[:-1]) if len(tf) > 1 else 1
    multipliers = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    return val * multipliers.get(unit, 3_600_000)

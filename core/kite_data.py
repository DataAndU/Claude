"""Market data fetching via Kite Connect + yfinance fallback.

Provides OHLCV data for Indian equities (NSE/BSE).
Falls back to yfinance when Kite is not connected (paper mode).
Generates synthetic data as last resort for testing.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import load_config
from core.kite_auth import get_kite, is_logged_in

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# In-memory LTP cache
_ltp_cache: Dict[str, float] = {}


async def fetch_historical(
    symbol: str,
    days: int = 180,
    interval: str = "day",
    exchange: str = "NSE",
) -> pd.DataFrame:
    """Fetch historical OHLCV data for a symbol.

    Tries Kite Connect first, then yfinance, then synthetic.
    Returns DataFrame with columns: date, open, high, low, close, volume.
    """
    cfg = load_config()

    # Try Kite Connect
    if is_logged_in():
        try:
            return await _fetch_kite_historical(symbol, days, interval, exchange)
        except Exception as e:
            logger.warning("Kite historical fetch failed for %s: %s", symbol, e)

    # Fallback to yfinance
    try:
        return _fetch_yfinance(symbol, days, interval, exchange)
    except Exception as e:
        logger.warning("yfinance fetch failed for %s: %s", symbol, e)

    # Last resort: synthetic data
    logger.info("Using synthetic data for %s", symbol)
    return _generate_synthetic(symbol, days)


async def _fetch_kite_historical(
    symbol: str, days: int, interval: str, exchange: str,
) -> pd.DataFrame:
    """Fetch from Kite Connect historical data API."""
    kite = get_kite()

    # Resolve instrument token
    instrument = f"{exchange}:{symbol}"
    instruments = kite.ltp([instrument])
    if instrument not in instruments:
        raise ValueError(f"Instrument {instrument} not found")

    token = instruments[instrument].get("instrument_token")
    if not token:
        # Try fetching from instruments list
        all_instruments = kite.instruments(exchange)
        for inst in all_instruments:
            if inst["tradingsymbol"] == symbol:
                token = inst["instrument_token"]
                break

    if not token:
        raise ValueError(f"Could not resolve instrument token for {symbol}")

    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    records = kite.historical_data(
        instrument_token=token,
        from_date=from_date.strftime("%Y-%m-%d"),
        to_date=to_date.strftime("%Y-%m-%d"),
        interval=interval,
    )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    df.rename(columns={"date": "date"}, inplace=True)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _fetch_yfinance(
    symbol: str, days: int, interval: str, exchange: str,
) -> pd.DataFrame:
    """Fetch from yfinance as fallback."""
    import yfinance as yf

    # Map NSE symbols to yfinance format
    ticker = f"{symbol}.NS" if exchange == "NSE" else f"{symbol}.BO"
    period_map = {
        30: "1mo", 60: "2mo", 90: "3mo",
        180: "6mo", 365: "1y", 730: "2y",
    }
    period = "6mo"
    for d, p in sorted(period_map.items()):
        if days <= d:
            period = p
            break

    data = yf.download(ticker, period=period, interval="1d", progress=False)
    if data.empty:
        raise ValueError(f"No yfinance data for {ticker}")

    df = data.reset_index()
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]

    # Normalize column names
    col_map = {"adj close": "adj_close", "date": "date"}
    df.rename(columns=col_map, inplace=True)

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["date", "open", "high", "low", "close", "volume"]].dropna()


def _generate_synthetic(symbol: str, days: int = 180) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for testing."""
    np.random.seed(hash(symbol) % 2**32)

    # Base price per symbol category
    base_prices = {
        "RELIANCE": 2500, "TCS": 3800, "INFY": 1500, "HDFCBANK": 1600,
        "ICICIBANK": 1000, "SBIN": 600, "BAJFINANCE": 7000, "ITC": 450,
        "HINDUNILVR": 2500, "KOTAKBANK": 1800, "TATAMOTORS": 800,
        "MARUTI": 11000, "AXISBANK": 1100, "LT": 3500, "SUNPHARMA": 1200,
        "TITAN": 3200, "BHARTIARTL": 1400, "WIPRO": 450, "HCLTECH": 1500,
        "ADANIENT": 2800,
    }
    base = base_prices.get(symbol, 1000 + random.randint(0, 2000))

    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")  # business days
    prices = [base]
    volumes = []

    for i in range(1, len(dates)):
        # Random walk with mean reversion
        drift = 0.0002
        volatility = 0.015
        mean_revert = -0.01 * (prices[-1] - base) / base
        ret = drift + mean_revert + volatility * np.random.randn()
        prices.append(prices[-1] * (1 + ret))
        volumes.append(int(np.random.lognormal(14, 0.5)))

    volumes.insert(0, int(np.random.lognormal(14, 0.5)))

    rows = []
    for i, (dt, close) in enumerate(zip(dates, prices)):
        intraday_vol = abs(np.random.randn()) * 0.01
        high = close * (1 + intraday_vol)
        low = close * (1 - intraday_vol)
        opn = low + (high - low) * random.random()
        rows.append({
            "date": dt, "open": round(opn, 2), "high": round(high, 2),
            "low": round(low, 2), "close": round(close, 2),
            "volume": volumes[i],
        })

    return pd.DataFrame(rows)


async def get_ltp(symbols: List[str], exchange: str = "NSE") -> Dict[str, float]:
    """Get last traded prices for symbols."""
    if is_logged_in():
        try:
            kite = get_kite()
            instruments = [f"{exchange}:{s}" for s in symbols]
            data = kite.ltp(instruments)
            return {
                s: data[f"{exchange}:{s}"]["last_price"]
                for s in symbols
                if f"{exchange}:{s}" in data
            }
        except Exception as e:
            logger.warning("LTP fetch failed: %s", e)

    # Fallback: synthetic LTP based on cached or random prices
    result = {}
    for s in symbols:
        if s in _ltp_cache:
            change = random.uniform(-0.02, 0.02)
            _ltp_cache[s] *= (1 + change)
        else:
            _ltp_cache[s] = 1000 + random.random() * 3000
        result[s] = round(_ltp_cache[s], 2)
    return result


async def get_quote(symbols: List[str], exchange: str = "NSE") -> Dict[str, dict]:
    """Get full quote data (OHLCV, volume, etc.) for symbols."""
    if is_logged_in():
        try:
            kite = get_kite()
            instruments = [f"{exchange}:{s}" for s in symbols]
            return kite.quote(instruments)
        except Exception as e:
            logger.warning("Quote fetch failed: %s", e)

    # Synthetic quotes
    ltps = await get_ltp(symbols, exchange)
    return {
        s: {
            "last_price": ltps.get(s, 0),
            "ohlc": {
                "open": ltps.get(s, 0) * 0.99,
                "high": ltps.get(s, 0) * 1.01,
                "low": ltps.get(s, 0) * 0.98,
                "close": ltps.get(s, 0),
            },
            "volume": random.randint(100000, 5000000),
            "change": round(random.uniform(-3, 3), 2),
        }
        for s in symbols
    }


async def get_positions() -> List[dict]:
    """Get current open positions from Kite."""
    if is_logged_in():
        try:
            kite = get_kite()
            positions = kite.positions()
            return positions.get("net", [])
        except Exception as e:
            logger.warning("Positions fetch failed: %s", e)
    return []


async def get_holdings() -> List[dict]:
    """Get current holdings from Kite."""
    if is_logged_in():
        try:
            kite = get_kite()
            return kite.holdings()
        except Exception as e:
            logger.warning("Holdings fetch failed: %s", e)
    return []


async def get_margins() -> dict:
    """Get account margins from Kite."""
    if is_logged_in():
        try:
            kite = get_kite()
            return kite.margins()
        except Exception as e:
            logger.warning("Margins fetch failed: %s", e)
    return {"equity": {"available": {"live_balance": 100000}}}

"""F&O symbol scanner — analyses NSE F&O stocks for trading signals.

Scores each F&O symbol on technical signals and returns ranked candidates
for intraday short-selling, intraday buying, and BTST opportunities.

Works in both paper mode (synthetic data) and live mode (Kite API).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from app.core.config import get_settings
from ml.features.engineering import build_features

logger = logging.getLogger(__name__)
settings = get_settings()

# Top NSE F&O stocks by liquidity
FNO_SYMBOLS: List[str] = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "BAJFINANCE", "ITC", "HINDUNILVR", "KOTAKBANK",
    "AXISBANK", "LT", "BHARTIARTL", "ASIANPAINT", "MARUTI",
    "HCLTECH", "SUNPHARMA", "TITAN", "WIPRO", "ULTRACEMCO",
    "TATAMOTORS", "POWERGRID", "NTPC", "TATASTEEL", "ADANIENT",
    "TECHM", "INDUSINDBK", "BAJAJFINSV", "HDFCLIFE", "ONGC",
    "COALINDIA", "JSWSTEEL", "M&M", "GRASIM", "DIVISLAB",
    "DRREDDY", "CIPLA", "EICHERMOT", "BPCL", "TATACONSUM",
]


def _generate_synthetic_ohlcv(symbol: str, days: int = 300) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for paper mode testing."""
    np.random.seed(hash(symbol) % 2**31)
    n = days
    base = np.random.uniform(500, 4000)
    returns = np.random.normal(0, 0.02, n)
    close = base * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "date": pd.date_range(end=datetime.now(), periods=n, freq="B"),
        "open": close * (1 + np.random.uniform(-0.01, 0.01, n)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.015, n))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.015, n))),
        "close": close,
        "volume": np.random.randint(500_000, 20_000_000, n),
    })
    return df


def score_for_short(featured: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Score a symbol for short-selling based on technical weakness.

    Higher score = stronger SELL signal.
    """
    if featured.empty:
        return {"symbol": symbol, "score": 0, "action": "SKIP", "reason": "no data"}

    latest = featured.iloc[-1]
    prev = featured.iloc[-2] if len(featured) > 1 else latest

    score = 0.0
    reasons = []

    rsi = latest.get("rsi", 50)
    if rsi > 75:
        score += 25; reasons.append(f"RSI overbought: {rsi:.1f}")
    elif rsi > 70:
        score += 15; reasons.append(f"RSI elevated: {rsi:.1f}")
    elif rsi > 60:
        score += 5

    macd_hist = latest.get("macd_hist", 0)
    prev_hist = prev.get("macd_hist", 0)
    if macd_hist < 0 and prev_hist >= 0:
        score += 20; reasons.append("MACD bearish crossover")
    elif macd_hist < 0:
        score += 10; reasons.append(f"MACD negative: {macd_hist:.4f}")

    bb_pct = latest.get("bb_pct", 0.5)
    if bb_pct > 1.0:
        score += 20; reasons.append(f"Above upper BB: {bb_pct:.2f}")
    elif bb_pct > 0.8:
        score += 10; reasons.append(f"Near upper BB: {bb_pct:.2f}")

    ret_1d = latest.get("return_1d", 0)
    if ret_1d < -0.02:
        score += 15; reasons.append(f"Sharp drop: {ret_1d*100:.1f}%")
    elif ret_1d < -0.005:
        score += 8; reasons.append(f"Declining: {ret_1d*100:.1f}%")

    vol = latest.get("volatility", 0)
    if vol > 0.3:
        score += 10; reasons.append(f"High volatility: {vol:.2f}")

    close = latest.get("close", 0)
    sma_5 = latest.get("sma_5", close)
    sma_20 = latest.get("sma_20", close)
    if close < sma_5 and close < sma_20:
        score += 15; reasons.append("Below SMA5 and SMA20")
    elif close < sma_5:
        score += 8; reasons.append("Below SMA5")

    ret_5d = latest.get("return_5d", 0)
    if ret_5d < -0.03:
        score += 10; reasons.append(f"5-day decline: {ret_5d*100:.1f}%")

    score = min(score, 100)

    if score >= 60:
        action = "STRONG SELL"
    elif score >= 40:
        action = "SELL"
    elif score >= 25:
        action = "WEAK SELL"
    else:
        action = "HOLD"

    return {
        "symbol": symbol, "score": round(score, 1), "action": action,
        "direction": "SELL", "reasons": reasons,
        "price": round(float(close), 2),
        "rsi": round(float(rsi), 1),
        "macd_hist": round(float(macd_hist), 4),
        "bb_pct": round(float(bb_pct), 2),
        "return_1d_pct": round(float(ret_1d * 100), 2),
        "return_5d_pct": round(float(ret_5d * 100 if ret_5d else 0), 2),
        "volatility": round(float(vol), 3),
        "sma_5": round(float(sma_5), 2),
        "sma_20": round(float(sma_20), 2),
    }


def score_for_buy(featured: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Score a symbol for buying (BTST long) based on technical strength.

    Higher score = stronger BUY signal. Used for BTST opportunities.
    """
    if featured.empty:
        return {"symbol": symbol, "score": 0, "action": "SKIP", "reason": "no data"}

    latest = featured.iloc[-1]
    prev = featured.iloc[-2] if len(featured) > 1 else latest

    score = 0.0
    reasons = []

    rsi = latest.get("rsi", 50)
    if rsi < 25:
        score += 25; reasons.append(f"RSI oversold: {rsi:.1f}")
    elif rsi < 30:
        score += 15; reasons.append(f"RSI low: {rsi:.1f}")
    elif rsi < 40:
        score += 5

    macd_hist = latest.get("macd_hist", 0)
    prev_hist = prev.get("macd_hist", 0)
    if macd_hist > 0 and prev_hist <= 0:
        score += 20; reasons.append("MACD bullish crossover")
    elif macd_hist > 0:
        score += 10; reasons.append(f"MACD positive: {macd_hist:.4f}")

    bb_pct = latest.get("bb_pct", 0.5)
    if bb_pct < 0.0:
        score += 20; reasons.append(f"Below lower BB: {bb_pct:.2f}")
    elif bb_pct < 0.2:
        score += 10; reasons.append(f"Near lower BB: {bb_pct:.2f}")

    ret_1d = latest.get("return_1d", 0)
    if ret_1d > 0.02:
        score += 15; reasons.append(f"Strong rally: {ret_1d*100:.1f}%")
    elif ret_1d > 0.005:
        score += 8; reasons.append(f"Rising: {ret_1d*100:.1f}%")

    vol = latest.get("volatility", 0)
    if vol > 0.3:
        score += 10; reasons.append(f"High volatility: {vol:.2f}")

    close = latest.get("close", 0)
    sma_5 = latest.get("sma_5", close)
    sma_20 = latest.get("sma_20", close)
    if close > sma_5 and close > sma_20:
        score += 15; reasons.append("Above SMA5 and SMA20")
    elif close > sma_5:
        score += 8; reasons.append("Above SMA5")

    ret_5d = latest.get("return_5d", 0)
    if ret_5d > 0.03:
        score += 10; reasons.append(f"5-day rally: {ret_5d*100:.1f}%")

    score = min(score, 100)

    if score >= 60:
        action = "STRONG BUY"
    elif score >= 40:
        action = "BUY"
    elif score >= 25:
        action = "WEAK BUY"
    else:
        action = "HOLD"

    return {
        "symbol": symbol, "score": round(score, 1), "action": action,
        "direction": "BUY", "reasons": reasons,
        "price": round(float(close), 2),
        "rsi": round(float(rsi), 1),
        "macd_hist": round(float(macd_hist), 4),
        "bb_pct": round(float(bb_pct), 2),
        "return_1d_pct": round(float(ret_1d * 100), 2),
        "return_5d_pct": round(float(ret_5d * 100 if ret_5d else 0), 2),
        "volatility": round(float(vol), 3),
        "sma_5": round(float(sma_5), 2),
        "sma_20": round(float(sma_20), 2),
    }


async def scan_fno_symbols(
    symbols: List[str] | None = None,
    top_n: int = 10,
    use_live_data: bool = False,
    scan_type: str = "sell",  # "sell", "buy", or "both"
) -> List[Dict[str, Any]]:
    """Scan F&O symbols and return top trading candidates.

    scan_type:
        "sell" — intraday short-sell candidates
        "buy" — BTST long candidates
        "both" — all signals sorted by score
    """
    symbols = symbols or FNO_SYMBOLS
    results = []

    for symbol in symbols:
        try:
            if use_live_data:
                from app.services.market_data import fetch_daily_prices
                df = await fetch_daily_prices(symbol, days=300, use_cache=True)
            else:
                df = _generate_synthetic_ohlcv(symbol)

            featured = build_features(df)
            if featured.empty:
                continue

            if scan_type in ("sell", "both"):
                sell_result = score_for_short(featured, symbol)
                results.append(sell_result)

            if scan_type in ("buy", "both"):
                buy_result = score_for_buy(featured, symbol)
                results.append(buy_result)

        except Exception as exc:
            logger.warning("Failed to scan %s: %s", symbol, exc)
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]

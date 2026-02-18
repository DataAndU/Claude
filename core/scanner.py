"""Stock scanner for NSE equities and F&O.

Scans the watchlist for trading opportunities using technical analysis
and AI-generated scores. Works with both live Kite data and synthetic data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.config import load_config
from core.feature_engine import build_features

logger = logging.getLogger(__name__)


async def scan_stocks(
    scan_type: str = "all",
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Scan watchlist stocks and score them.

    Args:
        scan_type: "buy", "sell", or "all"
        top_n: Number of top results to return
    """
    cfg = load_config()
    results = []

    for symbol in cfg.watchlist:
        try:
            from core.kite_data import fetch_historical
            df = await fetch_historical(symbol, days=cfg.lookback_days)
            if df.empty or len(df) < 60:
                continue

            featured = build_features(df, advanced=True)
            if featured.empty:
                continue

            latest = featured.iloc[-1]
            score_data = _compute_score(latest, symbol)

            if scan_type == "buy" and score_data["action"] not in ("BUY", "STRONG BUY"):
                continue
            if scan_type == "sell" and score_data["action"] not in ("SELL", "STRONG SELL"):
                continue

            results.append(score_data)

        except Exception as e:
            logger.warning("Scan failed for %s: %s", symbol, e)

    results.sort(key=lambda x: abs(x["score"]), reverse=True)
    return results[:top_n]


def _compute_score(row: pd.Series, symbol: str) -> Dict[str, Any]:
    """Compute a composite trading score from technical indicators."""
    score = 0.0
    reasons = []

    rsi = row.get("rsi", 50)
    macd_hist = row.get("macd_hist", 0)
    adx = row.get("adx", 25)
    bb_pct = row.get("bb_pct", 0.5)
    stoch_k = row.get("stoch_rsi_k", 0.5)
    vol_ratio = row.get("volume_ratio", 1.0)
    momentum = row.get("momentum_20", 0)
    di_cross = row.get("di_crossover", 0)
    ichimoku_above = row.get("ichimoku_above_cloud", 0)
    ichimoku_below = row.get("ichimoku_below_cloud", 0)
    engulfing = row.get("engulfing_score", 0)
    volatility = row.get("volatility", 0.2)
    atr = row.get("atr", 0)

    # RSI signal (+-20)
    if rsi < 30: score += 20; reasons.append(f"Oversold RSI={rsi:.0f}")
    elif rsi < 40: score += 10; reasons.append(f"Low RSI={rsi:.0f}")
    elif rsi > 70: score -= 20; reasons.append(f"Overbought RSI={rsi:.0f}")
    elif rsi > 60: score -= 10; reasons.append(f"High RSI={rsi:.0f}")

    # MACD (+-15)
    if macd_hist > 0: score += 15; reasons.append("MACD bullish")
    elif macd_hist < 0: score -= 15; reasons.append("MACD bearish")

    # ADX trend strength (+-10)
    if adx > 30:
        if momentum > 0: score += 10; reasons.append(f"Strong uptrend ADX={adx:.0f}")
        else: score -= 10; reasons.append(f"Strong downtrend ADX={adx:.0f}")

    # Bollinger Bands (+-10)
    if bb_pct < 0.1: score += 10; reasons.append("Below lower Bollinger")
    elif bb_pct > 0.9: score -= 10; reasons.append("Above upper Bollinger")

    # Stochastic RSI (+-10)
    if stoch_k < 0.2: score += 10; reasons.append("StochRSI oversold")
    elif stoch_k > 0.8: score -= 10; reasons.append("StochRSI overbought")

    # Volume (+-5)
    if vol_ratio > 1.5: score += 5 * np.sign(momentum); reasons.append(f"High volume {vol_ratio:.1f}x")

    # DI crossover (+-10)
    if di_cross > 5: score += 10; reasons.append("Bullish DI crossover")
    elif di_cross < -5: score -= 10; reasons.append("Bearish DI crossover")

    # Ichimoku (+-10)
    if ichimoku_above: score += 10; reasons.append("Above Ichimoku cloud")
    elif ichimoku_below: score -= 10; reasons.append("Below Ichimoku cloud")

    # Candlestick (+-5)
    if engulfing > 0: score += 5; reasons.append("Bullish engulfing")
    elif engulfing < 0: score -= 5; reasons.append("Bearish engulfing")

    # Determine action
    abs_score = abs(score)
    if score > 50: action = "STRONG BUY"
    elif score > 25: action = "BUY"
    elif score < -50: action = "STRONG SELL"
    elif score < -25: action = "SELL"
    else: action = "HOLD"

    # Signal quality
    quality = "D"
    if abs_score > 60 and adx > 25 and vol_ratio > 1.2: quality = "A"
    elif abs_score > 40 and adx > 20: quality = "B"
    elif abs_score > 20: quality = "C"

    return {
        "symbol": symbol,
        "score": round(score, 2),
        "action": action,
        "quality": quality,
        "price": round(float(row.get("close", 0)), 2),
        "rsi": round(float(rsi), 1),
        "adx": round(float(adx), 1),
        "macd_hist": round(float(macd_hist), 4),
        "volume_ratio": round(float(vol_ratio), 2),
        "volatility": round(float(volatility), 4),
        "atr": round(float(atr), 4),
        "reasons": reasons,
        "signal_quality": quality,
    }


async def scan_fno_opportunities(top_n: int = 5) -> List[Dict[str, Any]]:
    """Scan for F&O trading opportunities."""
    cfg = load_config()
    if not cfg.fno_enabled:
        return []

    buy_signals = await scan_stocks(scan_type="buy", top_n=top_n)
    sell_signals = await scan_stocks(scan_type="sell", top_n=top_n)

    opportunities = []
    for sig in buy_signals:
        opportunities.append({
            **sig,
            "fno_action": "BUY CE",
            "option_type": "CE",
            "exchange": "NFO",
            "product": "MIS",
        })
    for sig in sell_signals:
        opportunities.append({
            **sig,
            "fno_action": "BUY PE",
            "option_type": "PE",
            "exchange": "NFO",
            "product": "MIS",
        })

    opportunities.sort(key=lambda x: abs(x["score"]), reverse=True)
    return opportunities[:top_n]

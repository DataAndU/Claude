"""AI-powered market regime detection service.

Classifies the current market environment into regimes:
    - TRENDING_UP: Strong upward momentum, trend-following strategies preferred
    - TRENDING_DOWN: Strong downward momentum, short-selling opportunities
    - MEAN_REVERTING: Sideways/oscillating market, reversion strategies preferred
    - HIGH_VOLATILITY: Unstable conditions, reduce position sizes
    - LOW_VOLATILITY: Calm market, increase position sizes cautiously

The regime detector uses a combination of statistical indicators:
    - ADX for trend strength
    - Hurst exponent proxy for mean-reversion vs trend
    - Volatility percentile ranking
    - Price slope analysis
    - Volume profile analysis

Regime detection directly influences:
    - Which trading strategies are activated
    - Position sizing multipliers
    - Confidence thresholds
    - Stop-loss / take-profit distances
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ml.features.engineering import build_features

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


# Regime-specific trading parameters
REGIME_PARAMS: Dict[MarketRegime, Dict[str, Any]] = {
    MarketRegime.TRENDING_UP: {
        "preferred_strategies": ["btst_buy", "options_btst"],
        "position_size_multiplier": 1.2,
        "confidence_threshold_adj": -0.05,  # lower threshold = more trades
        "sl_multiplier": 1.5,   # wider stops in trends
        "tp_multiplier": 2.0,   # bigger targets in trends
        "max_trades_adj": 2,    # allow more trades
    },
    MarketRegime.TRENDING_DOWN: {
        "preferred_strategies": ["intraday_sell", "options_intraday"],
        "position_size_multiplier": 1.1,
        "confidence_threshold_adj": -0.05,
        "sl_multiplier": 1.5,
        "tp_multiplier": 2.0,
        "max_trades_adj": 2,
    },
    MarketRegime.MEAN_REVERTING: {
        "preferred_strategies": ["intraday_sell", "btst_buy"],
        "position_size_multiplier": 0.9,
        "confidence_threshold_adj": 0.05,  # higher threshold = more selective
        "sl_multiplier": 0.8,   # tighter stops in range
        "tp_multiplier": 0.7,   # smaller targets in range
        "max_trades_adj": 0,
    },
    MarketRegime.HIGH_VOLATILITY: {
        "preferred_strategies": ["options_intraday"],
        "position_size_multiplier": 0.5,   # reduce size in volatile markets
        "confidence_threshold_adj": 0.10,  # much more selective
        "sl_multiplier": 2.0,   # very wide stops
        "tp_multiplier": 1.5,
        "max_trades_adj": -2,   # fewer trades
    },
    MarketRegime.LOW_VOLATILITY: {
        "preferred_strategies": ["btst_buy", "intraday_sell"],
        "position_size_multiplier": 1.0,
        "confidence_threshold_adj": 0.0,
        "sl_multiplier": 0.7,
        "tp_multiplier": 0.8,
        "max_trades_adj": 0,
    },
}


# In-memory cache of recent regime detections per symbol
_regime_cache: Dict[str, Dict[str, Any]] = {}


def detect_regime(df: pd.DataFrame, symbol: str = "MARKET") -> Dict[str, Any]:
    """Detect the current market regime from OHLCV data.

    Returns a dict with regime classification and supporting metrics.
    """
    featured = build_features(df, advanced=True)
    if featured.empty or len(featured) < 20:
        return {
            "symbol": symbol,
            "regime": MarketRegime.MEAN_REVERTING,
            "confidence": 0.3,
            "params": REGIME_PARAMS[MarketRegime.MEAN_REVERTING],
            "metrics": {},
        }

    latest = featured.iloc[-1]
    lookback = featured.tail(20)

    # Collect regime scores
    scores: Dict[MarketRegime, float] = {r: 0.0 for r in MarketRegime}

    # --- ADX-based trend detection ---
    adx = latest.get("adx", 25)
    di_cross = latest.get("di_crossover", 0)
    if adx > 30:
        if di_cross > 0:
            scores[MarketRegime.TRENDING_UP] += 30
        else:
            scores[MarketRegime.TRENDING_DOWN] += 30
    elif adx < 20:
        scores[MarketRegime.MEAN_REVERTING] += 25

    # --- Trend slope analysis ---
    slope_20 = latest.get("trend_slope_20", 0)
    slope_50 = latest.get("trend_slope_50", 0)
    if slope_20 > 0 and slope_50 > 0:
        scores[MarketRegime.TRENDING_UP] += 20
    elif slope_20 < 0 and slope_50 < 0:
        scores[MarketRegime.TRENDING_DOWN] += 20
    elif abs(slope_20) < abs(slope_50) * 0.3:
        scores[MarketRegime.MEAN_REVERTING] += 15

    # --- Hurst exponent proxy ---
    hurst = latest.get("hurst_proxy", 0.5)
    if hurst > 0.6:
        # Trending behavior
        if slope_20 > 0:
            scores[MarketRegime.TRENDING_UP] += 15
        else:
            scores[MarketRegime.TRENDING_DOWN] += 15
    elif hurst < 0.4:
        scores[MarketRegime.MEAN_REVERTING] += 20

    # --- Volatility regime ---
    vol_regime = latest.get("volatility_regime", 0.5)
    if vol_regime > 0.85:
        scores[MarketRegime.HIGH_VOLATILITY] += 35
    elif vol_regime < 0.15:
        scores[MarketRegime.LOW_VOLATILITY] += 30

    volatility = latest.get("volatility", 0.2)
    if volatility > 0.4:
        scores[MarketRegime.HIGH_VOLATILITY] += 20
    elif volatility < 0.1:
        scores[MarketRegime.LOW_VOLATILITY] += 15

    # --- Volume analysis ---
    vol_trend = latest.get("vol_price_trend", 0)
    if vol_trend > 0.5:
        scores[MarketRegime.TRENDING_UP] += 10
    elif vol_trend < -0.5:
        scores[MarketRegime.TRENDING_DOWN] += 10

    # --- Moving average alignment ---
    close = latest.get("close", 0)
    sma_5 = latest.get("sma_5", close)
    sma_20 = latest.get("sma_20", close)
    sma_50 = latest.get("sma_50", close)
    if close > sma_5 > sma_20 > sma_50:
        scores[MarketRegime.TRENDING_UP] += 15
    elif close < sma_5 < sma_20 < sma_50:
        scores[MarketRegime.TRENDING_DOWN] += 15

    # --- RSI extremes suggest mean reversion ---
    rsi = latest.get("rsi", 50)
    if 35 <= rsi <= 65:
        scores[MarketRegime.MEAN_REVERTING] += 10
    mr_score = latest.get("mean_reversion_score", 0)
    if mr_score > 0.3:
        scores[MarketRegime.MEAN_REVERTING] += 15

    # Determine regime
    total = sum(scores.values()) or 1.0
    regime = max(scores, key=scores.get)
    confidence = scores[regime] / total

    result = {
        "symbol": symbol,
        "regime": regime,
        "confidence": round(float(confidence), 4),
        "params": REGIME_PARAMS[regime],
        "scores": {r.value: round(s / total, 4) for r, s in scores.items()},
        "metrics": {
            "adx": round(float(adx), 2),
            "di_crossover": round(float(di_cross), 2),
            "trend_slope_20": round(float(slope_20), 6),
            "hurst_proxy": round(float(hurst), 4),
            "volatility": round(float(volatility), 4),
            "volatility_regime": round(float(vol_regime), 4),
            "rsi": round(float(rsi), 2),
            "mean_reversion_score": round(float(mr_score), 4),
        },
    }

    _regime_cache[symbol] = result
    return result


def get_cached_regime(symbol: str) -> Optional[Dict[str, Any]]:
    """Retrieve the most recently detected regime for a symbol."""
    return _regime_cache.get(symbol)


def get_all_regimes() -> Dict[str, Dict[str, Any]]:
    """Return all cached regime detections."""
    return _regime_cache.copy()


def get_regime_adjustment(regime: MarketRegime) -> Dict[str, Any]:
    """Get trading parameter adjustments for a given regime."""
    return REGIME_PARAMS.get(regime, REGIME_PARAMS[MarketRegime.MEAN_REVERTING])

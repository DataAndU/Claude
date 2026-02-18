"""Strategy engine — generates signals with multi-timeframe confirmation.

Orchestrates feature engineering, ML predictions, and regime-aware
signal filtering into actionable trade signals.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import load_config
from core.feature_engine import build_features
from core.model_engine import LABEL_MAP, get_latest_model, predict

logger = logging.getLogger(__name__)


def generate_signal(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    model_type: str = "random_forest",
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a single trading signal from OHLCV data.

    Returns:
        {"signal": "BUY"/"SELL"/"HOLD", "confidence": float, "regime": str, ...}
    """
    cfg = load_config()

    # Build features
    featured = build_features(df)
    if featured.empty or len(featured) < 20:
        return _hold_signal(symbol, timeframe, reason="insufficient data")

    # Find model
    if model_path is None:
        model_path = get_latest_model(symbol, timeframe, model_type)
    if model_path is None:
        return _hold_signal(symbol, timeframe, reason="no trained model")

    # Predict
    try:
        preds, probs = predict(model_path, featured, model_type)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        return _hold_signal(symbol, timeframe, reason=f"prediction error: {e}")

    if len(preds) == 0:
        return _hold_signal(symbol, timeframe, reason="empty predictions")

    # Take latest prediction
    latest_pred = int(preds[-1])
    latest_conf = float(probs[-1])
    latest_row = featured.iloc[-1]
    regime = latest_row.get("regime", "unknown")

    signal_str = LABEL_MAP.get(latest_pred, "HOLD")

    # Confidence threshold filter
    if latest_conf < cfg.min_confidence:
        signal_str = "HOLD"

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": signal_str,
        "confidence": round(latest_conf, 4),
        "model_type": model_type,
        "regime": regime,
        "price": round(float(latest_row.get("close", 0)), 8),
        "rsi": round(float(latest_row.get("rsi", 50)), 2),
        "atr": round(float(latest_row.get("atr", 0)), 8),
        "adx": round(float(latest_row.get("adx", 25)), 2),
        "macd_hist": round(float(latest_row.get("macd_hist", 0)), 8),
        "volatility": round(float(latest_row.get("volatility", 0)), 4),
    }


def generate_multi_timeframe_signal(
    data_by_tf: Dict[str, pd.DataFrame],
    symbol: str,
    model_type: str = "random_forest",
) -> Dict[str, Any]:
    """Generate a signal confirmed across multiple timeframes.

    Process:
        1. Generate signal on primary timeframe
        2. Check confirmation from higher timeframes
        3. Only keep signal if majority of timeframes agree
    """
    cfg = load_config()
    primary_tf = cfg.primary_timeframe
    confirm_tfs = cfg.confirmation_timeframes

    if primary_tf not in data_by_tf:
        return _hold_signal(symbol, primary_tf, reason="no primary TF data")

    # Primary signal
    primary_sig = generate_signal(
        data_by_tf[primary_tf], symbol, primary_tf, model_type,
    )
    primary_direction = primary_sig["signal"]

    if primary_direction == "HOLD":
        return primary_sig

    # Confirmation signals
    confirmations = []
    for tf in confirm_tfs:
        if tf not in data_by_tf:
            continue
        tf_sig = generate_signal(data_by_tf[tf], symbol, tf, model_type)
        confirmations.append(tf_sig)

    if not confirmations:
        # No confirmation data — use primary only
        primary_sig["confirmed"] = False
        primary_sig["confirmation_count"] = 0
        return primary_sig

    # Count how many confirm the primary direction
    agree = sum(1 for s in confirmations if s["signal"] == primary_direction)
    total = len(confirmations)
    confirmed = agree >= (total / 2)

    if not confirmed:
        logger.info(
            "Signal %s for %s not confirmed (%d/%d agree)",
            primary_direction, symbol, agree, total,
        )
        return _hold_signal(symbol, primary_tf, reason="not confirmed by higher TFs")

    primary_sig["confirmed"] = True
    primary_sig["confirmation_count"] = agree
    primary_sig["confirmation_total"] = total

    # Boost confidence if strongly confirmed
    if agree == total:
        primary_sig["confidence"] = min(primary_sig["confidence"] * 1.15, 1.0)

    return primary_sig


def generate_all_signals(
    data_map: Dict[str, Dict[str, pd.DataFrame]],
    model_type: str = "random_forest",
) -> List[Dict[str, Any]]:
    """Generate signals for all symbols with multi-TF confirmation."""
    signals = []
    for symbol, tf_data in data_map.items():
        sig = generate_multi_timeframe_signal(tf_data, symbol, model_type)
        signals.append(sig)
    return signals


def _hold_signal(symbol: str, timeframe: str, reason: str = "") -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": "HOLD",
        "confidence": 0.0,
        "model_type": "",
        "regime": "unknown",
        "reason": reason,
        "price": 0.0,
        "rsi": 50.0,
        "atr": 0.0,
        "adx": 25.0,
        "macd_hist": 0.0,
        "volatility": 0.0,
    }

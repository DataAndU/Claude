"""AI-powered signal generation for Indian equity markets.

Combines ML predictions with multi-indicator confidence scoring,
multi-timeframe analysis, and signal quality grading (A/B/C/D).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.feature_engine import build_features
from core.model_engine import LABEL_MAP, predict

logger = logging.getLogger(__name__)


def generate_signal(
    df: pd.DataFrame,
    model_path: str,
    model_type: str = "random_forest",
    regime: str = "mean_reverting",
) -> Dict[str, Any]:
    """Generate a single trading signal for the latest data point.

    Returns the most recent signal with AI confidence scoring.
    """
    signals = generate_signals(df, model_path, model_type, regime)
    if not signals:
        return {"signal": "HOLD", "confidence": 0.0, "quality": "D"}
    return signals[-1]


def generate_signals(
    df: pd.DataFrame,
    model_path: str,
    model_type: str = "random_forest",
    regime: str = "mean_reverting",
) -> List[Dict[str, Any]]:
    """Generate trading signals with AI confidence scoring."""
    featured = build_features(df, advanced=True)
    if featured.empty:
        return []

    preds, model_probs = predict(model_path, featured, model_type)
    if len(preds) == 0:
        return []

    # Align featured DataFrame with predictions
    featured = featured.iloc[-len(preds):].reset_index(drop=True)

    # Multi-timeframe analysis
    mtf_signals = _multi_timeframe_analysis(featured)

    signals: List[Dict] = []
    for idx, pred in enumerate(preds):
        row = featured.iloc[idx]
        pred_int = int(pred)

        # AI confidence scoring
        confidence = _compute_ai_confidence(
            row, pred_int,
            model_conf=float(model_probs[idx]) if idx < len(model_probs) else 0.5,
            mtf=mtf_signals.get(idx, {}),
        )

        signal_dict = {
            "index": idx,
            "signal": LABEL_MAP.get(pred_int, "HOLD"),
            "confidence": round(float(confidence), 4),
            "price": round(float(row["close"]), 2) if "close" in row.index else 0.0,
            "rsi": round(float(row.get("rsi", 50)), 2),
            "macd_hist": round(float(row.get("macd_hist", 0)), 6),
            "adx": round(float(row.get("adx", 25)), 2),
            "atr": round(float(row.get("atr", 0)), 4),
            "volatility": round(float(row.get("volatility", 0)), 4),
            "volume_ratio": round(float(row.get("volume_ratio", 1.0)), 4),
            "quality": _compute_signal_quality(row, pred_int, confidence),
            "explanation": _generate_explanation(row, pred_int, confidence),
        }

        if idx in mtf_signals:
            signal_dict["mtf_alignment"] = mtf_signals[idx]

        signals.append(signal_dict)

    return signals


def scan_watchlist(
    symbol_data: Dict[str, pd.DataFrame],
    model_paths: Dict[str, str],
    model_type: str = "random_forest",
    regime: str = "mean_reverting",
    min_confidence: float = 0.6,
) -> List[Dict[str, Any]]:
    """Scan multiple symbols and return actionable signals sorted by confidence."""
    results = []
    for symbol, df in symbol_data.items():
        mp = model_paths.get(symbol)
        if not mp:
            continue
        try:
            sig = generate_signal(df, mp, model_type, regime)
            if sig["confidence"] >= min_confidence and sig["signal"] != "HOLD":
                sig["symbol"] = symbol
                results.append(sig)
        except Exception as e:
            logger.warning("Signal generation failed for %s: %s", symbol, e)

    return sorted(results, key=lambda x: x["confidence"], reverse=True)


# ─── AI Confidence Scoring ──────────────────────────────────────────────────

def _compute_ai_confidence(
    row: pd.Series, pred: int,
    model_conf: float = 0.5, mtf: dict = None,
) -> float:
    """Multi-dimensional confidence scoring.

    Dimensions: model probability, technical alignment, trend strength,
    multi-timeframe alignment, volume confirmation, stochastic RSI.
    """
    scores = []
    weights = []

    # 1. Model confidence (30%)
    scores.append(model_conf)
    weights.append(0.30)

    # 2. Technical alignment (25%)
    tech = _technical_alignment(row, pred)
    scores.append(tech)
    weights.append(0.25)

    # 3. Trend strength via ADX (15%)
    adx = row.get("adx", 25)
    if pred in (0, 2):
        trend_conf = min(adx / 50.0, 1.0)
    else:
        trend_conf = max(1.0 - adx / 50.0, 0.0)
    scores.append(trend_conf)
    weights.append(0.15)

    # 4. Multi-timeframe alignment (15%)
    mtf_align = (mtf or {}).get("alignment_score", 0.5)
    scores.append(mtf_align)
    weights.append(0.15)

    # 5. Volume confirmation (10%)
    vol_ratio = row.get("volume_ratio", 1.0)
    vol_trend = row.get("vol_price_trend", 0)
    if pred == 2:
        vol_conf = min(vol_ratio / 2.0, 1.0) * (0.5 + 0.5 * min(max(vol_trend, 0), 1))
    elif pred == 0:
        vol_conf = min(vol_ratio / 2.0, 1.0) * (0.5 + 0.5 * min(max(-vol_trend, 0), 1))
    else:
        vol_conf = 0.5
    scores.append(vol_conf)
    weights.append(0.10)

    # 6. Stochastic RSI (5%)
    stoch_k = row.get("stoch_rsi_k", 0.5)
    if pred == 2 and stoch_k < 0.2:
        stoch_conf = 0.9
    elif pred == 0 and stoch_k > 0.8:
        stoch_conf = 0.9
    elif pred == 1 and 0.3 <= stoch_k <= 0.7:
        stoch_conf = 0.7
    else:
        stoch_conf = 0.4
    scores.append(stoch_conf)
    weights.append(0.05)

    final = sum(s * w for s, w in zip(scores, weights))
    return min(max(final, 0.0), 1.0)


def _technical_alignment(row: pd.Series, pred: int) -> float:
    alignment = 0.0
    count = 0

    rsi = row.get("rsi", 50)
    macd_hist = row.get("macd_hist", 0)
    bb_pct = row.get("bb_pct", 0.5)
    di_cross = row.get("di_crossover", 0)
    ichimoku_above = row.get("ichimoku_above_cloud", 0)
    engulfing = row.get("engulfing_score", 0)

    if pred == 2:  # BUY
        if rsi < 40: alignment += 1.0; count += 1
        elif rsi < 50: alignment += 0.5; count += 1
        else: count += 1
        if macd_hist > 0: alignment += 1.0; count += 1
        else: count += 1
        if bb_pct < 0.2: alignment += 1.0; count += 1
        elif bb_pct < 0.5: alignment += 0.5; count += 1
        else: count += 1
        if di_cross > 0: alignment += 1.0; count += 1
        else: count += 1
        if ichimoku_above > 0: alignment += 1.0; count += 1
        else: count += 1
        if engulfing > 0: alignment += 0.8; count += 1
        else: count += 1
    elif pred == 0:  # SELL
        if rsi > 60: alignment += 1.0; count += 1
        elif rsi > 50: alignment += 0.5; count += 1
        else: count += 1
        if macd_hist < 0: alignment += 1.0; count += 1
        else: count += 1
        if bb_pct > 0.8: alignment += 1.0; count += 1
        elif bb_pct > 0.5: alignment += 0.5; count += 1
        else: count += 1
        if di_cross < 0: alignment += 1.0; count += 1
        else: count += 1
        if engulfing < 0: alignment += 0.8; count += 1
        else: count += 1
    else:  # HOLD
        if 40 <= rsi <= 60: alignment += 1.0; count += 1
        else: count += 1
        if abs(macd_hist) < 0.01: alignment += 1.0; count += 1
        else: count += 1
        count += 3

    return alignment / max(count, 1)


def _multi_timeframe_analysis(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    if len(df) < 50:
        return {}

    results = {}
    for idx in range(50, len(df)):
        short_window = df.iloc[idx - 5:idx + 1]
        medium_window = df.iloc[idx - 20:idx + 1]
        long_window = df.iloc[idx - 50:idx + 1]

        short_rsi = short_window["rsi"].iloc[-1] if "rsi" in short_window else 50
        short_macd = short_window["macd_hist"].iloc[-1] if "macd_hist" in short_window else 0
        short_signal = 2 if (short_rsi < 40 and short_macd > 0) else (0 if (short_rsi > 60 and short_macd < 0) else 1)

        med_slope = (medium_window["close"].iloc[-1] - medium_window["close"].iloc[0]) / medium_window["close"].iloc[0]
        med_signal = 2 if med_slope > 0.02 else (0 if med_slope < -0.02 else 1)

        long_slope = (long_window["close"].iloc[-1] - long_window["close"].iloc[0]) / long_window["close"].iloc[0]
        long_signal = 2 if long_slope > 0.05 else (0 if long_slope < -0.05 else 1)

        signals = [short_signal, med_signal, long_signal]
        counts = Counter(signals)
        dominant = counts.most_common(1)[0]

        results[idx] = {
            "short_signal": LABEL_MAP[short_signal],
            "medium_signal": LABEL_MAP[med_signal],
            "long_signal": LABEL_MAP[long_signal],
            "dominant_signal": LABEL_MAP[dominant[0]],
            "alignment_score": round(dominant[1] / 3.0, 4),
        }

    return results


def _compute_signal_quality(row: pd.Series, pred: int, confidence: float) -> str:
    adx = row.get("adx", 25)
    vol_ratio = row.get("volume_ratio", 1.0)

    quality_score = confidence * 0.5
    if adx > 25: quality_score += 0.2
    if vol_ratio > 1.2: quality_score += 0.15
    if row.get("regime_trend_strength", 0) > 0.3: quality_score += 0.15

    if quality_score >= 0.75: return "A"
    elif quality_score >= 0.55: return "B"
    elif quality_score >= 0.35: return "C"
    return "D"


def _generate_explanation(row: pd.Series, pred: int, confidence: float) -> List[str]:
    reasons = []
    signal = LABEL_MAP.get(pred, "HOLD")

    rsi = row.get("rsi", 50)
    macd_hist = row.get("macd_hist", 0)
    adx = row.get("adx", 25)
    stoch_k = row.get("stoch_rsi_k", 0.5)
    vol_ratio = row.get("volume_ratio", 1.0)
    bb_pct = row.get("bb_pct", 0.5)

    if pred == 2:
        if rsi < 30: reasons.append(f"RSI deeply oversold ({rsi:.1f})")
        elif rsi < 40: reasons.append(f"RSI approaching oversold ({rsi:.1f})")
        if macd_hist > 0: reasons.append("MACD histogram turning positive")
        if stoch_k < 0.2: reasons.append(f"Stochastic RSI oversold ({stoch_k:.2f})")
        if bb_pct < 0.1: reasons.append("Price near lower Bollinger Band")
    elif pred == 0:
        if rsi > 70: reasons.append(f"RSI deeply overbought ({rsi:.1f})")
        elif rsi > 60: reasons.append(f"RSI approaching overbought ({rsi:.1f})")
        if macd_hist < 0: reasons.append("MACD histogram turning negative")
        if stoch_k > 0.8: reasons.append(f"Stochastic RSI overbought ({stoch_k:.2f})")
        if bb_pct > 0.9: reasons.append("Price near upper Bollinger Band")

    if adx > 30: reasons.append(f"Strong trend (ADX={adx:.0f})")
    elif adx < 20: reasons.append(f"Weak trend/ranging (ADX={adx:.0f})")
    if vol_ratio > 2.0: reasons.append(f"Volume spike ({vol_ratio:.1f}x avg)")
    if confidence > 0.75: reasons.append("High AI confidence")
    elif confidence < 0.4: reasons.append("Low AI confidence — cautious")

    return reasons if reasons else [f"{signal} signal with moderate alignment"]

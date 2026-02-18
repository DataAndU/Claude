"""AI-powered signal generation service.

Combines ML predictions with multi-indicator confidence scoring,
multi-timeframe analysis, market regime awareness, and adaptive
ensemble weighting for high-quality trade signals.

Enhancements over v1:
    - AI confidence scoring using model probabilities + technical confirmation
    - Multi-timeframe signal alignment (short/medium/long)
    - Regime-aware signal filtering
    - Adaptive ensemble with dynamic model weighting
    - Signal quality scoring with explanations
    - Transformer model support with attention-based insights
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.features.engineering import build_features
from ml.training.random_forest import FEATURE_COLS, predict_rf

try:
    from ml.training.lstm_model import predict_lstm
except ImportError:
    predict_lstm = None  # type: ignore

try:
    from ml.training.transformer_model import predict_transformer, predict_transformer_with_confidence
except ImportError:
    predict_transformer = None  # type: ignore
    predict_transformer_with_confidence = None  # type: ignore

try:
    from ml.training.adaptive_ensemble import adaptive_ensemble_predict
except ImportError:
    adaptive_ensemble_predict = None  # type: ignore

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


def generate_signals(
    df: pd.DataFrame,
    model_path: str,
    model_type: str = "rf",
    ensemble: bool = False,
    rf_path: str | None = None,
    lstm_path: str | None = None,
    transformer_path: str | None = None,
    regime: str = "mean_reverting",
    use_ai_confidence: bool = True,
) -> List[Dict]:
    """Generate trading signals with AI-powered confidence scoring.

    Args:
        df: Raw OHLCV DataFrame.
        model_path: Path to primary model file.
        model_type: One of "rf", "lstm", "transformer".
        ensemble: If True, use adaptive ensemble across all available models.
        rf_path: Path to Random Forest model (for ensemble).
        lstm_path: Path to LSTM model (for ensemble).
        transformer_path: Path to Transformer model (for ensemble).
        regime: Current market regime string (from regime detector).
        use_ai_confidence: If True, use advanced multi-indicator confidence scoring.
    """
    featured = build_features(df, advanced=True)
    if featured.empty:
        return []

    # Adaptive ensemble mode
    if ensemble and adaptive_ensemble_predict is not None:
        result = adaptive_ensemble_predict(
            df=featured,
            rf_path=rf_path,
            lstm_path=lstm_path,
            transformer_path=transformer_path,
            regime=regime,
        )
        preds = result["predictions"]
        model_confidence = result["confidence"]
        ensemble_meta = {
            "model_weights": result["model_weights"],
            "disagreement": result["disagreement"],
            "agreement_score": result.get("agreement_score", 1.0),
            "signal_strength": result["signal_strength"],
        }
        featured = featured.iloc[-len(preds):].reset_index(drop=True)

    elif model_type == "transformer" and predict_transformer_with_confidence is not None:
        preds, model_confidence, attn_weights = predict_transformer_with_confidence(
            model_path, featured,
        )
        featured = featured.iloc[-len(preds):].reset_index(drop=True)
        ensemble_meta = None

    elif model_type == "lstm" and predict_lstm is not None:
        preds = predict_lstm(model_path, featured)
        model_confidence = np.full(len(preds), 0.6)
        featured = featured.iloc[-len(preds):].reset_index(drop=True)
        ensemble_meta = None

    else:
        preds = predict_rf(model_path, featured)
        model_confidence = np.full(len(preds), 0.6)
        ensemble_meta = None

    # Multi-timeframe analysis on the full featured data
    mtf_signals = _multi_timeframe_analysis(featured)

    signals: List[Dict] = []
    for idx, pred in enumerate(preds):
        row = featured.iloc[idx]
        pred_int = int(pred)

        # AI confidence scoring
        if use_ai_confidence:
            confidence = _compute_ai_confidence(
                row, pred_int,
                model_conf=float(model_confidence[idx]) if idx < len(model_confidence) else 0.5,
                mtf=mtf_signals.get(idx, {}),
            )
        else:
            confidence = _compute_confidence(row, pred_int)

        signal_dict = {
            "index": idx,
            "signal": LABEL_MAP.get(pred_int, "HOLD"),
            "confidence": round(float(confidence), 4),
            "price": round(float(row["close"]), 4) if "close" in row.index else 0.0,
            "risk_score": round(float(row.get("risk_score", 0.5)), 4),
            "rsi": round(float(row.get("rsi", 50)), 2),
            "macd_hist": round(float(row.get("macd_hist", 0)), 6),
            # AI-powered additions
            "adx": round(float(row.get("adx", 25)), 2),
            "stoch_rsi_k": round(float(row.get("stoch_rsi_k", 0.5)), 4),
            "regime_trend_strength": round(float(row.get("regime_trend_strength", 0.25)), 4),
            "volume_ratio": round(float(row.get("volume_ratio", 1.0)), 4),
            "vwap_deviation": round(float(row.get("vwap_deviation", 0)), 6),
            "signal_quality": _compute_signal_quality(row, pred_int, confidence),
            "explanation": _generate_signal_explanation(row, pred_int, confidence),
        }

        # Add multi-timeframe alignment
        if idx in mtf_signals:
            signal_dict["mtf_alignment"] = mtf_signals[idx]

        # Add ensemble metadata to last signal
        if ensemble_meta and idx == len(preds) - 1:
            signal_dict["ensemble_meta"] = ensemble_meta

        signals.append(signal_dict)

    return signals


def _compute_confidence(row: pd.Series, pred: int) -> float:
    """Legacy confidence scoring (backward compatible)."""
    score = 0.5
    rsi = row.get("rsi", 50)
    macd_hist = row.get("macd_hist", 0)

    if pred == 2:  # BUY
        if rsi < 40:
            score += 0.15
        if macd_hist > 0:
            score += 0.15
    elif pred == 0:  # SELL
        if rsi > 60:
            score += 0.15
        if macd_hist < 0:
            score += 0.15
    else:
        if 40 <= rsi <= 60:
            score += 0.1
        if abs(macd_hist) < 0.01:
            score += 0.1

    return min(max(score, 0.0), 1.0)


def _compute_ai_confidence(
    row: pd.Series, pred: int,
    model_conf: float = 0.5,
    mtf: dict | None = None,
) -> float:
    """Advanced AI confidence scoring combining multiple signal dimensions.

    Dimensions:
        1. Model probability (from softmax/RF)
        2. Technical indicator alignment with prediction
        3. Trend strength confirmation (ADX)
        4. Multi-timeframe alignment
        5. Volume confirmation
        6. Regime alignment
    """
    scores = []
    weights = []

    # 1. Model confidence (weight: 30%)
    scores.append(model_conf)
    weights.append(0.30)

    # 2. Technical indicator alignment (weight: 25%)
    tech_score = _technical_alignment_score(row, pred)
    scores.append(tech_score)
    weights.append(0.25)

    # 3. Trend strength via ADX (weight: 15%)
    adx = row.get("adx", 25)
    if pred in (0, 2):  # directional signal
        trend_conf = min(adx / 50.0, 1.0)
    else:  # HOLD — low ADX is confirming
        trend_conf = max(1.0 - adx / 50.0, 0.0)
    scores.append(trend_conf)
    weights.append(0.15)

    # 4. Multi-timeframe alignment (weight: 15%)
    if mtf:
        mtf_alignment = mtf.get("alignment_score", 0.5)
    else:
        mtf_alignment = 0.5
    scores.append(mtf_alignment)
    weights.append(0.15)

    # 5. Volume confirmation (weight: 10%)
    vol_ratio = row.get("volume_ratio", 1.0)
    vol_trend = row.get("vol_price_trend", 0)
    if pred == 2:  # BUY
        vol_conf = min(vol_ratio / 2.0, 1.0) * (0.5 + 0.5 * min(max(vol_trend, 0), 1))
    elif pred == 0:  # SELL
        vol_conf = min(vol_ratio / 2.0, 1.0) * (0.5 + 0.5 * min(max(-vol_trend, 0), 1))
    else:
        vol_conf = 0.5
    scores.append(vol_conf)
    weights.append(0.10)

    # 6. Stochastic RSI confirmation (weight: 5%)
    stoch_k = row.get("stoch_rsi_k", 0.5)
    if pred == 2 and stoch_k < 0.2:  # oversold = good for buy
        stoch_conf = 0.9
    elif pred == 0 and stoch_k > 0.8:  # overbought = good for sell
        stoch_conf = 0.9
    elif pred == 1 and 0.3 <= stoch_k <= 0.7:
        stoch_conf = 0.7
    else:
        stoch_conf = 0.4
    scores.append(stoch_conf)
    weights.append(0.05)

    final = sum(s * w for s, w in zip(scores, weights))
    return min(max(final, 0.0), 1.0)


def _technical_alignment_score(row: pd.Series, pred: int) -> float:
    """Score how well technical indicators align with the prediction."""
    alignment = 0.0
    count = 0

    rsi = row.get("rsi", 50)
    macd_hist = row.get("macd_hist", 0)
    bb_pct = row.get("bb_pct", 0.5)
    di_cross = row.get("di_crossover", 0)
    ichimoku_above = row.get("ichimoku_above_cloud", 0)
    ichimoku_below = row.get("ichimoku_below_cloud", 0)
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
        if ichimoku_below > 0: alignment += 1.0; count += 1
        else: count += 1
        if engulfing < 0: alignment += 0.8; count += 1
        else: count += 1

    else:  # HOLD
        if 40 <= rsi <= 60: alignment += 1.0; count += 1
        else: count += 1
        if abs(macd_hist) < 0.01: alignment += 1.0; count += 1
        else: count += 1
        if 0.3 <= bb_pct <= 0.7: alignment += 1.0; count += 1
        else: count += 1
        count += 3  # placeholder for other indicators

    return alignment / max(count, 1)


def _multi_timeframe_analysis(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """Analyze signals across short (5), medium (20), and long (50) timeframes.

    Returns a dict mapping row index -> timeframe alignment info.
    """
    if len(df) < 50:
        return {}

    results = {}
    for idx in range(50, len(df)):
        short_window = df.iloc[idx - 5:idx + 1]
        medium_window = df.iloc[idx - 20:idx + 1]
        long_window = df.iloc[idx - 50:idx + 1]

        # Short-term signal
        short_rsi = short_window["rsi"].iloc[-1] if "rsi" in short_window else 50
        short_macd = short_window["macd_hist"].iloc[-1] if "macd_hist" in short_window else 0
        short_signal = 2 if (short_rsi < 40 and short_macd > 0) else (0 if (short_rsi > 60 and short_macd < 0) else 1)

        # Medium-term signal
        med_slope = (medium_window["close"].iloc[-1] - medium_window["close"].iloc[0]) / medium_window["close"].iloc[0]
        med_signal = 2 if med_slope > 0.02 else (0 if med_slope < -0.02 else 1)

        # Long-term signal
        long_slope = (long_window["close"].iloc[-1] - long_window["close"].iloc[0]) / long_window["close"].iloc[0]
        long_signal = 2 if long_slope > 0.05 else (0 if long_slope < -0.05 else 1)

        # Alignment: how many timeframes agree
        signals = [short_signal, med_signal, long_signal]
        from collections import Counter
        counts = Counter(signals)
        dominant = counts.most_common(1)[0]
        alignment_score = dominant[1] / 3.0

        results[idx] = {
            "short_signal": LABEL_MAP[short_signal],
            "medium_signal": LABEL_MAP[med_signal],
            "long_signal": LABEL_MAP[long_signal],
            "dominant_signal": LABEL_MAP[dominant[0]],
            "alignment_score": round(alignment_score, 4),
            "timeframes_aligned": dominant[1],
        }

    return results


def _compute_signal_quality(row: pd.Series, pred: int, confidence: float) -> str:
    """Classify signal quality as A/B/C/D based on confidence and confirmation."""
    adx = row.get("adx", 25)
    vol_ratio = row.get("volume_ratio", 1.0)

    quality_score = confidence * 0.5
    if adx > 25:
        quality_score += 0.2
    if vol_ratio > 1.2:
        quality_score += 0.15
    if row.get("regime_trend_strength", 0) > 0.3:
        quality_score += 0.15

    if quality_score >= 0.75:
        return "A"
    elif quality_score >= 0.55:
        return "B"
    elif quality_score >= 0.35:
        return "C"
    return "D"


def _generate_signal_explanation(row: pd.Series, pred: int, confidence: float) -> List[str]:
    """Generate human-readable reasons for the signal."""
    reasons = []
    signal = LABEL_MAP.get(pred, "HOLD")

    rsi = row.get("rsi", 50)
    macd_hist = row.get("macd_hist", 0)
    adx = row.get("adx", 25)
    bb_pct = row.get("bb_pct", 0.5)
    stoch_k = row.get("stoch_rsi_k", 0.5)
    vol_ratio = row.get("volume_ratio", 1.0)
    ichimoku_above = row.get("ichimoku_above_cloud", 0)
    vwap_dev = row.get("vwap_deviation", 0)

    if pred == 2:
        if rsi < 30: reasons.append(f"RSI deeply oversold ({rsi:.1f})")
        elif rsi < 40: reasons.append(f"RSI approaching oversold ({rsi:.1f})")
        if macd_hist > 0: reasons.append("MACD histogram turning positive")
        if stoch_k < 0.2: reasons.append(f"Stochastic RSI oversold ({stoch_k:.2f})")
        if bb_pct < 0.1: reasons.append("Price below lower Bollinger Band")
        if ichimoku_above > 0: reasons.append("Price above Ichimoku cloud")
        if vwap_dev < -0.02: reasons.append(f"Trading below VWAP ({vwap_dev:.2%})")

    elif pred == 0:
        if rsi > 70: reasons.append(f"RSI deeply overbought ({rsi:.1f})")
        elif rsi > 60: reasons.append(f"RSI approaching overbought ({rsi:.1f})")
        if macd_hist < 0: reasons.append("MACD histogram turning negative")
        if stoch_k > 0.8: reasons.append(f"Stochastic RSI overbought ({stoch_k:.2f})")
        if bb_pct > 0.9: reasons.append("Price above upper Bollinger Band")
        if vwap_dev > 0.02: reasons.append(f"Trading above VWAP ({vwap_dev:.2%})")

    if adx > 30: reasons.append(f"Strong trend (ADX={adx:.0f})")
    elif adx < 20: reasons.append(f"Weak trend/ranging (ADX={adx:.0f})")
    if vol_ratio > 2.0: reasons.append(f"Volume spike ({vol_ratio:.1f}x avg)")

    if confidence > 0.75: reasons.append("High model confidence")
    elif confidence < 0.4: reasons.append("Low model confidence — cautious signal")

    return reasons if reasons else [f"{signal} signal with moderate indicator alignment"]

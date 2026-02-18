"""Adaptive ensemble — dynamically weights RF, LSTM, and Transformer predictions.

Instead of simple majority voting, this module tracks each model's recent
performance and adjusts weights accordingly using exponential moving average
of per-model accuracy. Models that perform better in the current market
regime receive higher weights.

Features:
    - Dynamic weight adjustment based on rolling accuracy
    - Regime-aware model selection (some models excel in certain regimes)
    - Calibrated confidence scores from probability averaging
    - Disagreement detection (signals when models conflict)
    - Performance tracking for continuous improvement
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.training.random_forest import FEATURE_COLS as RF_FEATURE_COLS
from ml.training.random_forest import predict_rf

try:
    from ml.training.lstm_model import predict_lstm
except ImportError:
    predict_lstm = None

try:
    from ml.training.transformer_model import predict_transformer, predict_transformer_with_confidence
except ImportError:
    predict_transformer = None
    predict_transformer_with_confidence = None

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


class ModelPerformanceTracker:
    """Track per-model prediction accuracy over a rolling window."""

    def __init__(self, window: int = 50):
        self.window = window
        self._records: Dict[str, deque] = {}

    def record(self, model_type: str, predicted: int, actual: int) -> None:
        if model_type not in self._records:
            self._records[model_type] = deque(maxlen=self.window)
        self._records[model_type].append(int(predicted == actual))

    def get_accuracy(self, model_type: str) -> float:
        records = self._records.get(model_type, [])
        if not records:
            return 0.5  # prior: assume 50% if no data
        return sum(records) / len(records)

    def get_all_accuracies(self) -> Dict[str, float]:
        return {mt: self.get_accuracy(mt) for mt in self._records}


# Global tracker
_tracker = ModelPerformanceTracker()


def get_performance_tracker() -> ModelPerformanceTracker:
    return _tracker


def compute_dynamic_weights(
    available_models: List[str],
    regime: str = "mean_reverting",
) -> Dict[str, float]:
    """Compute model weights based on recent performance and market regime.

    Models that perform better recently get higher weight. Regime-specific
    biases further adjust weights (e.g., Transformer may excel in trending
    markets while RF may be more robust in mean-reverting conditions).
    """
    # Base weights from recent accuracy
    accuracies = {m: _tracker.get_accuracy(m) for m in available_models}

    # Regime-specific biases
    regime_biases = {
        "trending_up": {"rf": 0.8, "lstm": 1.1, "transformer": 1.3},
        "trending_down": {"rf": 0.9, "lstm": 1.1, "transformer": 1.2},
        "mean_reverting": {"rf": 1.2, "lstm": 0.9, "transformer": 1.0},
        "high_volatility": {"rf": 1.1, "lstm": 0.8, "transformer": 1.1},
        "low_volatility": {"rf": 1.0, "lstm": 1.0, "transformer": 1.0},
    }
    biases = regime_biases.get(regime, {m: 1.0 for m in available_models})

    # Weighted score = accuracy * regime_bias
    raw_weights = {}
    for m in available_models:
        acc = accuracies.get(m, 0.5)
        bias = biases.get(m, 1.0)
        raw_weights[m] = acc * bias

    total = sum(raw_weights.values()) or 1.0
    return {m: w / total for m, w in raw_weights.items()}


def adaptive_ensemble_predict(
    df: pd.DataFrame,
    rf_path: Optional[str] = None,
    lstm_path: Optional[str] = None,
    transformer_path: Optional[str] = None,
    regime: str = "mean_reverting",
) -> Dict[str, Any]:
    """Run adaptive ensemble prediction across all available models.

    Returns:
        Dict with keys:
            predictions: final ensemble predictions
            confidence: calibrated confidence scores
            model_weights: dynamic weights used
            model_predictions: per-model raw predictions
            disagreement: bool indicating model disagreement
            signal_strength: combined signal strength metric
    """
    available_models = []
    model_preds = {}
    model_confidences = {}

    # RF predictions
    if rf_path:
        try:
            rf_pred = predict_rf(rf_path, df)
            model_preds["rf"] = rf_pred
            # RF doesn't natively output probabilities easily here, so use uniform confidence
            model_confidences["rf"] = np.full(len(rf_pred), 0.6)
            available_models.append("rf")
        except Exception as e:
            logger.warning("RF prediction failed: %s", e)

    # LSTM predictions
    if lstm_path and predict_lstm is not None:
        try:
            lstm_pred = predict_lstm(lstm_path, df)
            model_preds["lstm"] = lstm_pred
            model_confidences["lstm"] = np.full(len(lstm_pred), 0.6)
            available_models.append("lstm")
        except Exception as e:
            logger.warning("LSTM prediction failed: %s", e)

    # Transformer predictions (with confidence)
    if transformer_path and predict_transformer_with_confidence is not None:
        try:
            t_pred, t_conf, t_attn = predict_transformer_with_confidence(transformer_path, df)
            model_preds["transformer"] = t_pred
            model_confidences["transformer"] = t_conf
            available_models.append("transformer")
        except Exception as e:
            logger.warning("Transformer prediction failed: %s", e)

    if not available_models:
        return {
            "predictions": np.array([]),
            "confidence": np.array([]),
            "model_weights": {},
            "model_predictions": {},
            "disagreement": False,
            "signal_strength": 0.0,
        }

    # Compute dynamic weights
    weights = compute_dynamic_weights(available_models, regime)

    # Align prediction lengths (models may produce different lengths)
    min_len = min(len(model_preds[m]) for m in available_models)
    for m in available_models:
        model_preds[m] = model_preds[m][-min_len:]
        model_confidences[m] = model_confidences[m][-min_len:]

    # Weighted voting
    ensemble_scores = np.zeros((min_len, 3))  # 3 classes
    for m in available_models:
        w = weights[m]
        for i in range(min_len):
            pred_class = int(model_preds[m][i])
            conf = model_confidences[m][i]
            if 0 <= pred_class <= 2:
                ensemble_scores[i, pred_class] += w * conf

    # Normalize to probabilities
    row_sums = ensemble_scores.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    ensemble_probs = ensemble_scores / row_sums

    final_preds = ensemble_probs.argmax(axis=1)
    final_confidence = ensemble_probs.max(axis=1)

    # Disagreement detection
    if len(available_models) >= 2:
        agreements = np.zeros(min_len)
        for i in range(min_len):
            votes = [int(model_preds[m][i]) for m in available_models]
            most_common = max(set(votes), key=votes.count)
            agreements[i] = votes.count(most_common) / len(votes)
        avg_agreement = float(agreements.mean())
        has_disagreement = avg_agreement < 0.7
    else:
        avg_agreement = 1.0
        has_disagreement = False

    # Signal strength: combines confidence with agreement
    signal_strength = float(final_confidence.mean() * avg_agreement)

    return {
        "predictions": final_preds,
        "confidence": final_confidence,
        "model_weights": weights,
        "model_predictions": {m: model_preds[m].tolist() for m in available_models},
        "disagreement": has_disagreement,
        "agreement_score": round(avg_agreement, 4),
        "signal_strength": round(signal_strength, 4),
        "ensemble_probabilities": ensemble_probs,
    }


def record_outcome(model_preds: Dict[str, int], actual: int) -> None:
    """Record actual outcome to update model weights over time."""
    for model_type, pred in model_preds.items():
        _tracker.record(model_type, pred, actual)

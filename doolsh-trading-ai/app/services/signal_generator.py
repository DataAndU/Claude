"""Signal generation service — combines ML predictions with risk scoring."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from ml.features.engineering import build_features
from ml.training.lstm_model import predict_lstm
from ml.training.random_forest import FEATURE_COLS, predict_rf

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


def generate_signals(
    df: pd.DataFrame,
    model_path: str,
    model_type: str = "rf",
    ensemble: bool = False,
    rf_path: str | None = None,
    lstm_path: str | None = None,
) -> List[Dict]:
    featured = build_features(df)
    if featured.empty:
        return []

    if ensemble and rf_path and lstm_path:
        rf_preds = predict_rf(rf_path, featured)
        lstm_preds = predict_lstm(lstm_path, featured)
        min_len = min(len(rf_preds), len(lstm_preds))
        rf_preds = rf_preds[-min_len:]
        lstm_preds = lstm_preds[-min_len:]
        combined = np.round((rf_preds + lstm_preds) / 2).astype(int)
        featured = featured.iloc[-min_len:].reset_index(drop=True)
        preds = combined
    elif model_type == "lstm":
        preds = predict_lstm(model_path, featured)
        featured = featured.iloc[-len(preds):].reset_index(drop=True)
    else:
        preds = predict_rf(model_path, featured)

    signals: List[Dict] = []
    for idx, pred in enumerate(preds):
        row = featured.iloc[idx]
        confidence = _compute_confidence(row, int(pred))
        signals.append({
            "index": idx,
            "signal": LABEL_MAP.get(int(pred), "HOLD"),
            "confidence": round(float(confidence), 4),
            "price": round(float(row["close"]), 4) if "close" in row.index else 0.0,
            "risk_score": round(float(row.get("risk_score", 0.5)), 4),
            "rsi": round(float(row.get("rsi", 50)), 2),
            "macd_hist": round(float(row.get("macd_hist", 0)), 6),
        })
    return signals


def _compute_confidence(row: pd.Series, pred: int) -> float:
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

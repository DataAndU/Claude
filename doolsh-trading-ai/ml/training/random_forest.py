"""Random Forest classifier for market signal prediction."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

FEATURE_COLS = [
    "sma_5", "sma_10", "sma_20", "ema_5", "ema_10", "ema_20",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_pct", "bb_width",
    "return_1d", "return_5d", "volatility", "risk_score",
]


def _label_signal(row: pd.Series, threshold: float = 0.005) -> int:
    """Map future return to signal label: 0=SELL, 1=HOLD, 2=BUY."""
    ret = row.get("future_return", 0.0)
    if ret > threshold:
        return 2  # BUY
    elif ret < -threshold:
        return 0  # SELL
    return 1  # HOLD


def prepare_dataset(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.005,
) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["future_return"] = df["close"].pct_change(horizon).shift(-horizon)
    df["label"] = df.apply(lambda r: _label_signal(r, threshold), axis=1)
    df.dropna(inplace=True)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available]
    y = df["label"]
    return X, y


def train_random_forest(
    df: pd.DataFrame,
    symbol: str,
    horizon: int = 5,
    n_estimators: int = 200,
    max_depth: int | None = 12,
    version: str = "v1",
) -> Dict[str, Any]:
    X, y = prepare_dataset(df, horizon=horizon)

    split_idx = int(len(X) * settings.default_train_test_split)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    # Time-series cross validation
    tscv = TimeSeriesSplit(n_splits=settings.cross_validation_folds)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=tscv, scoring="accuracy")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, target_names=["SELL", "HOLD", "BUY"])

    # Persist
    save_dir = Path(settings.model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"rf_{symbol}_{version}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    logger.info("RF model saved to %s | accuracy=%.4f", model_path, acc)

    return {
        "model_type": "rf",
        "symbol": symbol,
        "version": version,
        "file_path": str(model_path),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
        "classification_report": report,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "horizon": horizon,
        },
        "feature_importance": dict(
            zip(X_train.columns.tolist(), clf.feature_importances_.tolist())
        ),
    }


def predict_rf(model_path: str, features: pd.DataFrame) -> np.ndarray:
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    available = [c for c in FEATURE_COLS if c in features.columns]
    return clf.predict(features[available])

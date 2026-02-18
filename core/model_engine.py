"""ML model engine — RandomForest, XGBoost, lightweight LSTM, auto-retrain.

All models run CPU-only with small memory footprint.
Provides feature importance output and model persistence.
"""

from __future__ import annotations

import gc
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from core.config import load_config
from core.feature_engine import FEATURE_COLS

logger = logging.getLogger(__name__)

MODEL_DIR = Path("database/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─── Label Creation ───────────────────────────────────────────────────────────


def create_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.005) -> pd.Series:
    """Create classification labels: 0=SELL, 1=HOLD, 2=BUY based on future returns."""
    future_ret = df["close"].pct_change(horizon).shift(-horizon)
    labels = pd.Series(1, index=df.index, dtype=int)  # default HOLD
    labels[future_ret > threshold] = 2   # BUY
    labels[future_ret < -threshold] = 0  # SELL
    return labels


def prepare_dataset(
    df: pd.DataFrame, horizon: int = 5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and labels y from featured DataFrame."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    labels = create_labels(df, horizon=horizon)
    mask = labels.notna() & df[available].notna().all(axis=1)
    X = df.loc[mask, available].copy()
    y = labels[mask].copy()
    return X, y


# ─── Random Forest ────────────────────────────────────────────────────────────


def train_random_forest(
    df: pd.DataFrame, symbol: str, timeframe: str = "1h",
) -> Dict[str, Any]:
    """Train a Random Forest classifier and save to disk."""
    cfg = load_config()
    X, y = prepare_dataset(df, horizon=cfg.prediction_horizon)
    if len(X) < 100:
        raise ValueError(f"Not enough data: {len(X)} rows")

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    params = cfg.rf_params
    clf = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 10),
        random_state=42, n_jobs=-1, class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Feature importance
    importance = dict(zip(X_train.columns.tolist(), clf.feature_importances_.tolist()))

    # Save
    model_path = MODEL_DIR / f"rf_{symbol.replace('/', '_')}_{timeframe}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "features": X_train.columns.tolist()}, f)

    logger.info("RF trained: acc=%.4f f1=%.4f path=%s", acc, f1, model_path)
    gc.collect()
    return {
        "model_type": "random_forest", "symbol": symbol, "timeframe": timeframe,
        "accuracy": float(acc), "f1": float(f1), "file_path": str(model_path),
        "feature_importance": importance,
    }


def predict_rf(model_path: str, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Predict using saved RF model. Returns (predictions, probabilities)."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    clf = data["model"]
    features = data["features"]
    available = [c for c in features if c in df.columns]
    X = df[available].values
    preds = clf.predict(X)
    probs = clf.predict_proba(X).max(axis=1)
    return preds, probs


# ─── XGBoost ──────────────────────────────────────────────────────────────────


def train_xgboost(
    df: pd.DataFrame, symbol: str, timeframe: str = "1h",
) -> Dict[str, Any]:
    """Train an XGBoost classifier and save to disk."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("xgboost not installed")

    cfg = load_config()
    X, y = prepare_dataset(df, horizon=cfg.prediction_horizon)
    if len(X) < 100:
        raise ValueError(f"Not enough data: {len(X)} rows")

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    params = cfg.xgb_params
    clf = XGBClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.1),
        use_label_encoder=False, eval_metric="mlogloss",
        tree_method="hist", n_jobs=-1, random_state=42,
    )
    clf.fit(X_train, y_train, verbose=False)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    importance = dict(zip(X_train.columns.tolist(), clf.feature_importances_.tolist()))

    model_path = MODEL_DIR / f"xgb_{symbol.replace('/', '_')}_{timeframe}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "features": X_train.columns.tolist()}, f)

    logger.info("XGB trained: acc=%.4f f1=%.4f path=%s", acc, f1, model_path)
    gc.collect()
    return {
        "model_type": "xgboost", "symbol": symbol, "timeframe": timeframe,
        "accuracy": float(acc), "f1": float(f1), "file_path": str(model_path),
        "feature_importance": importance,
    }


def predict_xgb(model_path: str, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Predict using saved XGBoost model."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    clf = data["model"]
    features = data["features"]
    available = [c for c in features if c in df.columns]
    X = df[available].values
    preds = clf.predict(X)
    probs = clf.predict_proba(X).max(axis=1)
    return preds, probs


# ─── Lightweight LSTM ─────────────────────────────────────────────────────────


def train_lstm(
    df: pd.DataFrame, symbol: str, timeframe: str = "1h",
) -> Dict[str, Any]:
    """Train a lightweight LSTM (CPU-only, small hidden size)."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        raise ImportError("PyTorch not installed — LSTM unavailable")

    cfg = load_config()
    lp = cfg.lstm_params
    seq_len = lp.get("seq_len", 20)
    hidden = lp.get("hidden_size", 64)
    epochs = lp.get("epochs", 30)
    batch_size = lp.get("batch_size", 32)

    X, y = prepare_dataset(df, horizon=cfg.prediction_horizon)
    if len(X) < seq_len + 50:
        raise ValueError(f"Not enough data for LSTM: {len(X)} rows")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # Create sequences
    Xs, ys = [], []
    for i in range(len(X_scaled) - seq_len):
        Xs.append(X_scaled[i:i + seq_len])
        ys.append(y.iloc[i + seq_len])
    Xs = np.array(Xs)
    ys = np.array(ys)

    split = int(len(Xs) * 0.8)
    X_train, X_test = Xs[:split], Xs[split:]
    y_train, y_test = ys[:split], ys[split:]

    class SimpleLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes=3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    device = torch.device("cpu")
    model = SimpleLSTM(X_train.shape[2], hidden).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            logger.info("LSTM epoch %d/%d loss=%.4f", epoch, epochs, total_loss / len(loader))

    model.eval()
    with torch.no_grad():
        test_out = model(torch.tensor(X_test, dtype=torch.float32))
        preds = test_out.argmax(dim=1).numpy()

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    model_path = MODEL_DIR / f"lstm_{symbol.replace('/', '_')}_{timeframe}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "features": X.columns.tolist(),
        "input_size": X_train.shape[2],
        "hidden_size": hidden,
        "seq_len": seq_len,
    }, model_path)

    logger.info("LSTM trained: acc=%.4f f1=%.4f path=%s", acc, f1, model_path)
    gc.collect()
    return {
        "model_type": "lstm", "symbol": symbol, "timeframe": timeframe,
        "accuracy": float(acc), "f1": float(f1), "file_path": str(model_path),
        "feature_importance": {},
    }


def predict_lstm(model_path: str, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Predict using saved LSTM model."""
    import torch
    import torch.nn as nn

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    features = ckpt["features"]
    seq_len = ckpt["seq_len"]

    available = [c for c in features if c in df.columns]
    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt["scaler_mean"])
    scaler.scale_ = np.array(ckpt["scaler_scale"])
    X_scaled = scaler.transform(df[available].values)

    if len(X_scaled) < seq_len:
        return np.array([]), np.array([])

    Xs = np.array([X_scaled[i:i + seq_len] for i in range(len(X_scaled) - seq_len + 1)])

    class SimpleLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes=3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = SimpleLSTM(ckpt["input_size"], ckpt["hidden_size"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(Xs, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
        preds = logits.argmax(dim=1).numpy()

    return preds, probs.max(axis=1)


# ─── Unified Interface ────────────────────────────────────────────────────────

LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


def train_model(
    df: pd.DataFrame, symbol: str, timeframe: str = "1h", model_type: str = "random_forest",
) -> Dict[str, Any]:
    """Train any supported model type."""
    if model_type == "random_forest":
        return train_random_forest(df, symbol, timeframe)
    elif model_type == "xgboost":
        return train_xgboost(df, symbol, timeframe)
    elif model_type == "lstm":
        return train_lstm(df, symbol, timeframe)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict(
    model_path: str, df: pd.DataFrame, model_type: str = "random_forest",
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict using any supported model type."""
    if model_type in ("random_forest", "rf"):
        return predict_rf(model_path, df)
    elif model_type in ("xgboost", "xgb"):
        return predict_xgb(model_path, df)
    elif model_type == "lstm":
        return predict_lstm(model_path, df)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_latest_model(symbol: str, timeframe: str, model_type: str) -> Optional[str]:
    """Find the latest saved model file for a given symbol/timeframe."""
    prefix_map = {"random_forest": "rf", "xgboost": "xgb", "lstm": "lstm"}
    prefix = prefix_map.get(model_type, model_type)
    sym_clean = symbol.replace("/", "_")
    ext = ".pt" if model_type == "lstm" else ".pkl"
    path = MODEL_DIR / f"{prefix}_{sym_clean}_{timeframe}{ext}"
    return str(path) if path.exists() else None

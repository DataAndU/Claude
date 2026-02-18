"""LSTM model for sequential market signal prediction using PyTorch."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

FEATURE_COLS = [
    "sma_5", "sma_10", "sma_20", "ema_5", "ema_10", "ema_20",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_pct", "bb_width", "return_1d", "return_5d", "volatility", "risk_score",
]


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is not installed. Install with: "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )


def _create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def _label_signal(returns: pd.Series, threshold: float = 0.005) -> np.ndarray:
    labels = np.ones(len(returns), dtype=np.int64)
    labels[returns > threshold] = 2
    labels[returns < -threshold] = 0
    return labels


def _build_lstm_model(input_size: int, hidden_size: int = 128, num_layers: int = 2,
                      num_classes: int = 3, dropout: float = 0.3):
    """Build an LSTM classifier. Only callable when torch is available."""
    _require_torch()

    class LSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(self.dropout(lstm_out[:, -1, :]))

    return LSTMClassifier()


def train_lstm(
    df: pd.DataFrame, symbol: str, seq_len: int = 30,
    hidden_size: int = 128, num_layers: int = 2, epochs: int = 50,
    batch_size: int = 64, lr: float = 1e-3, horizon: int = 5, version: str = "v1",
) -> Dict[str, Any]:
    _require_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    available = [c for c in FEATURE_COLS if c in df.columns]
    data = df[available].copy()
    future_return = df["close"].pct_change(horizon).shift(-horizon)
    data = data.iloc[:len(future_return)]
    future_return = future_return.iloc[:len(data)]
    mask = future_return.notna() & data.notna().all(axis=1)
    data = data[mask].reset_index(drop=True)
    future_return = future_return[mask].reset_index(drop=True)
    labels = _label_signal(future_return)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.values)
    X_seq, y_seq = _create_sequences(X_scaled, labels, seq_len)

    split = int(len(X_seq) * settings.default_train_test_split)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    model = _build_lstm_model(input_size=len(available), hidden_size=hidden_size, num_layers=num_layers).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 0.5, 1.0], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)
        scheduler_lr.step(epoch_loss)
        if epoch % 10 == 0:
            logger.info("Epoch %d/%d loss=%.6f", epoch, epochs, epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    save_dir = Path(settings.model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"lstm_{symbol}_{version}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "feature_cols": available, "seq_len": seq_len,
        "hidden_size": hidden_size, "num_layers": num_layers,
    }, model_path)

    return {
        "model_type": "lstm", "symbol": symbol, "version": version,
        "file_path": str(model_path), "accuracy": float(acc),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "hyperparameters": {"seq_len": seq_len, "hidden_size": hidden_size, "num_layers": num_layers,
                           "epochs": epochs, "batch_size": batch_size, "lr": lr, "horizon": horizon},
    }


def predict_lstm(model_path: str, df: pd.DataFrame) -> np.ndarray:
    _require_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    feature_cols = checkpoint["feature_cols"]
    seq_len = checkpoint["seq_len"]

    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint["scaler_mean"])
    scaler.scale_ = np.array(checkpoint["scaler_scale"])

    data = scaler.transform(df[feature_cols].values)
    if len(data) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows, got {len(data)}")

    X = np.array([data[i:i + seq_len] for i in range(len(data) - seq_len + 1)])
    model = _build_lstm_model(input_size=len(feature_cols), hidden_size=checkpoint["hidden_size"],
                              num_layers=checkpoint["num_layers"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy()

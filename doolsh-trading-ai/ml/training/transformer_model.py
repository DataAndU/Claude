"""Transformer attention model for sequential market signal prediction.

Uses multi-head self-attention to capture long-range dependencies in price
action patterns. Supports positional encoding and temporal attention for
identifying which historical time steps matter most for current predictions.

Key advantages over LSTM:
    - Parallelizable training (no sequential dependency)
    - Better at capturing long-range patterns
    - Attention weights provide interpretable signal explanations
    - More robust to varying sequence lengths
"""

from __future__ import annotations

import logging
import math
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
    "stoch_rsi_k", "stoch_rsi_d", "adx", "di_crossover",
    "vwap_deviation", "volume_ratio", "vol_price_trend",
    "ichimoku_tk_cross", "ichimoku_cloud_width",
    "trend_slope_20", "hurst_proxy", "regime_trend_strength",
    "mean_reversion_score", "hammer_score", "engulfing_score",
]


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is not installed. Install with: "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time-series transformer."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalAttentionBlock(nn.Module):
    """Multi-head attention block that learns which time steps matter most."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_weights


class MarketTransformer(nn.Module):
    """Transformer-based model for market signal classification.

    Architecture:
        Input projection → Positional encoding → N x Temporal attention blocks
        → Global average pooling → Classification head
    """

    def __init__(self, input_size: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 3, num_classes: int = 3, dropout: float = 0.2,
                 max_seq_len: int = 100):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        self.attention_blocks = nn.ModuleList([
            TemporalAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self._attn_weights = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        self._attn_weights = []
        for block in self.attention_blocks:
            x, weights = block(x)
            self._attn_weights.append(weights)
        # Global average pooling across time dimension
        x = x.mean(dim=1)
        return self.classifier(x)

    def get_attention_weights(self) -> list:
        return [w.detach().cpu().numpy() for w in self._attn_weights]


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


def train_transformer(
    df: pd.DataFrame, symbol: str, seq_len: int = 30,
    d_model: int = 64, n_heads: int = 4, n_layers: int = 3,
    epochs: int = 60, batch_size: int = 64, lr: float = 5e-4,
    horizon: int = 5, version: str = "v1", dropout: float = 0.2,
) -> Dict[str, Any]:
    """Train a Transformer attention model for market signal prediction."""
    _require_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    available = [c for c in FEATURE_COLS if c in df.columns]
    if not available:
        raise ValueError("No feature columns found in DataFrame")

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

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    model = MarketTransformer(
        input_size=len(available), d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, dropout=dropout, max_seq_len=seq_len,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 0.5, 1.0], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_loss = float("inf")
    patience_counter = 0
    patience_limit = 15

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
        scheduler_lr.step()

        if epoch % 10 == 0:
            logger.info("Transformer Epoch %d/%d loss=%.6f", epoch, epochs, epoch_loss)

        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.info("Early stopping at epoch %d", epoch)
                break

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(test_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    save_dir = Path(settings.model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"transformer_{symbol}_{version}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_cols": available,
        "seq_len": seq_len,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout,
    }, model_path)

    logger.info("Transformer model saved to %s | accuracy=%.4f f1=%.4f", model_path, acc, f1)
    return {
        "model_type": "transformer",
        "symbol": symbol,
        "version": version,
        "file_path": str(model_path),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "hyperparameters": {
            "seq_len": seq_len, "d_model": d_model, "n_heads": n_heads,
            "n_layers": n_layers, "epochs": epochs, "batch_size": batch_size,
            "lr": lr, "horizon": horizon, "dropout": dropout,
        },
        "avg_confidence": float(probs.max(axis=1).mean()),
    }


def predict_transformer(model_path: str, df: pd.DataFrame) -> np.ndarray:
    """Generate predictions using a trained Transformer model."""
    _require_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    feature_cols = checkpoint["feature_cols"]
    seq_len = checkpoint["seq_len"]

    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint["scaler_mean"])
    scaler.scale_ = np.array(checkpoint["scaler_scale"])

    available = [c for c in feature_cols if c in df.columns]
    data = scaler.transform(df[available].values)
    if len(data) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows, got {len(data)}")

    X = np.array([data[i:i + seq_len] for i in range(len(data) - seq_len + 1)])
    model = MarketTransformer(
        input_size=len(available),
        d_model=checkpoint["d_model"],
        n_heads=checkpoint["n_heads"],
        n_layers=checkpoint["n_layers"],
        dropout=checkpoint.get("dropout", 0.2),
        max_seq_len=seq_len,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        return logits.argmax(dim=1).cpu().numpy()


def predict_transformer_with_confidence(model_path: str, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """Generate predictions with confidence scores and attention weights.

    Returns:
        (predictions, confidence_scores, attention_weights_per_layer)
    """
    _require_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    feature_cols = checkpoint["feature_cols"]
    seq_len = checkpoint["seq_len"]

    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint["scaler_mean"])
    scaler.scale_ = np.array(checkpoint["scaler_scale"])

    available = [c for c in feature_cols if c in df.columns]
    data = scaler.transform(df[available].values)
    if len(data) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows, got {len(data)}")

    X = np.array([data[i:i + seq_len] for i in range(len(data) - seq_len + 1)])
    model = MarketTransformer(
        input_size=len(available),
        d_model=checkpoint["d_model"],
        n_heads=checkpoint["n_heads"],
        n_layers=checkpoint["n_layers"],
        dropout=checkpoint.get("dropout", 0.2),
        max_seq_len=seq_len,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        confidence = probs.max(axis=1)
        attn_weights = model.get_attention_weights()

    return preds, confidence, attn_weights

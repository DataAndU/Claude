"""Tests for the feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features.engineering import (
    add_bollinger_bands,
    add_macd,
    add_moving_averages,
    add_returns,
    add_rsi,
    build_features,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="B"),
            "open": close + np.random.randn(n) * 0.2,
            "high": close + abs(np.random.randn(n) * 0.5),
            "low": close - abs(np.random.randn(n) * 0.5),
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, n),
        }
    )
    return df


def test_add_moving_averages(sample_ohlcv: pd.DataFrame):
    df = add_moving_averages(sample_ohlcv.copy())
    assert "sma_5" in df.columns
    assert "ema_20" in df.columns
    assert df["sma_5"].notna().sum() > 0


def test_add_rsi(sample_ohlcv: pd.DataFrame):
    df = add_rsi(sample_ohlcv.copy())
    assert "rsi" in df.columns
    valid = df["rsi"].dropna()
    assert valid.min() >= 0
    assert valid.max() <= 100


def test_add_macd(sample_ohlcv: pd.DataFrame):
    df = add_macd(sample_ohlcv.copy())
    assert "macd" in df.columns
    assert "macd_signal" in df.columns
    assert "macd_hist" in df.columns


def test_add_bollinger_bands(sample_ohlcv: pd.DataFrame):
    df = add_bollinger_bands(sample_ohlcv.copy())
    assert "bb_upper" in df.columns
    assert "bb_lower" in df.columns
    valid = df.dropna(subset=["bb_upper", "bb_lower"])
    assert (valid["bb_upper"] >= valid["bb_lower"]).all()


def test_add_returns(sample_ohlcv: pd.DataFrame):
    df = add_returns(sample_ohlcv.copy())
    assert "return_1d" in df.columns
    assert "log_return" in df.columns


def test_build_features(sample_ohlcv: pd.DataFrame):
    df = build_features(sample_ohlcv)
    assert len(df) > 0
    assert "risk_score" in df.columns
    assert df["risk_score"].between(0, 1).all()

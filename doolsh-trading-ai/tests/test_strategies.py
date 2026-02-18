"""Tests for built-in strategy implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategies.base import (
    DualMovingAverageCrossover,
    MACDCrossoverStrategy,
    RSIMeanReversionStrategy,
    STRATEGY_REGISTRY,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="B"),
            "open": close + np.random.randn(n) * 0.2,
            "high": close + abs(np.random.randn(n) * 0.5),
            "low": close - abs(np.random.randn(n) * 0.5),
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, n),
        }
    )


def test_macd_strategy(sample_ohlcv):
    strategy = MACDCrossoverStrategy()
    signals = strategy.generate_signals(sample_ohlcv)
    assert len(signals) > 0
    assert all(s["signal"] in ("BUY", "SELL", "HOLD") for s in signals)


def test_rsi_strategy(sample_ohlcv):
    strategy = RSIMeanReversionStrategy()
    signals = strategy.generate_signals(sample_ohlcv)
    assert len(signals) > 0


def test_dual_ma_strategy(sample_ohlcv):
    strategy = DualMovingAverageCrossover()
    signals = strategy.generate_signals(sample_ohlcv)
    assert len(signals) > 0


def test_strategy_registry():
    assert "macd_crossover" in STRATEGY_REGISTRY
    assert "rsi_mean_reversion" in STRATEGY_REGISTRY
    assert "dual_ma_crossover" in STRATEGY_REGISTRY

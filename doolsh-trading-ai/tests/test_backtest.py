"""Tests for the backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestConfig, BacktestEngine


@pytest.fixture
def price_df():
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({"date": pd.date_range("2023-01-01", periods=n, freq="B"), "close": close})


@pytest.fixture
def signals(price_df):
    sigs = []
    for i in range(len(price_df)):
        if i % 20 == 0:
            sigs.append({"index": i, "signal": "BUY"})
        elif i % 20 == 10:
            sigs.append({"index": i, "signal": "SELL"})
        else:
            sigs.append({"index": i, "signal": "HOLD"})
    return sigs


def test_backtest_runs(price_df, signals):
    result = BacktestEngine().run(price_df, signals)
    assert result.total_trades > 0
    assert len(result.equity_curve) == len(price_df)


def test_backtest_no_signals(price_df):
    result = BacktestEngine().run(price_df, [])
    assert result.total_trades == 0


def test_backtest_metrics_range(price_df, signals):
    result = BacktestEngine().run(price_df, signals)
    assert 0 <= result.win_rate <= 100
    assert result.max_drawdown_pct >= 0

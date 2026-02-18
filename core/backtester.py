"""Backtesting engine — simulates trading on historical data.

Computes: Sharpe ratio, max drawdown, win rate, expectancy, CAGR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import load_config
from core.feature_engine import build_features
from core.model_engine import LABEL_MAP, predict

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    initial_capital: float = 10000.0
    final_capital: float = 10000.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    expectancy: float = 0.0
    cagr: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


def run_backtest(
    df: pd.DataFrame,
    model_path: str,
    model_type: str = "random_forest",
    initial_capital: float = 0.0,
    commission_pct: float = 0.0,
    slippage_pct: float = 0.0,
    atr_sl_mult: float = 0.0,
    atr_tp_mult: float = 0.0,
    position_pct: float = 0.0,
) -> BacktestResult:
    """Run a full backtest on historical OHLCV data."""
    cfg = load_config()
    capital = initial_capital or cfg.backtest_capital
    commission = commission_pct or cfg.commission_pct
    slippage = slippage_pct or cfg.slippage_pct
    sl_mult = atr_sl_mult or cfg.atr_sl_multiplier
    tp_mult = atr_tp_mult or cfg.atr_tp_multiplier
    pos_pct = position_pct or cfg.max_position_pct

    # Build features and predict
    featured = build_features(df)
    if featured.empty or len(featured) < 50:
        return BacktestResult(initial_capital=capital)

    try:
        preds, probs = predict(model_path, featured, model_type)
    except Exception as e:
        logger.error("Backtest prediction failed: %s", e)
        return BacktestResult(initial_capital=capital)

    # Align predictions with data
    min_len = min(len(preds), len(featured))
    preds = preds[-min_len:]
    probs = probs[-min_len:]
    featured = featured.iloc[-min_len:].reset_index(drop=True)

    # Simulate trades
    equity = capital
    peak_equity = capital
    position = None  # {"side", "entry", "qty", "sl", "tp"}
    trades = []
    equity_curve = [capital]

    for i in range(len(featured)):
        row = featured.iloc[i]
        close = row["close"]
        atr = row.get("atr", 0)

        # Check exit if in position
        if position is not None:
            hit_sl = False
            hit_tp = False
            exit_price = close

            if position["side"] == "BUY":
                if row["low"] <= position["sl"]:
                    hit_sl = True
                    exit_price = position["sl"]
                elif row["high"] >= position["tp"]:
                    hit_tp = True
                    exit_price = position["tp"]
            else:
                if row["high"] >= position["sl"]:
                    hit_sl = True
                    exit_price = position["sl"]
                elif row["low"] <= position["tp"]:
                    hit_tp = True
                    exit_price = position["tp"]

            if hit_sl or hit_tp:
                # Apply slippage
                exit_price *= (1 - slippage) if position["side"] == "BUY" else (1 + slippage)

                if position["side"] == "BUY":
                    pnl = (exit_price - position["entry"]) * position["qty"]
                else:
                    pnl = (position["entry"] - exit_price) * position["qty"]

                pnl -= position["entry"] * position["qty"] * commission  # entry commission
                pnl -= exit_price * position["qty"] * commission  # exit commission

                equity += pnl
                trades.append({
                    "entry": position["entry"], "exit": exit_price,
                    "side": position["side"], "qty": position["qty"],
                    "pnl": pnl, "reason": "stop_loss" if hit_sl else "take_profit",
                })
                position = None

        # Check for new signal
        if position is None:
            pred = int(preds[i])
            conf = float(probs[i])

            if pred == 2 and conf >= cfg.min_confidence:  # BUY
                side = "BUY"
            elif pred == 0 and conf >= cfg.min_confidence:  # SELL
                side = "SELL"
            else:
                equity_curve.append(equity)
                continue

            # Position sizing
            risk_amount = equity * pos_pct
            if atr > 0:
                sl_dist = atr * sl_mult
                tp_dist = atr * tp_mult
                qty = risk_amount / sl_dist if sl_dist > 0 else 0
            else:
                sl_dist = close * 0.02
                tp_dist = close * 0.04
                qty = risk_amount / sl_dist if sl_dist > 0 else 0

            if qty <= 0:
                equity_curve.append(equity)
                continue

            entry_price = close * (1 + slippage) if side == "BUY" else close * (1 - slippage)

            if side == "BUY":
                sl = entry_price - sl_dist
                tp = entry_price + tp_dist
            else:
                sl = entry_price + sl_dist
                tp = entry_price - tp_dist

            position = {"side": side, "entry": entry_price, "qty": qty, "sl": sl, "tp": tp}

        if equity > peak_equity:
            peak_equity = equity
        equity_curve.append(equity)

    # Close any remaining position at last price
    if position is not None:
        last_close = featured.iloc[-1]["close"]
        if position["side"] == "BUY":
            pnl = (last_close - position["entry"]) * position["qty"]
        else:
            pnl = (position["entry"] - last_close) * position["qty"]
        pnl -= position["entry"] * position["qty"] * commission * 2
        equity += pnl
        trades.append({
            "entry": position["entry"], "exit": last_close,
            "side": position["side"], "qty": position["qty"],
            "pnl": pnl, "reason": "end_of_data",
        })

    # Compute metrics
    result = BacktestResult(initial_capital=capital, final_capital=round(equity, 2))
    result.trades = trades
    result.equity_curve = equity_curve
    result.total_trades = len(trades)

    if trades:
        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = round(len(wins) / len(trades), 4) if trades else 0

        result.avg_win = round(np.mean(wins), 4) if wins else 0
        result.avg_loss = round(abs(np.mean(losses)), 4) if losses else 0

        result.expectancy = round(
            result.win_rate * result.avg_win - (1 - result.win_rate) * result.avg_loss, 4
        )

        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        result.profit_factor = round(total_wins / total_losses, 4) if total_losses > 0 else float("inf")

    result.total_return_pct = round((equity - capital) / capital * 100, 2)

    # Max drawdown
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1)
    result.max_drawdown_pct = round(float(dd.max()) * 100, 2)

    # Sharpe ratio (assuming hourly returns, annualized)
    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        returns = returns[np.isfinite(returns)]
        if len(returns) > 0 and returns.std() > 0:
            result.sharpe_ratio = round(float(returns.mean() / returns.std() * np.sqrt(365 * 24)), 4)

    # CAGR
    n_bars = len(equity_curve)
    if n_bars > 1 and capital > 0 and equity > 0:
        years = n_bars / (365 * 24)  # assuming hourly data
        if years > 0:
            result.cagr = round((((equity / capital) ** (1 / years)) - 1) * 100, 2)

    return result

"""Backtesting engine for Indian equity strategies.

Simulates trading on historical data with Zerodha-like costs.
Computes: Sharpe ratio, max drawdown, win rate, CAGR, profit factor.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.config import load_config
from core.feature_engine import build_features
from core.model_engine import LABEL_MAP, predict

logger = logging.getLogger(__name__)


def run_backtest(
    df: pd.DataFrame,
    model_path: str,
    model_type: str = "random_forest",
    initial_capital: float = 0,
    min_confidence: float = 0.6,
) -> Dict[str, Any]:
    """Run a backtest on historical data.

    Returns performance metrics and trade list.
    """
    cfg = load_config()
    capital = initial_capital or cfg.backtest_capital
    commission = cfg.commission_pct
    slippage = cfg.slippage_pct
    stt = cfg.stt_pct

    # Build features and predict
    featured = build_features(df, advanced=True)
    if featured.empty or len(featured) < 50:
        return {"error": "Not enough data for backtest"}

    preds, probs = predict(model_path, featured, model_type)
    if len(preds) == 0:
        return {"error": "Model prediction returned empty"}

    featured = featured.iloc[-len(preds):].reset_index(drop=True)

    # Simulate trades
    equity = capital
    peak_equity = capital
    max_drawdown = 0.0
    position = None
    trades: List[Dict] = []
    equity_curve = [capital]

    for i in range(len(preds)):
        pred = int(preds[i])
        conf = float(probs[i]) if i < len(probs) else 0.5
        price = float(featured.iloc[i]["close"])
        signal = LABEL_MAP.get(pred, "HOLD")

        # Exit logic
        if position is not None:
            exit_price = price * (1 - slippage if position["side"] == "BUY" else 1 + slippage)
            should_exit = False

            if signal != position["signal"]:
                should_exit = True
            elif position["side"] == "BUY" and price <= position["sl"]:
                should_exit = True
            elif position["side"] == "BUY" and price >= position["tp"]:
                should_exit = True
            elif position["side"] == "SELL" and price >= position["sl"]:
                should_exit = True
            elif position["side"] == "SELL" and price <= position["tp"]:
                should_exit = True

            if should_exit:
                cost = exit_price * position["qty"] * (commission + stt)
                if position["side"] == "BUY":
                    pnl = (exit_price - position["entry"]) * position["qty"] - cost
                else:
                    pnl = (position["entry"] - exit_price) * position["qty"] - cost

                equity += pnl
                trades.append({
                    "entry_price": position["entry"],
                    "exit_price": round(exit_price, 2),
                    "side": position["side"],
                    "qty": position["qty"],
                    "pnl": round(pnl, 2),
                    "confidence": position["confidence"],
                })
                position = None

        # Entry logic
        if position is None and signal != "HOLD" and conf >= min_confidence:
            entry_price = price * (1 + slippage if signal == "BUY" else 1 - slippage)
            qty = max(1, int(equity * 0.1 / entry_price))
            cost = entry_price * qty * commission

            atr = float(featured.iloc[i].get("atr", entry_price * 0.02))
            sl_pct = max(atr / entry_price * 1.5, cfg.stop_loss_pct)
            tp_pct = max(atr / entry_price * 3.0, cfg.take_profit_pct)

            if signal == "BUY":
                sl = entry_price * (1 - sl_pct)
                tp = entry_price * (1 + tp_pct)
            else:
                sl = entry_price * (1 + sl_pct)
                tp = entry_price * (1 - tp_pct)

            position = {
                "side": signal, "entry": round(entry_price, 2),
                "qty": qty, "confidence": conf,
                "sl": round(sl, 2), "tp": round(tp, 2),
                "signal": signal,
            }
            equity -= cost

        equity_curve.append(equity)
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_drawdown:
            max_drawdown = dd

    # Close any remaining position
    if position is not None and len(featured) > 0:
        last_price = float(featured.iloc[-1]["close"])
        if position["side"] == "BUY":
            pnl = (last_price - position["entry"]) * position["qty"]
        else:
            pnl = (position["entry"] - last_price) * position["qty"]
        equity += pnl
        trades.append({
            "entry_price": position["entry"],
            "exit_price": round(last_price, 2),
            "side": position["side"],
            "qty": position["qty"],
            "pnl": round(pnl, 2),
            "confidence": position["confidence"],
        })

    # Compute metrics
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_profit = sum(t["pnl"] for t in wins)
    total_loss = sum(abs(t["pnl"]) for t in losses)

    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    total_return = (equity - capital) / capital
    n_days = len(equity_curve)
    cagr = ((equity / capital) ** (252 / max(n_days, 1)) - 1) if equity > 0 else 0

    return {
        "initial_capital": capital,
        "final_equity": round(equity, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe_ratio": round(float(sharpe), 4),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "total_trades": len(trades),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": round(len(wins) / max(len(trades), 1), 4),
        "profit_factor": round(total_profit / max(total_loss, 0.01), 4),
        "avg_win": round(total_profit / max(len(wins), 1), 2),
        "avg_loss": round(total_loss / max(len(losses), 1), 2),
        "trades": trades[-50:],
        "equity_curve": equity_curve[::max(1, len(equity_curve) // 200)],
    }

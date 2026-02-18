"""Backtesting engine with portfolio simulation, slippage, and commission models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_rate: float = 0.001  # 0.1 %
    slippage_bps: float = 5.0      # basis points
    position_size_pct: float = 0.1  # 10 % of equity per trade
    max_open_positions: int = 5
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10


@dataclass
class Position:
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    entry_price: float
    entry_index: int
    entry_date: Optional[str] = None


@dataclass
class TradeRecord:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    commission: float
    slippage: float
    entry_date: Optional[str] = None
    exit_date: Optional[str] = None


@dataclass
class BacktestResult:
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_pnl: float
    total_commission: float
    total_slippage: float
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)


class BacktestEngine:
    """Event-driven backtester operating on signal arrays."""

    def __init__(self, config: BacktestConfig | None = None):
        self.cfg = config or BacktestConfig()
        self.equity = self.cfg.initial_capital
        self.positions: List[Position] = []
        self.closed_trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []

    def _apply_slippage(self, price: float, side: str) -> float:
        slip = price * (self.cfg.slippage_bps / 10_000)
        return price + slip if side == "BUY" else price - slip

    def _commission(self, notional: float) -> float:
        return notional * self.cfg.commission_rate

    def _position_qty(self, price: float) -> float:
        notional = self.equity * self.cfg.position_size_pct
        return notional / price if price > 0 else 0.0

    def run(
        self,
        prices: pd.DataFrame,
        signals: List[Dict],
    ) -> BacktestResult:
        """Execute backtest over a price DataFrame aligned with signal list.

        Parameters
        ----------
        prices : pd.DataFrame
            Must contain at least a ``close`` column.
        signals : list[dict]
            Each dict must have ``signal`` ("BUY" / "SELL" / "HOLD") and an
            ``index`` key that corresponds to the row offset in ``prices``.
        """
        self.equity = self.cfg.initial_capital
        self.positions.clear()
        self.closed_trades.clear()
        self.equity_curve.clear()

        signal_map: Dict[int, str] = {s["index"]: s["signal"] for s in signals}

        for i in range(len(prices)):
            close = prices.iloc[i]["close"]
            date_val = str(prices.iloc[i].get("date", i))

            # Check stop-loss / take-profit on open positions
            self._check_exits(close, i, date_val)

            sig = signal_map.get(i)
            if sig == "BUY" and len(self.positions) < self.cfg.max_open_positions:
                fill_price = self._apply_slippage(close, "BUY")
                qty = self._position_qty(fill_price)
                if qty > 0:
                    comm = self._commission(fill_price * qty)
                    self.equity -= comm
                    self.positions.append(
                        Position(
                            symbol=prices.get("symbol", pd.Series(["???"])).iloc[0]
                            if "symbol" in prices.columns
                            else "SYM",
                            side="BUY",
                            quantity=qty,
                            entry_price=fill_price,
                            entry_index=i,
                            entry_date=date_val,
                        )
                    )
            elif sig == "SELL":
                self._close_all_positions(close, i, date_val)

            # Mark-to-market
            unrealized = sum(
                (close - p.entry_price) * p.quantity for p in self.positions
            )
            self.equity_curve.append(self.equity + unrealized)

        # Force-close remaining at last price
        if len(prices) > 0:
            last_close = prices.iloc[-1]["close"]
            last_date = str(prices.iloc[-1].get("date", len(prices) - 1))
            self._close_all_positions(last_close, len(prices) - 1, last_date)

        return self._compile_results(len(prices))

    def _check_exits(self, price: float, idx: int, date: str) -> None:
        remaining: List[Position] = []
        for pos in self.positions:
            pnl_pct = (price - pos.entry_price) / pos.entry_price
            if pnl_pct <= -self.cfg.stop_loss_pct or pnl_pct >= self.cfg.take_profit_pct:
                self._close_position(pos, price, idx, date)
            else:
                remaining.append(pos)
        self.positions = remaining

    def _close_all_positions(self, price: float, idx: int, date: str) -> None:
        for pos in self.positions:
            self._close_position(pos, price, idx, date)
        self.positions.clear()

    def _close_position(self, pos: Position, price: float, idx: int, date: str) -> None:
        fill = self._apply_slippage(price, "SELL")
        raw_pnl = (fill - pos.entry_price) * pos.quantity
        notional = fill * pos.quantity
        comm = self._commission(notional)
        slip_cost = abs(price - fill) * pos.quantity
        net_pnl = raw_pnl - comm
        self.equity += net_pnl
        self.closed_trades.append(
            TradeRecord(
                symbol=pos.symbol,
                side=pos.side,
                quantity=pos.quantity,
                entry_price=pos.entry_price,
                exit_price=fill,
                pnl=net_pnl,
                commission=comm,
                slippage=slip_cost,
                entry_date=pos.entry_date,
                exit_date=date,
            )
        )

    def _compile_results(self, n_bars: int) -> BacktestResult:
        curve = np.array(self.equity_curve) if self.equity_curve else np.array([self.cfg.initial_capital])
        returns = np.diff(curve) / curve[:-1] if len(curve) > 1 else np.array([0.0])

        total_ret = (curve[-1] / self.cfg.initial_capital - 1) * 100
        ann_factor = 252 / max(n_bars, 1)
        ann_ret = ((1 + total_ret / 100) ** ann_factor - 1) * 100

        # Sharpe
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = (returns.mean() / downside.std()) * np.sqrt(252)
        else:
            sortino = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(curve)
        dd = (peak - curve) / peak
        max_dd = float(dd.max()) * 100

        pnls = [t.pnl for t in self.closed_trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        win_rate = len(winners) / len(pnls) * 100 if pnls else 0.0
        gross_profit = sum(winners) if winners else 0.0
        gross_loss = abs(sum(losers)) if losers else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        total_comm = sum(t.commission for t in self.closed_trades)
        total_slip = sum(t.slippage for t in self.closed_trades)

        trade_dicts = [
            {
                "symbol": t.symbol,
                "side": t.side,
                "quantity": round(t.quantity, 4),
                "entry_price": round(t.entry_price, 4),
                "exit_price": round(t.exit_price, 4),
                "pnl": round(t.pnl, 2),
                "commission": round(t.commission, 2),
                "slippage": round(t.slippage, 4),
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
            }
            for t in self.closed_trades
        ]

        return BacktestResult(
            total_return_pct=round(total_ret, 4),
            annualized_return_pct=round(ann_ret, 4),
            sharpe_ratio=round(float(sharpe), 4),
            sortino_ratio=round(float(sortino), 4),
            max_drawdown_pct=round(max_dd, 4),
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 4),
            total_trades=len(self.closed_trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            avg_pnl=round(float(np.mean(pnls)), 2) if pnls else 0.0,
            total_commission=round(total_comm, 2),
            total_slippage=round(total_slip, 4),
            equity_curve=curve.tolist(),
            trades=trade_dicts,
        )

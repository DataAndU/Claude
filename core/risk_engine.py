"""Risk management engine — position sizing, stop-loss, kill-switch.

ALL trades must pass through check_risk() before execution.
System REFUSES to trade if any risk rule is violated.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from core.config import load_config
from core.database import Trade, get_session

logger = logging.getLogger(__name__)


class RiskState:
    """In-memory risk tracking for the current trading day."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.trade_date = date.today()
        self.realized_pnl: float = 0.0
        self.trade_count: int = 0
        self.open_trades: Dict[str, Dict] = {}
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
        self.last_trade_time: Optional[datetime] = None
        self.daily_trades: List[Dict] = []

    def ensure_today(self):
        if date.today() != self.trade_date:
            logger.info("New day — resetting risk state")
            carry_equity = self.current_equity
            self.reset()
            # Preserve equity across day resets instead of losing it
            self.current_equity = carry_equity
            self.peak_equity = carry_equity


_state = RiskState()


def get_risk_state() -> RiskState:
    _state.ensure_today()
    return _state


def init_risk(capital: float) -> None:
    """Initialize risk state with starting capital."""
    state = get_risk_state()
    state.current_equity = capital
    state.peak_equity = capital
    logger.info("Risk initialized with capital=%.2f", capital)


def check_risk(
    symbol: str,
    side: str,
    price: float,
    signal: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate a proposed trade against ALL risk rules.

    Returns:
        {"allowed": bool, "reason": str, "quantity": float, "stop_loss": float, "take_profit": float}
    """
    cfg = load_config()
    state = get_risk_state()

    # Rule 0: Kill switch
    if cfg.kill_switch:
        return _deny("KILL SWITCH is ON — all trading halted")

    # Rule 0b: Zero or negative equity
    if state.current_equity <= 0:
        return _deny(f"Equity is {state.current_equity:.2f} — cannot trade with zero or negative equity")

    # Rule 1: Confidence threshold
    confidence = signal.get("confidence", 0)
    if confidence < cfg.min_confidence:
        return _deny(f"Confidence {confidence:.2f} < min {cfg.min_confidence}")

    # Rule 2: Max daily loss
    if state.current_equity > 0:
        daily_loss_pct = abs(min(state.realized_pnl, 0)) / state.current_equity
        if daily_loss_pct >= cfg.max_daily_loss_pct:
            return _deny(f"Daily loss {daily_loss_pct:.2%} >= limit {cfg.max_daily_loss_pct:.2%}")

    # Rule 3: Max drawdown
    if state.peak_equity > 0:
        drawdown = (state.peak_equity - state.current_equity) / state.peak_equity
        if drawdown >= cfg.max_drawdown_pct:
            return _deny(f"Drawdown {drawdown:.2%} >= limit {cfg.max_drawdown_pct:.2%}")

    # Rule 4: Max open trades
    if len(state.open_trades) >= cfg.max_open_trades:
        if symbol not in state.open_trades:
            return _deny(f"Max open trades {cfg.max_open_trades} reached")

    # Rule 5: Cooldown timer
    if state.last_trade_time:
        cooldown = timedelta(minutes=cfg.cooldown_minutes)
        elapsed = datetime.now(timezone.utc) - state.last_trade_time
        if elapsed < cooldown:
            remaining = (cooldown - elapsed).seconds // 60
            return _deny(f"Cooldown active — {remaining} minutes remaining")

    # Rule 6: High volatility regime caution
    regime = signal.get("regime", "unknown")
    vol = signal.get("volatility", 0)
    if regime == "high_vol" and confidence < 0.75:
        return _deny(f"High volatility regime requires confidence >= 0.75 (got {confidence:.2f})")

    # ── Compute position size ──
    atr = signal.get("atr", 0)
    quantity = compute_position_size(price, atr, state.current_equity)
    if quantity <= 0:
        return _deny("Position size computed to zero")

    # ── Compute SL/TP ──
    stop_loss, take_profit = compute_sl_tp(price, atr, side)

    return {
        "allowed": True,
        "reason": "ok",
        "quantity": quantity,
        "stop_loss": round(stop_loss, 8),
        "take_profit": round(take_profit, 8),
        "regime": regime,
    }


def compute_position_size(price: float, atr: float, equity: float) -> float:
    """Fixed fractional position sizing with ATR adjustment.

    Risk a fixed % of equity per trade. Position size = risk_amount / (ATR * multiplier).
    """
    cfg = load_config()
    risk_amount = equity * cfg.max_position_pct

    if atr > 0 and price > 0:
        risk_per_unit = atr * cfg.atr_sl_multiplier
        quantity = risk_amount / risk_per_unit
    else:
        # Fallback: 2% of equity / price
        quantity = risk_amount / max(price, 0.01)

    # Ensure minimum meaningful size
    return max(round(quantity, 8), 0.0)


def compute_sl_tp(price: float, atr: float, side: str) -> tuple:
    """ATR-based stop loss and take profit."""
    cfg = load_config()

    if atr <= 0:
        # Fallback to percentage
        sl_dist = price * 0.02
        tp_dist = price * 0.04
    else:
        sl_dist = atr * cfg.atr_sl_multiplier
        tp_dist = atr * cfg.atr_tp_multiplier

    if side == "BUY":
        stop_loss = price - sl_dist
        take_profit = price + tp_dist
    else:
        stop_loss = price + sl_dist
        take_profit = price - tp_dist

    return stop_loss, take_profit


def record_open(symbol: str, side: str, quantity: float, price: float,
                stop_loss: float, take_profit: float) -> None:
    """Record a trade opening."""
    state = get_risk_state()
    state.trade_count += 1
    state.last_trade_time = datetime.now(timezone.utc)
    state.open_trades[symbol] = {
        "side": side, "quantity": quantity, "entry_price": price,
        "stop_loss": stop_loss, "take_profit": take_profit,
        "opened_at": datetime.now(timezone.utc).isoformat(),
    }


def record_close(symbol: str, pnl: float) -> None:
    """Record a trade close."""
    state = get_risk_state()
    state.realized_pnl += pnl
    state.current_equity += pnl
    if state.current_equity > state.peak_equity:
        state.peak_equity = state.current_equity
    state.open_trades.pop(symbol, None)
    state.daily_trades.append({"symbol": symbol, "pnl": pnl})
    logger.info("Trade closed: %s pnl=%.4f day_pnl=%.4f equity=%.2f",
                symbol, pnl, state.realized_pnl, state.current_equity)


def should_close(symbol: str, current_price: float) -> Optional[str]:
    """Check if an open position should be closed (SL/TP hit)."""
    state = get_risk_state()
    pos = state.open_trades.get(symbol)
    if not pos:
        return None

    side = pos["side"]
    sl = pos["stop_loss"]
    tp = pos["take_profit"]

    if side == "BUY":
        if current_price <= sl:
            return "stop_loss"
        if current_price >= tp:
            return "take_profit"
    else:
        if current_price >= sl:
            return "stop_loss"
        if current_price <= tp:
            return "take_profit"

    return None


def get_risk_summary() -> Dict[str, Any]:
    """Get current risk analytics."""
    state = get_risk_state()
    dd = 0.0
    if state.peak_equity > 0:
        dd = (state.peak_equity - state.current_equity) / state.peak_equity

    wins = sum(1 for t in state.daily_trades if t["pnl"] > 0)
    losses = sum(1 for t in state.daily_trades if t["pnl"] < 0)
    total = len(state.daily_trades)

    return {
        "current_equity": round(state.current_equity, 2),
        "peak_equity": round(state.peak_equity, 2),
        "realized_pnl": round(state.realized_pnl, 4),
        "drawdown_pct": round(dd, 4),
        "trade_count": state.trade_count,
        "open_trades": len(state.open_trades),
        "open_positions": state.open_trades,
        "win_rate": round(wins / total, 4) if total else 0.0,
        "wins": wins,
        "losses": losses,
    }


def _deny(reason: str) -> Dict[str, Any]:
    logger.warning("RISK DENIED: %s", reason)
    return {"allowed": False, "reason": reason, "quantity": 0, "stop_loss": 0, "take_profit": 0}

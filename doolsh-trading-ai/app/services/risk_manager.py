"""Risk management engine — enforces position limits, daily loss caps, and sizing.

This module sits between the signal generator and the order manager.
Every trade request must pass through check_risk() before an order is placed.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RiskState:
    """Mutable runtime state tracking intraday risk metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.trade_date: date = date.today()
        self.realized_pnl: float = 0.0
        self.trade_count: int = 0
        self.open_positions: Dict[str, Dict[str, Any]] = {}  # symbol → info

    def _ensure_today(self) -> None:
        if date.today() != self.trade_date:
            logger.info("New trading day detected — resetting risk state")
            self.reset()


_state = RiskState()


def get_risk_state() -> RiskState:
    _state._ensure_today()
    return _state


def check_risk(
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    confidence: float,
) -> Dict[str, Any]:
    """Validate a proposed trade against all risk rules.

    Returns {"allowed": True/False, "reason": "...", "adjusted_qty": int}
    """
    state = get_risk_state()

    # Rule 1: confidence threshold
    if confidence < settings.min_confidence_threshold:
        return _deny(f"Confidence {confidence:.2f} < threshold {settings.min_confidence_threshold}")

    # Rule 2: daily loss limit
    if state.realized_pnl <= -settings.max_daily_loss:
        return _deny(f"Daily loss limit hit: {state.realized_pnl:.2f} INR")

    # Rule 3: max trades per day
    if state.trade_count >= settings.max_trade_count_per_day:
        return _deny(f"Max trade count {settings.max_trade_count_per_day} reached")

    # Rule 4: max open positions
    if side == "BUY" and len(state.open_positions) >= settings.max_open_positions:
        if symbol not in state.open_positions:
            return _deny(f"Max open positions {settings.max_open_positions} reached")

    # Rule 5: max position value
    position_value = quantity * price
    if position_value > settings.max_position_value:
        adjusted_qty = max(1, int(settings.max_position_value / price))
        logger.warning(
            "Position value %.0f > limit %.0f — reducing qty %d → %d",
            position_value, settings.max_position_value, quantity, adjusted_qty,
        )
        quantity = adjusted_qty

    return {
        "allowed": True,
        "reason": "ok",
        "adjusted_qty": quantity,
        "position_value": round(quantity * price, 2),
    }


def record_trade_open(symbol: str, side: str, quantity: int, price: float) -> None:
    state = get_risk_state()
    state.trade_count += 1
    if side == "BUY":
        state.open_positions[symbol] = {
            "side": side,
            "quantity": quantity,
            "entry_price": price,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }


def record_trade_close(symbol: str, pnl: float) -> None:
    state = get_risk_state()
    state.realized_pnl += pnl
    state.open_positions.pop(symbol, None)
    logger.info("Closed %s  pnl=%.2f  day_pnl=%.2f", symbol, pnl, state.realized_pnl)


def compute_position_size(price: float, equity: float = 100_000.0) -> int:
    """Kelly-lite position sizing capped by max_position_value."""
    max_val = min(settings.max_position_value, equity * 0.1)
    qty = max(1, int(max_val / price))
    return qty


def should_stop_loss(entry_price: float, current_price: float) -> bool:
    return current_price <= entry_price * (1 - settings.stop_loss_pct)


def should_take_profit(entry_price: float, current_price: float) -> bool:
    return current_price >= entry_price * (1 + settings.take_profit_pct)


def _deny(reason: str) -> Dict[str, Any]:
    logger.warning("RISK DENIED: %s", reason)
    return {"allowed": False, "reason": reason, "adjusted_qty": 0}

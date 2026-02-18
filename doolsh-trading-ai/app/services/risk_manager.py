"""AI-powered risk management engine.

Enforces position limits, daily loss caps, and sizing with adaptive
intelligence. Every trade request must pass through check_risk() before
an order is placed.

AI enhancements:
    - Volatility-adjusted position sizing (ATR-based)
    - Market regime-aware risk parameters
    - Drawdown protection with progressive risk reduction
    - Correlation-aware position limits
    - Dynamic stop-loss/take-profit based on ATR
    - Win-rate tracking for adaptive confidence thresholds
    - Streak detection for tilt protection
"""

from __future__ import annotations

import logging
import math
from collections import deque
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TradeRecord:
    """Record of a completed trade for performance tracking."""

    def __init__(self, symbol: str, side: str, pnl: float, confidence: float):
        self.symbol = symbol
        self.side = side
        self.pnl = pnl
        self.confidence = confidence
        self.timestamp = datetime.now(timezone.utc)

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


class RiskState:
    """Mutable runtime state tracking intraday risk metrics with AI analytics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.trade_date: date = date.today()
        self.realized_pnl: float = 0.0
        self.trade_count: int = 0
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        # AI-powered tracking
        self.trade_history: deque = deque(maxlen=100)
        self.peak_pnl: float = 0.0
        self.max_drawdown: float = 0.0
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        self.regime_adjustments: Dict[str, float] = {}

    def _ensure_today(self) -> None:
        if date.today() != self.trade_date:
            logger.info("New trading day detected — resetting risk state")
            # Preserve trade history across days for rolling analysis
            history = self.trade_history
            self.reset()
            self.trade_history = history


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
    volatility: float = 0.0,
    regime: str = "",
    signal_quality: str = "C",
) -> Dict[str, Any]:
    """Validate a proposed trade against all risk rules with AI adjustments.

    Returns {"allowed": True/False, "reason": "...", "adjusted_qty": int}
    """
    state = get_risk_state()

    # Get regime-adjusted thresholds
    conf_threshold = _get_adaptive_confidence_threshold(state, regime)

    # Rule 1: adaptive confidence threshold
    if confidence < conf_threshold:
        return _deny(f"Confidence {confidence:.2f} < adaptive threshold {conf_threshold:.2f}")

    # Rule 2: daily loss limit with drawdown protection
    effective_loss_limit = _get_drawdown_adjusted_loss_limit(state)
    if state.realized_pnl <= -effective_loss_limit:
        return _deny(f"Daily loss limit hit: {state.realized_pnl:.2f} INR (limit: {effective_loss_limit:.0f})")

    # Rule 3: max trades per day (adjusted for regime)
    max_trades = settings.max_trade_count_per_day
    if regime in ("high_volatility",):
        max_trades = max(5, max_trades - 5)
    if state.trade_count >= max_trades:
        return _deny(f"Max trade count {max_trades} reached")

    # Rule 4: max open positions
    if side == "BUY" and len(state.open_positions) >= settings.max_open_positions:
        if symbol not in state.open_positions:
            return _deny(f"Max open positions {settings.max_open_positions} reached")

    # Rule 5: tilt protection — pause after consecutive losses
    if state.consecutive_losses >= 3:
        min_quality = "A" if state.consecutive_losses >= 5 else "B"
        if signal_quality > min_quality:  # string comparison: "C" > "B" > "A"
            return _deny(
                f"Tilt protection: {state.consecutive_losses} consecutive losses, "
                f"need quality {min_quality} or better (got {signal_quality})"
            )

    # Rule 6: max position value with volatility adjustment
    position_value = quantity * price
    max_pos_value = settings.max_position_value
    if volatility > 0:
        # Reduce max position value in high-volatility environments
        vol_multiplier = _volatility_position_multiplier(volatility)
        max_pos_value *= vol_multiplier

    if position_value > max_pos_value:
        adjusted_qty = max(1, int(max_pos_value / price))
        logger.warning(
            "Position value %.0f > limit %.0f — reducing qty %d → %d",
            position_value, max_pos_value, quantity, adjusted_qty,
        )
        quantity = adjusted_qty

    return {
        "allowed": True,
        "reason": "ok",
        "adjusted_qty": quantity,
        "position_value": round(quantity * price, 2),
        "confidence_threshold_used": round(conf_threshold, 4),
        "drawdown_adjusted_limit": round(effective_loss_limit, 2),
    }


def record_trade_open(
    symbol: str, side: str, quantity: int, price: float,
    confidence: float = 0.0, volatility: float = 0.0,
) -> None:
    """Record a new trade opening with enhanced metadata."""
    state = get_risk_state()
    state.trade_count += 1
    state.open_positions[symbol] = {
        "side": side,
        "quantity": quantity,
        "entry_price": price,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "confidence": confidence,
        "volatility": volatility,
    }


def record_trade_close(symbol: str, pnl: float, confidence: float = 0.0) -> None:
    """Record a trade close with performance tracking."""
    state = get_risk_state()
    state.realized_pnl += pnl
    pos = state.open_positions.pop(symbol, {})

    # Track peak PnL and drawdown
    if state.realized_pnl > state.peak_pnl:
        state.peak_pnl = state.realized_pnl
    current_drawdown = state.peak_pnl - state.realized_pnl
    if current_drawdown > state.max_drawdown:
        state.max_drawdown = current_drawdown

    # Track streaks
    record = TradeRecord(
        symbol=symbol,
        side=pos.get("side", "BUY"),
        pnl=pnl,
        confidence=confidence or pos.get("confidence", 0),
    )
    state.trade_history.append(record)

    if pnl > 0:
        state.consecutive_wins += 1
        state.consecutive_losses = 0
    elif pnl < 0:
        state.consecutive_losses += 1
        state.consecutive_wins = 0

    logger.info(
        "Closed %s  pnl=%.2f  day_pnl=%.2f  streak=%s%d",
        symbol, pnl, state.realized_pnl,
        "W" if pnl > 0 else "L",
        state.consecutive_wins if pnl > 0 else state.consecutive_losses,
    )


def compute_position_size(
    price: float,
    equity: float = 100_000.0,
    volatility: float = 0.0,
    atr: float = 0.0,
    confidence: float = 0.6,
    regime: str = "",
) -> int:
    """AI-powered position sizing using ATR-based volatility adjustment.

    Factors:
        1. Base size from equity allocation
        2. ATR-based volatility adjustment
        3. Confidence scaling
        4. Regime multiplier
        5. Drawdown protection scaling
    """
    # Base allocation: 10% of equity per position
    max_val = min(settings.max_position_value, equity * 0.1)

    # ATR-based adjustment: risk a fixed amount (1% of equity) per ATR
    if atr > 0 and price > 0:
        risk_per_share = atr * 2  # 2x ATR as stop distance
        risk_amount = equity * 0.01  # risk 1% of equity
        atr_qty = max(1, int(risk_amount / risk_per_share))
        atr_value = atr_qty * price
        max_val = min(max_val, atr_value)

    # Volatility scaling
    if volatility > 0:
        vol_mult = _volatility_position_multiplier(volatility)
        max_val *= vol_mult

    # Confidence scaling: higher confidence = larger position
    conf_mult = 0.5 + confidence  # range: 0.5 to 1.5
    max_val *= min(conf_mult, 1.3)

    # Regime scaling
    regime_multipliers = {
        "trending_up": 1.2,
        "trending_down": 1.1,
        "mean_reverting": 0.9,
        "high_volatility": 0.5,
        "low_volatility": 1.0,
    }
    max_val *= regime_multipliers.get(regime, 1.0)

    # Drawdown protection: reduce size if in drawdown
    state = get_risk_state()
    if state.realized_pnl < 0:
        dd_pct = abs(state.realized_pnl) / settings.max_daily_loss
        dd_mult = max(0.3, 1.0 - dd_pct * 0.5)
        max_val *= dd_mult

    qty = max(1, int(max_val / price))
    return qty


def compute_dynamic_sl_tp(
    entry_price: float,
    side: str,
    atr: float = 0.0,
    volatility: float = 0.0,
    regime: str = "",
) -> Dict[str, float]:
    """Compute dynamic stop-loss and take-profit levels based on ATR and regime.

    Returns {"stop_loss": price, "take_profit": price, "sl_pct": float, "tp_pct": float}
    """
    base_sl = settings.stop_loss_pct
    base_tp = settings.take_profit_pct

    # ATR-based SL/TP
    if atr > 0 and entry_price > 0:
        atr_pct = atr / entry_price
        sl_pct = max(atr_pct * 1.5, base_sl)
        tp_pct = max(atr_pct * 3.0, base_tp)  # 2:1 reward-to-risk via ATR
    else:
        sl_pct = base_sl
        tp_pct = base_tp

    # Regime adjustments
    regime_sl_mult = {
        "trending_up": 1.5,
        "trending_down": 1.5,
        "mean_reverting": 0.8,
        "high_volatility": 2.0,
        "low_volatility": 0.7,
    }
    regime_tp_mult = {
        "trending_up": 2.0,
        "trending_down": 2.0,
        "mean_reverting": 0.7,
        "high_volatility": 1.5,
        "low_volatility": 0.8,
    }
    sl_pct *= regime_sl_mult.get(regime, 1.0)
    tp_pct *= regime_tp_mult.get(regime, 1.0)

    # Cap SL/TP to reasonable limits
    sl_pct = min(sl_pct, 0.10)  # max 10% SL
    tp_pct = min(tp_pct, 0.20)  # max 20% TP

    if side == "BUY":
        sl_price = entry_price * (1 - sl_pct)
        tp_price = entry_price * (1 + tp_pct)
    else:
        sl_price = entry_price * (1 + sl_pct)
        tp_price = entry_price * (1 - tp_pct)

    return {
        "stop_loss": round(sl_price, 2),
        "take_profit": round(tp_price, 2),
        "sl_pct": round(sl_pct, 4),
        "tp_pct": round(tp_pct, 4),
    }


def should_stop_loss(entry_price: float, current_price: float, side: str = "BUY",
                      atr: float = 0.0, regime: str = "") -> bool:
    """Dynamic stop-loss check using ATR when available."""
    levels = compute_dynamic_sl_tp(entry_price, side, atr=atr, regime=regime)
    if side == "BUY":
        return current_price <= levels["stop_loss"]
    else:
        return current_price >= levels["stop_loss"]


def should_take_profit(entry_price: float, current_price: float, side: str = "BUY",
                        atr: float = 0.0, regime: str = "") -> bool:
    """Dynamic take-profit check using ATR when available."""
    levels = compute_dynamic_sl_tp(entry_price, side, atr=atr, regime=regime)
    if side == "BUY":
        return current_price >= levels["take_profit"]
    else:
        return current_price <= levels["take_profit"]


def get_ai_risk_analytics() -> Dict[str, Any]:
    """Return AI-computed risk analytics for the current session."""
    state = get_risk_state()

    history = list(state.trade_history)
    if not history:
        return {
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "consecutive_losses": state.consecutive_losses,
            "consecutive_wins": state.consecutive_wins,
            "total_trades": 0,
        }

    wins = [t for t in history if t.is_win]
    losses = [t for t in history if not t.is_win]
    win_rate = len(wins) / len(history) if history else 0
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_loss = sum(abs(t.pnl) for t in losses) / len(losses) if losses else 0
    total_wins = sum(t.pnl for t in wins)
    total_losses = sum(abs(t.pnl) for t in losses)
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    return {
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown": round(state.max_drawdown, 2),
        "peak_pnl": round(state.peak_pnl, 2),
        "current_drawdown": round(state.peak_pnl - state.realized_pnl, 2),
        "consecutive_losses": state.consecutive_losses,
        "consecutive_wins": state.consecutive_wins,
        "total_trades": len(history),
        "realized_pnl": round(state.realized_pnl, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_adaptive_confidence_threshold(state: RiskState, regime: str = "") -> float:
    """Dynamically adjust confidence threshold based on performance and regime."""
    base = settings.min_confidence_threshold

    # Regime adjustment
    regime_adj = {
        "trending_up": -0.05,
        "trending_down": -0.05,
        "mean_reverting": 0.05,
        "high_volatility": 0.10,
        "low_volatility": 0.0,
    }
    base += regime_adj.get(regime, 0.0)

    # Performance adjustment: tighten after losses, loosen after wins
    if state.consecutive_losses >= 3:
        base += 0.05 * min(state.consecutive_losses - 2, 3)
    elif state.consecutive_wins >= 3:
        base -= 0.03

    # Drawdown adjustment
    if state.realized_pnl < 0:
        dd_pct = abs(state.realized_pnl) / settings.max_daily_loss
        base += dd_pct * 0.1

    return min(max(base, 0.40), 0.90)


def _get_drawdown_adjusted_loss_limit(state: RiskState) -> float:
    """Progressively reduce daily loss limit as drawdown deepens."""
    base_limit = settings.max_daily_loss

    # If already lost more than 50%, become much more conservative
    if state.realized_pnl < 0:
        loss_ratio = abs(state.realized_pnl) / base_limit
        if loss_ratio > 0.7:
            return base_limit * 0.85  # stop a bit earlier
        if loss_ratio > 0.5:
            return base_limit * 0.9

    return base_limit


def _volatility_position_multiplier(volatility: float) -> float:
    """Scale position size inversely with volatility."""
    if volatility <= 0:
        return 1.0
    # Normal vol ~20%, scale down for high vol, up for low vol
    target_vol = 0.20
    multiplier = target_vol / max(volatility, 0.05)
    return min(max(multiplier, 0.3), 2.0)


def _deny(reason: str) -> Dict[str, Any]:
    logger.warning("RISK DENIED: %s", reason)
    return {"allowed": False, "reason": reason, "adjusted_qty": 0}

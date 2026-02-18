"""AI-powered risk management engine for Zerodha trading.

Enforces position limits, daily loss caps, and adaptive sizing.
Every trade must pass through check_risk() before order placement.

AI features:
    - Volatility-adjusted position sizing (ATR-based)
    - Market regime-aware risk parameters
    - Drawdown protection with progressive risk reduction
    - Dynamic stop-loss/take-profit based on ATR + regime
    - Win-rate tracking and tilt protection
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import date, datetime, timezone
from typing import Any, Dict

from core.config import load_config

logger = logging.getLogger(__name__)


class TradeRecord:
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
    """Mutable runtime state tracking intraday risk metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.trade_date: date = date.today()
        self.realized_pnl: float = 0.0
        self.trade_count: int = 0
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: deque = deque(maxlen=100)
        self.peak_pnl: float = 0.0
        self.max_drawdown: float = 0.0
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0

    def _ensure_today(self) -> None:
        if date.today() != self.trade_date:
            logger.info("New trading day — resetting risk state")
            history = self.trade_history
            self.reset()
            self.trade_history = history


_state = RiskState()


def get_risk_state() -> RiskState:
    _state._ensure_today()
    return _state


def check_risk(
    symbol: str, side: str, quantity: int, price: float,
    confidence: float, volatility: float = 0.0,
    regime: str = "", signal_quality: str = "C",
) -> Dict[str, Any]:
    """Validate a trade against all risk rules. Returns allow/deny with reason."""
    cfg = load_config()
    state = get_risk_state()

    # Kill switch
    if cfg.kill_switch:
        return _deny("Kill switch is active")

    # Adaptive confidence threshold
    conf_threshold = _adaptive_confidence_threshold(state, regime)
    if confidence < conf_threshold:
        return _deny(f"Confidence {confidence:.2f} < threshold {conf_threshold:.2f}")

    # Daily loss limit
    effective_limit = _drawdown_adjusted_limit(state)
    if state.realized_pnl <= -effective_limit:
        return _deny(f"Daily loss limit: {state.realized_pnl:.0f} INR (limit: {effective_limit:.0f})")

    # Max trades per day
    max_trades = cfg.max_trades_per_day
    if regime == "high_volatility":
        max_trades = max(5, max_trades - 5)
    if state.trade_count >= max_trades:
        return _deny(f"Max trades {max_trades} reached")

    # Max open positions
    if side == "BUY" and len(state.open_positions) >= cfg.max_open_positions:
        if symbol not in state.open_positions:
            return _deny(f"Max positions {cfg.max_open_positions} reached")

    # Tilt protection
    if cfg.tilt_protection and state.consecutive_losses >= 3:
        min_quality = "A" if state.consecutive_losses >= 5 else "B"
        if signal_quality > min_quality:
            return _deny(
                f"Tilt protection: {state.consecutive_losses} losses, "
                f"need quality {min_quality}+ (got {signal_quality})"
            )

    # Position value with volatility adjustment
    position_value = quantity * price
    max_pos_value = cfg.max_position_value
    if volatility > 0:
        vol_mult = _volatility_multiplier(volatility)
        max_pos_value *= vol_mult

    if position_value > max_pos_value:
        quantity = max(1, int(max_pos_value / price))

    return {
        "allowed": True, "reason": "ok",
        "adjusted_qty": quantity,
        "position_value": round(quantity * price, 2),
        "confidence_threshold_used": round(conf_threshold, 4),
    }


def record_trade_open(
    symbol: str, side: str, quantity: int, price: float,
    confidence: float = 0.0, volatility: float = 0.0,
) -> None:
    state = get_risk_state()
    state.trade_count += 1
    state.open_positions[symbol] = {
        "side": side, "quantity": quantity, "entry_price": price,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "confidence": confidence, "volatility": volatility,
    }


def record_trade_close(symbol: str, pnl: float, confidence: float = 0.0) -> None:
    state = get_risk_state()
    state.realized_pnl += pnl
    pos = state.open_positions.pop(symbol, {})

    if state.realized_pnl > state.peak_pnl:
        state.peak_pnl = state.realized_pnl
    dd = state.peak_pnl - state.realized_pnl
    if dd > state.max_drawdown:
        state.max_drawdown = dd

    record = TradeRecord(symbol, pos.get("side", "BUY"), pnl, confidence)
    state.trade_history.append(record)

    if pnl > 0:
        state.consecutive_wins += 1
        state.consecutive_losses = 0
    elif pnl < 0:
        state.consecutive_losses += 1
        state.consecutive_wins = 0


def compute_position_size(
    price: float, equity: float = 100000.0,
    volatility: float = 0.0, atr: float = 0.0,
    confidence: float = 0.6, regime: str = "",
) -> int:
    """AI-powered position sizing."""
    cfg = load_config()
    max_val = min(cfg.max_position_value, equity * 0.1)

    # ATR-based sizing
    if atr > 0 and price > 0:
        risk_per_share = atr * 2
        risk_amount = equity * 0.01
        atr_qty = max(1, int(risk_amount / risk_per_share))
        max_val = min(max_val, atr_qty * price)

    # Volatility scaling
    if volatility > 0:
        max_val *= _volatility_multiplier(volatility)

    # Confidence scaling
    max_val *= min(0.5 + confidence, 1.3)

    # Regime scaling
    regime_mult = {
        "trending_up": 1.2, "trending_down": 1.1,
        "mean_reverting": 0.9, "high_volatility": 0.5,
    }
    max_val *= regime_mult.get(regime, 1.0)

    # Drawdown protection
    state = get_risk_state()
    if state.realized_pnl < 0:
        dd_pct = abs(state.realized_pnl) / cfg.max_daily_loss
        max_val *= max(0.3, 1.0 - dd_pct * 0.5)

    return max(1, int(max_val / price))


def compute_dynamic_sl_tp(
    entry_price: float, side: str,
    atr: float = 0.0, volatility: float = 0.0, regime: str = "",
) -> Dict[str, float]:
    """Compute dynamic SL/TP based on ATR and market regime."""
    cfg = load_config()
    base_sl = cfg.stop_loss_pct
    base_tp = cfg.take_profit_pct

    if atr > 0 and entry_price > 0:
        atr_pct = atr / entry_price
        sl_pct = max(atr_pct * 1.5, base_sl)
        tp_pct = max(atr_pct * 3.0, base_tp)
    else:
        sl_pct = base_sl
        tp_pct = base_tp

    sl_mult = {"trending_up": 1.5, "trending_down": 1.5, "mean_reverting": 0.8, "high_volatility": 2.0}
    tp_mult = {"trending_up": 2.0, "trending_down": 2.0, "mean_reverting": 0.7, "high_volatility": 1.5}
    sl_pct *= sl_mult.get(regime, 1.0)
    tp_pct *= tp_mult.get(regime, 1.0)

    sl_pct = min(sl_pct, 0.10)
    tp_pct = min(tp_pct, 0.20)

    if side == "BUY":
        return {"stop_loss": round(entry_price * (1 - sl_pct), 2), "take_profit": round(entry_price * (1 + tp_pct), 2)}
    else:
        return {"stop_loss": round(entry_price * (1 + sl_pct), 2), "take_profit": round(entry_price * (1 - tp_pct), 2)}


def should_stop_loss(entry: float, current: float, side: str = "BUY", regime: str = "") -> bool:
    levels = compute_dynamic_sl_tp(entry, side, regime=regime)
    return current <= levels["stop_loss"] if side == "BUY" else current >= levels["stop_loss"]


def should_take_profit(entry: float, current: float, side: str = "BUY", regime: str = "") -> bool:
    levels = compute_dynamic_sl_tp(entry, side, regime=regime)
    return current >= levels["take_profit"] if side == "BUY" else current <= levels["take_profit"]


def get_risk_analytics() -> Dict[str, Any]:
    state = get_risk_state()
    history = list(state.trade_history)
    if not history:
        return {
            "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0, "max_drawdown": 0.0,
            "consecutive_losses": 0, "consecutive_wins": 0,
            "total_trades": 0, "realized_pnl": 0.0,
        }

    wins = [t for t in history if t.is_win]
    losses = [t for t in history if not t.is_win]
    total_wins = sum(t.pnl for t in wins)
    total_losses = sum(abs(t.pnl) for t in losses)

    return {
        "win_rate": round(len(wins) / len(history), 4),
        "avg_win": round(total_wins / max(len(wins), 1), 2),
        "avg_loss": round(total_losses / max(len(losses), 1), 2),
        "profit_factor": round(total_wins / max(total_losses, 0.01), 4),
        "max_drawdown": round(state.max_drawdown, 2),
        "peak_pnl": round(state.peak_pnl, 2),
        "consecutive_losses": state.consecutive_losses,
        "consecutive_wins": state.consecutive_wins,
        "total_trades": len(history),
        "realized_pnl": round(state.realized_pnl, 2),
        "open_positions": len(state.open_positions),
    }


def _adaptive_confidence_threshold(state: RiskState, regime: str = "") -> float:
    cfg = load_config()
    base = cfg.min_confidence
    regime_adj = {"trending_up": -0.05, "trending_down": -0.05, "mean_reverting": 0.05, "high_volatility": 0.10}
    base += regime_adj.get(regime, 0.0)
    if state.consecutive_losses >= 3:
        base += 0.05 * min(state.consecutive_losses - 2, 3)
    elif state.consecutive_wins >= 3:
        base -= 0.03
    if state.realized_pnl < 0:
        base += abs(state.realized_pnl) / cfg.max_daily_loss * 0.1
    return min(max(base, 0.40), 0.90)


def _drawdown_adjusted_limit(state: RiskState) -> float:
    cfg = load_config()
    base = cfg.max_daily_loss
    if state.realized_pnl < 0:
        ratio = abs(state.realized_pnl) / base
        if ratio > 0.7: return base * 0.85
        if ratio > 0.5: return base * 0.9
    return base


def _volatility_multiplier(volatility: float) -> float:
    if volatility <= 0: return 1.0
    return min(max(0.20 / max(volatility, 0.05), 0.3), 2.0)


def _deny(reason: str) -> Dict[str, Any]:
    logger.warning("RISK DENIED: %s", reason)
    return {"allowed": False, "reason": reason, "adjusted_qty": 0}

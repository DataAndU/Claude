"""Execution engine — paper and live order execution via ccxt.

Paper mode: simulates fills at market price, logs to database.
Live mode:  sends real orders to Binance via ccxt (requires API keys).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.config import load_config
from core.database import Trade, get_session
from core.risk_engine import (
    check_risk,
    get_risk_state,
    record_close,
    record_open,
    should_close,
)

logger = logging.getLogger(__name__)


# ─── Paper Trading ────────────────────────────────────────────────────────────


def paper_open(symbol: str, side: str, price: float, signal: Dict) -> Optional[Dict]:
    """Open a paper trade after risk check."""
    risk = check_risk(symbol, side, price, signal)
    if not risk["allowed"]:
        logger.info("Paper open DENIED for %s: %s", symbol, risk["reason"])
        return None

    qty = risk["quantity"]
    sl = risk["stop_loss"]
    tp = risk["take_profit"]

    record_open(symbol, side, qty, price, sl, tp)

    # Log to database
    session = get_session()
    try:
        trade = Trade(
            symbol=symbol, side=side, price=price, quantity=qty,
            cost=price * qty, mode="paper",
            stop_loss=sl, take_profit=tp,
            model_type=signal.get("model_type", ""),
            confidence=signal.get("confidence", 0),
            status="open",
        )
        session.add(trade)
        session.commit()
        trade_id = trade.id
    except Exception as e:
        session.rollback()
        logger.error("DB error on paper_open: %s", e)
        trade_id = None
    finally:
        session.close()

    logger.info(
        "PAPER %s %s %.8f @ %.8f | SL=%.8f TP=%.8f conf=%.2f",
        side, symbol, qty, price, sl, tp, signal.get("confidence", 0),
    )
    return {
        "trade_id": trade_id, "symbol": symbol, "side": side,
        "price": price, "quantity": qty,
        "stop_loss": sl, "take_profit": tp,
        "mode": "paper",
    }


def paper_close(symbol: str, current_price: float, reason: str = "manual") -> Optional[Dict]:
    """Close a paper trade and record PnL."""
    state = get_risk_state()
    pos = state.open_trades.get(symbol)
    if not pos:
        return None

    side = pos["side"]
    qty = pos["quantity"]
    entry = pos["entry_price"]

    if side == "BUY":
        pnl = (current_price - entry) * qty
    else:
        pnl = (entry - current_price) * qty

    record_close(symbol, pnl)

    # Update database
    session = get_session()
    try:
        trade = session.query(Trade).filter(
            Trade.symbol == symbol, Trade.status == "open", Trade.mode == "paper"
        ).order_by(Trade.id.desc()).first()
        if trade:
            trade.pnl = pnl
            trade.status = "closed"
            trade.close_reason = reason
            trade.closed_at = datetime.now(timezone.utc)
            session.commit()
    except Exception as e:
        session.rollback()
        logger.error("DB error on paper_close: %s", e)
    finally:
        session.close()

    logger.info(
        "PAPER CLOSE %s @ %.8f | entry=%.8f pnl=%.4f reason=%s",
        symbol, current_price, entry, pnl, reason,
    )
    return {
        "symbol": symbol, "side": side, "entry": entry,
        "exit": current_price, "pnl": round(pnl, 8), "reason": reason,
    }


def paper_check_exits(prices: Dict[str, float]) -> list:
    """Check all open paper positions for SL/TP hits."""
    closed = []
    state = get_risk_state()
    for symbol in list(state.open_trades.keys()):
        price = prices.get(symbol, 0)
        if price <= 0:
            continue
        reason = should_close(symbol, price)
        if reason:
            result = paper_close(symbol, price, reason)
            if result:
                closed.append(result)
    return closed


# ─── Live Trading ─────────────────────────────────────────────────────────────


def live_open(symbol: str, side: str, signal: Dict) -> Optional[Dict]:
    """Open a live trade on the exchange."""
    cfg = load_config()
    if cfg.mode != "live":
        logger.error("live_open called but mode is '%s'", cfg.mode)
        return None

    try:
        import ccxt
    except ImportError:
        logger.error("ccxt not installed")
        return None

    api_key = os.getenv("EXCHANGE_API_KEY", "")
    api_secret = os.getenv("EXCHANGE_API_SECRET", "")
    if not api_key or not api_secret:
        logger.error("Exchange API keys not set in environment")
        return None

    # Get current price
    try:
        exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })
        ticker = exchange.fetch_ticker(symbol)
        price = ticker["last"]
    except Exception as e:
        logger.error("Failed to fetch price: %s", e)
        return None

    risk = check_risk(symbol, side, price, signal)
    if not risk["allowed"]:
        logger.info("Live open DENIED for %s: %s", symbol, risk["reason"])
        return None

    qty = risk["quantity"]
    sl = risk["stop_loss"]
    tp = risk["take_profit"]

    try:
        order = exchange.create_market_order(symbol, side.lower(), qty)
        filled_price = order.get("average", price)
        record_open(symbol, side, qty, filled_price, sl, tp)

        # Log to database
        session = get_session()
        try:
            trade = Trade(
                symbol=symbol, side=side, price=filled_price, quantity=qty,
                cost=filled_price * qty, mode="live",
                stop_loss=sl, take_profit=tp,
                model_type=signal.get("model_type", ""),
                confidence=signal.get("confidence", 0),
                status="open",
            )
            session.add(trade)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("DB error: %s", e)
        finally:
            session.close()

        logger.info("LIVE %s %s %.8f @ %.8f", side, symbol, qty, filled_price)
        return {
            "trade_id": order.get("id"), "symbol": symbol, "side": side,
            "price": filled_price, "quantity": qty,
            "stop_loss": sl, "take_profit": tp, "mode": "live",
        }
    except Exception as e:
        logger.error("Live order failed: %s", e)
        return None


# ─── Unified Interface ────────────────────────────────────────────────────────


def execute_signal(signal: Dict, prices: Optional[Dict[str, float]] = None) -> Optional[Dict]:
    """Execute a trading signal in the current mode (paper/live)."""
    cfg = load_config()
    symbol = signal.get("symbol", "")
    direction = signal.get("signal", "HOLD")

    if direction == "HOLD":
        return None

    state = get_risk_state()

    # If we already have a position in this symbol, check if we should reverse
    if symbol in state.open_trades:
        current_side = state.open_trades[symbol]["side"]
        if (direction == "BUY" and current_side == "SELL") or \
           (direction == "SELL" and current_side == "BUY"):
            # Close existing position first
            price = signal.get("price", 0)
            if prices and symbol in prices:
                price = prices[symbol]
            if cfg.mode == "paper":
                paper_close(symbol, price, "signal_reversal")
            # Then open new position below
        else:
            # Already positioned in same direction
            return None

    price = signal.get("price", 0)
    if prices and symbol in prices:
        price = prices[symbol]

    if price <= 0:
        return None

    if cfg.mode == "paper":
        return paper_open(symbol, direction, price, signal)
    elif cfg.mode == "live":
        return live_open(symbol, direction, signal)
    else:
        logger.warning("Unknown trading mode: %s", cfg.mode)
        return None

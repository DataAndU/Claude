"""Automated trading loop — the brain of the system.

Runs on a configurable schedule (APScheduler or manual invocation).

Cycle:
    1. Check if market is open
    2. For each watchlist symbol:
        a. Fetch latest data
        b. Run ML prediction / strategy
        c. Pass through risk manager
        d. Place order via order manager
    3. Monitor open positions for SL / TP
    4. Auto square-off before close
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone

from app.core.config import get_settings
from app.core.kite import is_logged_in
from app.services.market_data import fetch_intraday_prices, get_ltp
from app.services.order_manager import get_positions, place_order
from app.services.risk_manager import (
    check_risk,
    compute_position_size,
    get_risk_state,
    record_trade_close,
    record_trade_open,
    should_stop_loss,
    should_take_profit,
)
from app.services.signal_generator import generate_signals
from ml.features.engineering import build_features

logger = logging.getLogger(__name__)
settings = get_settings()

# IST offset
IST = timezone(timedelta(hours=5, minutes=30))

_auto_trading_enabled = False


def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:  # Saturday / Sunday
        return False
    market_open = time(settings.market_open_hour, settings.market_open_minute)
    market_close = time(settings.market_close_hour, settings.market_close_minute)
    return market_open <= now.time() <= market_close


def is_square_off_time() -> bool:
    now = datetime.now(IST)
    close = datetime.combine(now.date(), time(settings.market_close_hour, settings.market_close_minute))
    square_off = close - timedelta(minutes=settings.auto_square_off_minute)
    return now >= square_off.replace(tzinfo=IST)


async def trading_cycle(model_path: str, model_type: str = "rf") -> dict:
    """Execute one full trading cycle across the watchlist.

    Returns a summary dict of actions taken.
    """
    if not is_logged_in():
        return {"error": "Kite not logged in"}

    if not is_market_open():
        return {"status": "market_closed"}

    # Square-off check
    if is_square_off_time():
        return await _square_off_all()

    summary = {"signals": [], "orders": [], "skipped": [], "errors": []}

    ltps = await get_ltp(settings.watchlist_symbols)

    for symbol in settings.watchlist_symbols:
        try:
            df = await fetch_intraday_prices(symbol, interval="15minute", days=5)
            if len(df) < 50:
                summary["skipped"].append({"symbol": symbol, "reason": "insufficient data"})
                continue

            signals = generate_signals(df, model_path=model_path, model_type=model_type)
            if not signals:
                continue

            latest = signals[-1]
            summary["signals"].append({"symbol": symbol, **latest})

            if latest["signal"] == "HOLD":
                continue

            price = ltps.get(symbol, latest.get("price", 0))
            if price <= 0:
                continue

            side = latest["signal"]  # BUY or SELL
            qty = compute_position_size(price)

            risk = check_risk(symbol, side, qty, price, latest["confidence"])
            if not risk["allowed"]:
                summary["skipped"].append({"symbol": symbol, "reason": risk["reason"]})
                continue

            qty = risk["adjusted_qty"]
            result = await place_order(
                symbol=symbol,
                side=side,
                quantity=qty,
                order_type="MARKET",
                price=price,
            )
            record_trade_open(symbol, side, qty, price)
            summary["orders"].append(result)

        except Exception as exc:
            logger.exception("Error processing %s: %s", symbol, exc)
            summary["errors"].append({"symbol": symbol, "error": str(exc)})

    # Monitor existing positions for SL/TP
    await _monitor_positions(ltps, summary)

    return summary


async def _monitor_positions(ltps: dict, summary: dict) -> None:
    """Check open positions for stop-loss / take-profit triggers."""
    state = get_risk_state()
    to_close = []

    for symbol, pos in state.open_positions.items():
        current = ltps.get(symbol, 0)
        if current <= 0:
            continue

        entry = pos["entry_price"]
        if should_stop_loss(entry, current):
            to_close.append((symbol, current, entry, "stop_loss"))
        elif should_take_profit(entry, current):
            to_close.append((symbol, current, entry, "take_profit"))

    for symbol, current, entry, reason in to_close:
        try:
            qty = state.open_positions[symbol]["quantity"]
            result = await place_order(symbol=symbol, side="SELL", quantity=qty, order_type="MARKET", price=current)
            pnl = (current - entry) * qty
            record_trade_close(symbol, pnl)
            summary["orders"].append({**result, "close_reason": reason, "pnl": round(pnl, 2)})
        except Exception as exc:
            logger.exception("Failed to close %s: %s", symbol, exc)


async def _square_off_all() -> dict:
    """Close all open positions before market close."""
    state = get_risk_state()
    results = []
    symbols = list(state.open_positions.keys())

    if not symbols:
        return {"status": "square_off", "message": "no open positions"}

    ltps = await get_ltp(symbols)

    for symbol in symbols:
        pos = state.open_positions.get(symbol)
        if not pos:
            continue
        current = ltps.get(symbol, pos["entry_price"])
        qty = pos["quantity"]
        try:
            result = await place_order(symbol=symbol, side="SELL", quantity=qty, order_type="MARKET", price=current)
            pnl = (current - pos["entry_price"]) * qty
            record_trade_close(symbol, pnl)
            results.append({**result, "pnl": round(pnl, 2)})
        except Exception as exc:
            logger.exception("Square-off failed for %s: %s", symbol, exc)
            results.append({"symbol": symbol, "error": str(exc)})

    return {"status": "square_off", "closed": results}


def enable_auto_trading() -> None:
    global _auto_trading_enabled
    _auto_trading_enabled = True
    logger.info("Auto-trading ENABLED")


def disable_auto_trading() -> None:
    global _auto_trading_enabled
    _auto_trading_enabled = False
    logger.info("Auto-trading DISABLED")


def is_auto_trading_enabled() -> bool:
    return _auto_trading_enabled

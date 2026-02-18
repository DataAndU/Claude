"""Automated trading engine — scans F&O stocks, picks best instruments, trades.

Runs on a configurable schedule. Supports:
    - Intraday short-selling (MIS) based on F&O scan scores
    - BTST buying (NRML) based on bullish momentum scores
    - Options trading (NFO) with strike selection via Black-Scholes
    - Position monitoring with SL/TP auto-exit
    - Auto square-off before market close

Works in both PAPER mode (synthetic data) and LIVE mode (Kite API).
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.kite import is_logged_in
from app.services.risk_manager import (
    check_risk,
    compute_position_size,
    get_risk_state,
    record_trade_close,
    record_trade_open,
    should_stop_loss,
    should_take_profit,
)

logger = logging.getLogger(__name__)
settings = get_settings()

IST = timezone(timedelta(hours=5, minutes=30))

# ─────────────────────────────────────────────────────────────────────────────
# Runtime state
# ─────────────────────────────────────────────────────────────────────────────

_auto_trading_enabled = False
_trade_log: List[Dict[str, Any]] = []

# What modes to auto-trade
_auto_modes = {
    "intraday_sell": True,
    "btst_buy": True,
    "options_intraday": True,
    "options_btst": True,
}

# Minimum score to auto-trade
_min_score_intraday = 50.0
_min_score_btst = 50.0
_min_score_options = 50.0

# Max trades per cycle
_max_trades_per_cycle = 3

# Cycle count
_cycle_count = 0


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


def set_auto_modes(**modes) -> None:
    for k, v in modes.items():
        if k in _auto_modes:
            _auto_modes[k] = bool(v)


def get_auto_modes() -> Dict[str, bool]:
    return _auto_modes.copy()


def set_min_scores(intraday: float = None, btst: float = None, options: float = None):
    global _min_score_intraday, _min_score_btst, _min_score_options
    if intraday is not None:
        _min_score_intraday = intraday
    if btst is not None:
        _min_score_btst = btst
    if options is not None:
        _min_score_options = options


def get_min_scores() -> Dict[str, float]:
    return {
        "intraday": _min_score_intraday,
        "btst": _min_score_btst,
        "options": _min_score_options,
    }


def get_trade_log() -> List[Dict[str, Any]]:
    return _trade_log.copy()


def clear_trade_log() -> None:
    _trade_log.clear()


def get_cycle_count() -> int:
    return _cycle_count


# ─────────────────────────────────────────────────────────────────────────────
# Market hours
# ─────────────────────────────────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    market_open = time(settings.market_open_hour, settings.market_open_minute)
    market_close = time(settings.market_close_hour, settings.market_close_minute)
    return market_open <= now.time() <= market_close


def is_square_off_time() -> bool:
    now = datetime.now(IST)
    close_dt = datetime.combine(
        now.date(),
        time(settings.market_close_hour, settings.market_close_minute),
    )
    square_off = close_dt - timedelta(minutes=settings.auto_square_off_minute)
    return now >= square_off.replace(tzinfo=IST)


def _now_ist() -> str:
    return datetime.now(IST).strftime("%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# Main trading cycle
# ─────────────────────────────────────────────────────────────────────────────

async def trading_cycle(model_path: str = "", model_type: str = "rf") -> dict:
    """Execute one full auto-trading cycle.

    1. Scan F&O stocks for best signals
    2. Filter by minimum score
    3. Check risk limits
    4. Place orders for top candidates
    5. Monitor existing positions for SL/TP
    6. Auto square-off near close
    """
    global _cycle_count
    _cycle_count += 1

    summary = {
        "cycle": _cycle_count,
        "timestamp": _now_ist(),
        "signals_found": 0,
        "orders_placed": [],
        "positions_closed": [],
        "skipped": [],
        "errors": [],
    }

    # Square-off check first
    if is_square_off_time():
        close_result = await _square_off_all()
        summary["positions_closed"] = close_result.get("closed", [])
        _log("SQUARE_OFF", {"cycle": _cycle_count})
        return summary

    use_live = is_logged_in()
    trades_this_cycle = 0

    # ── 1. Intraday short-sell scan ──
    if _auto_modes.get("intraday_sell") and trades_this_cycle < _max_trades_per_cycle:
        try:
            from app.services.fno_scanner import scan_fno_symbols
            sells = await scan_fno_symbols(
                top_n=10, use_live_data=use_live, scan_type="sell"
            )
            for sig in sells:
                if trades_this_cycle >= _max_trades_per_cycle:
                    break
                if sig["score"] < _min_score_intraday:
                    continue
                if sig["action"] not in ("STRONG SELL", "SELL"):
                    continue

                result = await _execute_equity_trade(
                    symbol=sig["symbol"], side="SELL", price=sig["price"],
                    trade_type="intraday", signal=sig, summary=summary,
                )
                if result:
                    trades_this_cycle += 1
                    summary["signals_found"] += 1

        except Exception as e:
            summary["errors"].append({"scan": "intraday_sell", "error": str(e)})
            logger.exception("Intraday sell scan failed")

    # ── 2. BTST buy scan ──
    if _auto_modes.get("btst_buy") and trades_this_cycle < _max_trades_per_cycle:
        try:
            from app.services.fno_scanner import scan_fno_symbols
            buys = await scan_fno_symbols(
                top_n=10, use_live_data=use_live, scan_type="buy"
            )
            for sig in buys:
                if trades_this_cycle >= _max_trades_per_cycle:
                    break
                if sig["score"] < _min_score_btst:
                    continue
                if sig["action"] not in ("STRONG BUY", "BUY"):
                    continue

                result = await _execute_equity_trade(
                    symbol=sig["symbol"], side="BUY", price=sig["price"],
                    trade_type="btst", signal=sig, summary=summary,
                )
                if result:
                    trades_this_cycle += 1
                    summary["signals_found"] += 1

        except Exception as e:
            summary["errors"].append({"scan": "btst_buy", "error": str(e)})
            logger.exception("BTST buy scan failed")

    # ── 3. Options intraday ──
    if _auto_modes.get("options_intraday") and trades_this_cycle < _max_trades_per_cycle:
        try:
            from app.services.options_chain import scan_options_opportunities
            opts = await scan_options_opportunities(top_n=5, trade_type="intraday")
            for opt in opts:
                if trades_this_cycle >= _max_trades_per_cycle:
                    break
                if opt.get("equity_score", 0) < _min_score_options:
                    continue
                result = await _execute_options_trade(opt, "intraday", summary)
                if result:
                    trades_this_cycle += 1
                    summary["signals_found"] += 1

        except Exception as e:
            summary["errors"].append({"scan": "options_intraday", "error": str(e)})
            logger.exception("Options intraday scan failed")

    # ── 4. Options BTST ──
    if _auto_modes.get("options_btst") and trades_this_cycle < _max_trades_per_cycle:
        try:
            from app.services.options_chain import scan_options_opportunities
            opts = await scan_options_opportunities(top_n=5, trade_type="btst")
            for opt in opts:
                if trades_this_cycle >= _max_trades_per_cycle:
                    break
                if opt.get("equity_score", 0) < _min_score_options:
                    continue
                result = await _execute_options_trade(opt, "btst", summary)
                if result:
                    trades_this_cycle += 1
                    summary["signals_found"] += 1

        except Exception as e:
            summary["errors"].append({"scan": "options_btst", "error": str(e)})
            logger.exception("Options BTST scan failed")

    # ── 5. Monitor positions ──
    await _monitor_positions(summary)

    _log("CYCLE", {
        "cycle": _cycle_count,
        "signals": summary["signals_found"],
        "orders": len(summary["orders_placed"]),
        "closed": len(summary["positions_closed"]),
        "errors": len(summary["errors"]),
    })
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Trade execution
# ─────────────────────────────────────────────────────────────────────────────

async def _execute_equity_trade(
    symbol: str, side: str, price: float, trade_type: str,
    signal: dict, summary: dict,
) -> Optional[dict]:
    from app.services.order_manager import place_order

    qty = compute_position_size(price)
    confidence = signal.get("score", 0) / 100.0

    risk = check_risk(symbol, side, qty, price, confidence)
    if not risk["allowed"]:
        summary["skipped"].append({
            "symbol": symbol, "side": side, "reason": risk["reason"],
        })
        return None

    qty = risk["adjusted_qty"]
    product = "MIS" if trade_type == "intraday" else settings.btst_product

    try:
        result = await place_order(
            symbol=symbol, side=side, quantity=qty,
            order_type="MARKET", price=price,
            product=product, exchange="NSE", trade_type=trade_type,
        )
        record_trade_open(symbol, side, qty, price)
        result["score"] = signal.get("score", 0)
        result["action"] = signal.get("action", "")
        result["reasons"] = signal.get("reasons", [])
        summary["orders_placed"].append(result)
        _log("ORDER", {
            "symbol": symbol, "side": side, "qty": qty,
            "price": price, "type": trade_type,
            "score": signal.get("score", 0),
        })
        return result
    except Exception as e:
        summary["errors"].append({"symbol": symbol, "error": str(e)})
        logger.exception("Failed to place equity order for %s", symbol)
        return None


async def _execute_options_trade(
    option: dict, trade_type: str, summary: dict,
) -> Optional[dict]:
    from app.services.order_manager import place_order

    symbol = option.get("nfo_symbol", "")
    underlying = option.get("symbol", "")
    entry_price = option.get("entry_price", 0)
    qty = option.get("quantity", option.get("lot_size", 1))
    eq_score = option.get("equity_score", 0)
    confidence = eq_score / 100.0

    risk = check_risk(underlying, "BUY", qty, entry_price, confidence)
    if not risk["allowed"]:
        summary["skipped"].append({
            "symbol": symbol, "side": "BUY", "reason": risk["reason"],
        })
        return None

    product = option.get("product", "MIS")

    try:
        result = await place_order(
            symbol=symbol, side="BUY", quantity=qty,
            order_type="MARKET", price=entry_price,
            product=product, exchange="NFO", trade_type=trade_type,
        )
        record_trade_open(symbol, "BUY", qty, entry_price)
        result["option_type"] = option.get("option_type", "")
        result["strike"] = option.get("strike", 0)
        result["underlying"] = underlying
        result["direction"] = option.get("direction", "")
        result["sl_price"] = option.get("sl_price", 0)
        result["target_price"] = option.get("target_price", 0)
        result["equity_score"] = eq_score
        summary["orders_placed"].append(result)
        _log("OPTIONS", {
            "symbol": symbol, "underlying": underlying,
            "type": option.get("option_type"),
            "strike": option.get("strike"),
            "entry": entry_price, "qty": qty,
        })
        return result
    except Exception as e:
        summary["errors"].append({"symbol": symbol, "error": str(e)})
        logger.exception("Failed to place options order for %s", symbol)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Position monitoring
# ─────────────────────────────────────────────────────────────────────────────

async def _monitor_positions(summary: dict) -> None:
    state = get_risk_state()
    if not state.open_positions:
        return

    symbols = list(state.open_positions.keys())
    ltps = {}

    if is_logged_in():
        try:
            from app.services.market_data import get_ltp
            ltps = await get_ltp(symbols)
        except Exception:
            pass

    from app.services.order_manager import place_order
    to_close = []

    for symbol, pos in state.open_positions.items():
        current = ltps.get(symbol, 0)
        if current <= 0:
            entry = pos["entry_price"]
            current = entry * (1 + random.uniform(-0.03, 0.03))

        entry = pos["entry_price"]
        side = pos.get("side", "BUY")

        if side == "BUY":
            if should_stop_loss(entry, current):
                to_close.append((symbol, current, entry, "stop_loss", pos))
            elif should_take_profit(entry, current):
                to_close.append((symbol, current, entry, "take_profit", pos))
        else:
            if current >= entry * (1 + settings.stop_loss_pct):
                to_close.append((symbol, current, entry, "stop_loss", pos))
            elif current <= entry * (1 - settings.take_profit_pct):
                to_close.append((symbol, current, entry, "take_profit", pos))

    for symbol, current, entry, reason, pos in to_close:
        try:
            qty = pos["quantity"]
            close_side = "SELL" if pos.get("side") == "BUY" else "BUY"
            result = await place_order(
                symbol=symbol, side=close_side, quantity=qty,
                order_type="MARKET", price=current,
            )
            pnl = (current - entry) * qty if close_side == "SELL" else (entry - current) * qty
            record_trade_close(symbol, pnl)
            result["close_reason"] = reason
            result["pnl"] = round(pnl, 2)
            summary["positions_closed"].append(result)
            _log("CLOSE", {
                "symbol": symbol, "reason": reason,
                "entry": round(entry, 2), "exit": round(current, 2),
                "pnl": round(pnl, 2),
            })
        except Exception as exc:
            logger.exception("Failed to close %s: %s", symbol, exc)
            summary["errors"].append({"symbol": symbol, "error": str(exc)})


async def _square_off_all() -> dict:
    state = get_risk_state()
    results = []
    symbols = list(state.open_positions.keys())

    if not symbols:
        return {"status": "square_off", "message": "no open positions", "closed": []}

    ltps = {}
    if is_logged_in():
        try:
            from app.services.market_data import get_ltp
            ltps = await get_ltp(symbols)
        except Exception:
            pass

    from app.services.order_manager import place_order

    for symbol in symbols:
        pos = state.open_positions.get(symbol)
        if not pos:
            continue
        current = ltps.get(symbol, pos["entry_price"])
        qty = pos["quantity"]
        close_side = "SELL" if pos.get("side") == "BUY" else "BUY"

        try:
            result = await place_order(
                symbol=symbol, side=close_side, quantity=qty,
                order_type="MARKET", price=current,
            )
            pnl = (current - pos["entry_price"]) * qty if close_side == "SELL" else (pos["entry_price"] - current) * qty
            record_trade_close(symbol, pnl)
            result["pnl"] = round(pnl, 2)
            results.append(result)
            _log("SQUARE_OFF", {"symbol": symbol, "pnl": round(pnl, 2)})
        except Exception as exc:
            logger.exception("Square-off failed for %s: %s", symbol, exc)
            results.append({"symbol": symbol, "error": str(exc)})

    return {"status": "square_off", "closed": results}


# ─────────────────────────────────────────────────────────────────────────────
# Manual trigger
# ─────────────────────────────────────────────────────────────────────────────

async def scan_and_trade_once() -> dict:
    """Run one full scan-and-trade cycle manually."""
    return await trading_cycle()


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _log(action: str, data: dict) -> None:
    entry = {"time": _now_ist(), "action": action, **data}
    _trade_log.append(entry)
    if len(_trade_log) > 500:
        _trade_log.pop(0)
    logger.info("AUTO-TRADE [%s] %s", action, data)

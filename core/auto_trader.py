"""AI-powered automated trading engine for Zerodha Kite.

Runs on a configurable schedule (via APScheduler). Each cycle:
    1. Detect market regime
    2. Scan watchlist for opportunities
    3. Generate AI signals with confidence scoring
    4. Apply risk checks (volatility, drawdown, tilt)
    5. Place orders with dynamic position sizing
    6. Monitor positions with dynamic SL/TP
    7. Auto square-off MIS positions before market close

Works in PAPER mode (simulated) and LIVE mode (real Kite orders).
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional

from core.config import load_config
from core.kite_auth import is_logged_in
from core.risk_engine import (
    check_risk, compute_dynamic_sl_tp, compute_position_size,
    get_risk_analytics, get_risk_state, record_trade_close, record_trade_open,
    should_stop_loss, should_take_profit,
)

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Runtime state
_auto_trading_enabled = False
_trade_log: List[Dict[str, Any]] = []
_cycle_count = 0
_current_regime: Dict[str, Any] = {}


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


def get_trade_log() -> List[Dict[str, Any]]:
    return _trade_log.copy()


def get_cycle_count() -> int:
    return _cycle_count


def get_current_regime() -> Dict[str, Any]:
    return _current_regime.copy()


# ─── Market hours ────────────────────────────────────────────────────────────

def is_market_open() -> bool:
    cfg = load_config()
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    h_open, m_open = map(int, cfg.market_open.split(":"))
    h_close, m_close = map(int, cfg.market_close.split(":"))
    return time(h_open, m_open) <= now.time() <= time(h_close, m_close)


def is_square_off_time() -> bool:
    cfg = load_config()
    now = datetime.now(IST)
    h_close, m_close = map(int, cfg.market_close.split(":"))
    close_dt = datetime.combine(now.date(), time(h_close, m_close))
    square_off = close_dt - timedelta(minutes=cfg.square_off_minutes)
    return now >= square_off.replace(tzinfo=IST)


def _now_ist() -> str:
    return datetime.now(IST).strftime("%H:%M:%S")


# ─── Main trading cycle ─────────────────────────────────────────────────────

async def trading_cycle() -> dict:
    """Execute one full AI-powered auto-trading cycle."""
    global _cycle_count, _current_regime
    _cycle_count += 1
    cfg = load_config()

    summary = {
        "cycle": _cycle_count,
        "timestamp": _now_ist(),
        "signals_found": 0,
        "orders_placed": [],
        "positions_closed": [],
        "skipped": [],
        "errors": [],
        "regime": {},
        "analytics": {},
    }

    # Square-off check first
    if is_square_off_time():
        close_result = await _square_off_all()
        summary["positions_closed"] = close_result.get("closed", [])
        _log("SQUARE_OFF", {"cycle": _cycle_count})
        return summary

    trades_this_cycle = 0
    max_trades = 3

    # 1. Detect market regime
    regime_str = "mean_reverting"
    if cfg.regime_detection:
        try:
            regime_result = await _detect_regime()
            _current_regime = regime_result
            regime_str = regime_result.get("regime", "mean_reverting")
            summary["regime"] = regime_result
        except Exception as e:
            logger.warning("Regime detection failed: %s", e)

    # 2. Scan for intraday opportunities
    if cfg.intraday_enabled and trades_this_cycle < max_trades:
        try:
            from core.scanner import scan_stocks
            sells = await scan_stocks(scan_type="sell", top_n=5)
            for sig in sells:
                if trades_this_cycle >= max_trades:
                    break
                if sig["score"] > -25:
                    continue
                result = await _execute_trade(
                    sig, "SELL", "intraday", summary, regime_str,
                )
                if result:
                    trades_this_cycle += 1
                    summary["signals_found"] += 1
        except Exception as e:
            summary["errors"].append({"scan": "intraday_sell", "error": str(e)})

    # 3. Scan for BTST opportunities
    if cfg.btst_enabled and trades_this_cycle < max_trades:
        try:
            from core.scanner import scan_stocks
            buys = await scan_stocks(scan_type="buy", top_n=5)
            for sig in buys:
                if trades_this_cycle >= max_trades:
                    break
                if sig["score"] < 25:
                    continue
                result = await _execute_trade(
                    sig, "BUY", "btst", summary, regime_str,
                )
                if result:
                    trades_this_cycle += 1
                    summary["signals_found"] += 1
        except Exception as e:
            summary["errors"].append({"scan": "btst_buy", "error": str(e)})

    # 4. Scan for F&O opportunities
    if cfg.fno_enabled and trades_this_cycle < max_trades:
        try:
            from core.scanner import scan_fno_opportunities
            opts = await scan_fno_opportunities(top_n=3)
            for opt in opts:
                if trades_this_cycle >= max_trades:
                    break
                if abs(opt["score"]) < 40:
                    continue
                result = await _execute_trade(
                    opt, "BUY", "fno", summary, regime_str,
                )
                if result:
                    trades_this_cycle += 1
                    summary["signals_found"] += 1
        except Exception as e:
            summary["errors"].append({"scan": "fno", "error": str(e)})

    # 5. Monitor positions
    await _monitor_positions(summary, regime_str)

    # 6. Analytics
    summary["analytics"] = get_risk_analytics()

    _log("CYCLE", {
        "cycle": _cycle_count, "regime": regime_str,
        "signals": summary["signals_found"],
        "orders": len(summary["orders_placed"]),
        "closed": len(summary["positions_closed"]),
    })
    return summary


# ─── Trade execution ─────────────────────────────────────────────────────────

async def _execute_trade(
    signal: dict, side: str, trade_type: str,
    summary: dict, regime: str = "",
) -> Optional[dict]:
    from core.kite_orders import place_order

    cfg = load_config()
    symbol = signal["symbol"]
    price = signal["price"]
    confidence = abs(signal["score"]) / 100.0
    volatility = signal.get("volatility", 0)
    atr = signal.get("atr", 0)
    quality = signal.get("signal_quality", "C")

    # AI position sizing
    qty = compute_position_size(
        price=price, volatility=volatility, atr=atr,
        confidence=confidence, regime=regime,
    )

    # Quality filter
    if cfg.signal_quality_filter and quality == "D":
        summary["skipped"].append({"symbol": symbol, "reason": "Low quality (D)"})
        return None

    # Risk check
    risk = check_risk(
        symbol, side, qty, price, confidence,
        volatility=volatility, regime=regime, signal_quality=quality,
    )
    if not risk["allowed"]:
        summary["skipped"].append({"symbol": symbol, "reason": risk["reason"]})
        return None

    qty = risk["adjusted_qty"]
    product = cfg.intraday_product if trade_type == "intraday" else cfg.btst_product
    exchange = cfg.fno_exchange if trade_type == "fno" else cfg.exchange

    sl_tp = compute_dynamic_sl_tp(price, side, atr=atr, volatility=volatility, regime=regime)

    try:
        result = await place_order(
            symbol=symbol, side=side, quantity=qty,
            order_type="MARKET", price=price,
            product=product, exchange=exchange, trade_type=trade_type,
        )
        record_trade_open(symbol, side, qty, price, confidence, volatility)
        result["score"] = signal["score"]
        result["quality"] = quality
        result["sl"] = sl_tp["stop_loss"]
        result["tp"] = sl_tp["take_profit"]
        summary["orders_placed"].append(result)
        _log("ORDER", {"symbol": symbol, "side": side, "qty": qty, "price": price, "type": trade_type})
        return result
    except Exception as e:
        summary["errors"].append({"symbol": symbol, "error": str(e)})
        return None


# ─── Position monitoring ─────────────────────────────────────────────────────

async def _monitor_positions(summary: dict, regime: str = "") -> None:
    state = get_risk_state()
    if not state.open_positions:
        return

    from core.kite_data import get_ltp
    from core.kite_orders import place_order

    symbols = list(state.open_positions.keys())
    ltps = await get_ltp(symbols)

    to_close = []
    for symbol, pos in state.open_positions.items():
        current = ltps.get(symbol, 0)
        if current <= 0:
            current = pos["entry_price"] * (1 + random.uniform(-0.03, 0.03))

        entry = pos["entry_price"]
        side = pos.get("side", "BUY")

        if should_stop_loss(entry, current, side=side, regime=regime):
            to_close.append((symbol, current, entry, "stop_loss", pos))
        elif should_take_profit(entry, current, side=side, regime=regime):
            to_close.append((symbol, current, entry, "take_profit", pos))

    for symbol, current, entry, reason, pos in to_close:
        qty = pos["quantity"]
        close_side = "SELL" if pos.get("side") == "BUY" else "BUY"
        try:
            result = await place_order(symbol=symbol, side=close_side, quantity=qty, order_type="MARKET", price=current)
            pnl = (current - entry) * qty if close_side == "SELL" else (entry - current) * qty
            record_trade_close(symbol, pnl)
            result["close_reason"] = reason
            result["pnl"] = round(pnl, 2)
            summary["positions_closed"].append(result)
            _log("CLOSE", {"symbol": symbol, "reason": reason, "pnl": round(pnl, 2)})
        except Exception as e:
            summary["errors"].append({"symbol": symbol, "error": str(e)})


async def _square_off_all() -> dict:
    state = get_risk_state()
    if not state.open_positions:
        return {"closed": []}

    from core.kite_data import get_ltp
    from core.kite_orders import place_order

    symbols = list(state.open_positions.keys())
    ltps = await get_ltp(symbols)
    results = []

    for symbol in symbols:
        pos = state.open_positions.get(symbol)
        if not pos:
            continue
        current = ltps.get(symbol, pos["entry_price"])
        qty = pos["quantity"]
        close_side = "SELL" if pos.get("side") == "BUY" else "BUY"
        try:
            result = await place_order(symbol=symbol, side=close_side, quantity=qty, order_type="MARKET", price=current)
            pnl = (current - pos["entry_price"]) * qty if close_side == "SELL" else (pos["entry_price"] - current) * qty
            record_trade_close(symbol, pnl)
            result["pnl"] = round(pnl, 2)
            results.append(result)
        except Exception as e:
            results.append({"symbol": symbol, "error": str(e)})

    return {"closed": results}


# ─── Regime detection ────────────────────────────────────────────────────────

async def _detect_regime() -> Dict[str, Any]:
    """Detect market regime using NIFTY proxy."""
    from core.kite_data import fetch_historical
    from core.feature_engine import build_features

    df = await fetch_historical("RELIANCE", days=300)
    featured = build_features(df)
    if featured.empty:
        return {"regime": "mean_reverting", "confidence": 0.3}

    latest = featured.iloc[-1]
    adx = latest.get("adx", 25)
    vol = latest.get("volatility", 0.2)
    mom = latest.get("momentum_20", 0)
    vol_regime = latest.get("vol_regime", 0.5)

    if vol_regime > 0.8:
        regime = "high_volatility"
        conf = min(vol_regime, 0.95)
    elif adx > 30:
        regime = "trending_up" if mom > 0 else "trending_down"
        conf = min(adx / 50, 0.9)
    elif adx < 20:
        regime = "mean_reverting"
        conf = 0.7
    else:
        regime = "mean_reverting"
        conf = 0.5

    return {
        "regime": regime, "confidence": round(conf, 3),
        "adx": round(float(adx), 1), "volatility": round(float(vol), 4),
        "momentum": round(float(mom), 4),
    }


# ─── Manual trigger ──────────────────────────────────────────────────────────

async def scan_and_trade_once() -> dict:
    return await trading_cycle()


# ─── Logging ─────────────────────────────────────────────────────────────────

def _log(action: str, data: dict) -> None:
    entry = {"time": _now_ist(), "action": action, **data}
    _trade_log.append(entry)
    if len(_trade_log) > 500:
        _trade_log.pop(0)
    logger.info("AUTO-TRADE [%s] %s", action, data)

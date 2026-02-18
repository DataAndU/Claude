"""AI-powered automated trading engine.

Runs on a configurable schedule. Supports:
    - Intraday short-selling (MIS) based on F&O scan scores
    - BTST buying (NRML) based on bullish momentum scores
    - Options trading (NFO) with strike selection via Black-Scholes
    - Position monitoring with dynamic SL/TP auto-exit
    - Auto square-off before market close

AI enhancements (v2):
    - Market regime detection drives strategy selection
    - Adaptive ensemble predictions (RF + LSTM + Transformer)
    - AI confidence scoring with multi-indicator alignment
    - Volatility-adjusted position sizing (ATR-based)
    - Dynamic stop-loss/take-profit based on market conditions
    - Tilt protection after consecutive losses
    - Signal quality grading (A/B/C/D)
    - Multi-timeframe signal confirmation
    - Performance tracking with win-rate analytics

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
    compute_dynamic_sl_tp,
    compute_position_size,
    get_ai_risk_analytics,
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

# AI feature flags
_ai_features = {
    "regime_detection": True,
    "adaptive_ensemble": True,
    "ai_confidence": True,
    "dynamic_sl_tp": True,
    "volatility_sizing": True,
    "signal_quality_filter": True,
    "multi_timeframe": True,
    "tilt_protection": True,
}

# Current market regime (updated each cycle)
_current_regime: Dict[str, Any] = {}


def enable_auto_trading() -> None:
    global _auto_trading_enabled
    _auto_trading_enabled = True
    logger.info("Auto-trading ENABLED (AI features: %s)", _ai_features)


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


def set_ai_features(**features) -> None:
    """Enable/disable individual AI features."""
    for k, v in features.items():
        if k in _ai_features:
            _ai_features[k] = bool(v)


def get_ai_features() -> Dict[str, bool]:
    return _ai_features.copy()


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


def get_current_regime() -> Dict[str, Any]:
    return _current_regime.copy()


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
# Main trading cycle — AI-enhanced
# ─────────────────────────────────────────────────────────────────────────────

async def trading_cycle(model_path: str = "", model_type: str = "rf") -> dict:
    """Execute one full AI-powered auto-trading cycle.

    1. Detect market regime (AI)
    2. Adjust strategy selection based on regime
    3. Scan F&O stocks with AI-enhanced scoring
    4. Filter by adaptive confidence and signal quality
    5. Apply AI risk checks (volatility, drawdown, tilt)
    6. Place orders with dynamic position sizing
    7. Monitor positions with dynamic SL/TP
    8. Auto square-off near close
    """
    global _cycle_count, _current_regime
    _cycle_count += 1

    summary = {
        "cycle": _cycle_count,
        "timestamp": _now_ist(),
        "signals_found": 0,
        "orders_placed": [],
        "positions_closed": [],
        "skipped": [],
        "errors": [],
        "regime": {},
        "ai_analytics": {},
    }

    # Square-off check first
    if is_square_off_time():
        close_result = await _square_off_all()
        summary["positions_closed"] = close_result.get("closed", [])
        _log("SQUARE_OFF", {"cycle": _cycle_count})
        return summary

    use_live = is_logged_in()
    trades_this_cycle = 0

    # ── 0. Detect market regime (AI) ──
    regime_str = "mean_reverting"
    regime_params = {}
    if _ai_features.get("regime_detection"):
        try:
            regime_result = await _detect_market_regime(use_live)
            _current_regime = regime_result
            regime_str = regime_result.get("regime", "mean_reverting")
            if hasattr(regime_str, "value"):
                regime_str = regime_str.value
            regime_params = regime_result.get("params", {})
            summary["regime"] = {
                "type": regime_str,
                "confidence": regime_result.get("confidence", 0),
                "scores": regime_result.get("scores", {}),
            }
            logger.info("Market regime: %s (confidence=%.2f)", regime_str, regime_result.get("confidence", 0))
        except Exception as e:
            logger.warning("Regime detection failed, using default: %s", e)

    # Adjust max trades based on regime
    max_trades = _max_trades_per_cycle + regime_params.get("max_trades_adj", 0)
    max_trades = max(1, max_trades)

    # Determine which strategies are preferred for this regime
    preferred = set(regime_params.get("preferred_strategies", list(_auto_modes.keys())))

    # ── 1. Intraday short-sell scan ──
    if _auto_modes.get("intraday_sell") and trades_this_cycle < max_trades:
        # Boost priority if regime prefers this strategy
        is_preferred = "intraday_sell" in preferred
        try:
            from app.services.fno_scanner import scan_fno_symbols
            sells = await scan_fno_symbols(
                top_n=10, use_live_data=use_live, scan_type="sell"
            )
            for sig in sells:
                if trades_this_cycle >= max_trades:
                    break
                min_score = _min_score_intraday
                if is_preferred:
                    min_score *= 0.85  # lower threshold for preferred strategies
                if sig["score"] < min_score:
                    continue
                if sig["action"] not in ("STRONG SELL", "SELL"):
                    continue

                result = await _execute_equity_trade(
                    symbol=sig["symbol"], side="SELL", price=sig["price"],
                    trade_type="intraday", signal=sig, summary=summary,
                    regime=regime_str,
                )
                if result:
                    trades_this_cycle += 1
                    summary["signals_found"] += 1

        except Exception as e:
            summary["errors"].append({"scan": "intraday_sell", "error": str(e)})
            logger.exception("Intraday sell scan failed")

    # ── 2. BTST buy scan ──
    if _auto_modes.get("btst_buy") and trades_this_cycle < max_trades:
        is_preferred = "btst_buy" in preferred
        try:
            from app.services.fno_scanner import scan_fno_symbols
            buys = await scan_fno_symbols(
                top_n=10, use_live_data=use_live, scan_type="buy"
            )
            for sig in buys:
                if trades_this_cycle >= max_trades:
                    break
                min_score = _min_score_btst
                if is_preferred:
                    min_score *= 0.85
                if sig["score"] < min_score:
                    continue
                if sig["action"] not in ("STRONG BUY", "BUY"):
                    continue

                result = await _execute_equity_trade(
                    symbol=sig["symbol"], side="BUY", price=sig["price"],
                    trade_type="btst", signal=sig, summary=summary,
                    regime=regime_str,
                )
                if result:
                    trades_this_cycle += 1
                    summary["signals_found"] += 1

        except Exception as e:
            summary["errors"].append({"scan": "btst_buy", "error": str(e)})
            logger.exception("BTST buy scan failed")

    # ── 3. Options intraday ──
    if _auto_modes.get("options_intraday") and trades_this_cycle < max_trades:
        is_preferred = "options_intraday" in preferred
        try:
            from app.services.options_chain import scan_options_opportunities
            opts = await scan_options_opportunities(top_n=5, trade_type="intraday")
            for opt in opts:
                if trades_this_cycle >= max_trades:
                    break
                min_score = _min_score_options
                if is_preferred:
                    min_score *= 0.85
                if opt.get("equity_score", 0) < min_score:
                    continue
                result = await _execute_options_trade(opt, "intraday", summary, regime=regime_str)
                if result:
                    trades_this_cycle += 1
                    summary["signals_found"] += 1

        except Exception as e:
            summary["errors"].append({"scan": "options_intraday", "error": str(e)})
            logger.exception("Options intraday scan failed")

    # ── 4. Options BTST ──
    if _auto_modes.get("options_btst") and trades_this_cycle < max_trades:
        is_preferred = "options_btst" in preferred
        try:
            from app.services.options_chain import scan_options_opportunities
            opts = await scan_options_opportunities(top_n=5, trade_type="btst")
            for opt in opts:
                if trades_this_cycle >= max_trades:
                    break
                min_score = _min_score_options
                if is_preferred:
                    min_score *= 0.85
                if opt.get("equity_score", 0) < min_score:
                    continue
                result = await _execute_options_trade(opt, "btst", summary, regime=regime_str)
                if result:
                    trades_this_cycle += 1
                    summary["signals_found"] += 1

        except Exception as e:
            summary["errors"].append({"scan": "options_btst", "error": str(e)})
            logger.exception("Options BTST scan failed")

    # ── 5. Monitor positions with dynamic SL/TP ──
    await _monitor_positions(summary, regime=regime_str)

    # ── 6. AI analytics ──
    summary["ai_analytics"] = get_ai_risk_analytics()

    _log("CYCLE", {
        "cycle": _cycle_count,
        "regime": regime_str,
        "signals": summary["signals_found"],
        "orders": len(summary["orders_placed"]),
        "closed": len(summary["positions_closed"]),
        "errors": len(summary["errors"]),
    })
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Regime detection
# ─────────────────────────────────────────────────────────────────────────────

async def _detect_market_regime(use_live: bool) -> Dict[str, Any]:
    """Detect overall market regime using a representative index/symbol."""
    from app.services.regime_detector import detect_regime

    # Use NIFTY50 proxy (RELIANCE as liquid representative)
    representative = "RELIANCE"
    try:
        if use_live:
            from app.services.market_data import fetch_daily_prices
            df = await fetch_daily_prices(representative, days=300, use_cache=True)
        else:
            from app.services.fno_scanner import _generate_synthetic_ohlcv
            df = _generate_synthetic_ohlcv(representative)

        return detect_regime(df, symbol=representative)
    except Exception as e:
        logger.warning("Regime detection failed: %s", e)
        return {"regime": "mean_reverting", "confidence": 0.3, "params": {}, "scores": {}}


# ─────────────────────────────────────────────────────────────────────────────
# Trade execution — AI-enhanced
# ─────────────────────────────────────────────────────────────────────────────

async def _execute_equity_trade(
    symbol: str, side: str, price: float, trade_type: str,
    signal: dict, summary: dict, regime: str = "",
) -> Optional[dict]:
    from app.services.order_manager import place_order

    confidence = signal.get("score", 0) / 100.0
    volatility = signal.get("volatility", 0)
    atr = signal.get("atr", 0)

    # AI-powered position sizing
    qty = compute_position_size(
        price=price,
        volatility=volatility,
        atr=atr,
        confidence=confidence,
        regime=regime,
    )

    # Signal quality filtering
    signal_quality = signal.get("signal_quality", "C")
    if _ai_features.get("signal_quality_filter") and signal_quality == "D":
        summary["skipped"].append({
            "symbol": symbol, "side": side,
            "reason": f"Low signal quality: {signal_quality}",
        })
        return None

    # AI risk check with regime awareness
    risk = check_risk(
        symbol=symbol, side=side, qty=qty, price=price,
        confidence=confidence, volatility=volatility,
        regime=regime, signal_quality=signal_quality,
    )
    if not risk["allowed"]:
        summary["skipped"].append({
            "symbol": symbol, "side": side, "reason": risk["reason"],
        })
        return None

    qty = risk["adjusted_qty"]
    product = "MIS" if trade_type == "intraday" else settings.btst_product

    # Compute dynamic SL/TP levels
    sl_tp = compute_dynamic_sl_tp(
        entry_price=price, side=side, atr=atr,
        volatility=volatility, regime=regime,
    )

    try:
        result = await place_order(
            symbol=symbol, side=side, quantity=qty,
            order_type="MARKET", price=price,
            product=product, exchange="NSE", trade_type=trade_type,
        )
        record_trade_open(symbol, side, qty, price, confidence=confidence, volatility=volatility)
        result["score"] = signal.get("score", 0)
        result["action"] = signal.get("action", "")
        result["reasons"] = signal.get("reasons", [])
        result["signal_quality"] = signal_quality
        result["regime"] = regime
        result["dynamic_sl"] = sl_tp["stop_loss"]
        result["dynamic_tp"] = sl_tp["take_profit"]
        result["confidence_threshold"] = risk.get("confidence_threshold_used", 0)
        summary["orders_placed"].append(result)
        _log("ORDER", {
            "symbol": symbol, "side": side, "qty": qty,
            "price": price, "type": trade_type,
            "score": signal.get("score", 0),
            "regime": regime, "quality": signal_quality,
            "sl": sl_tp["stop_loss"], "tp": sl_tp["take_profit"],
        })
        return result
    except Exception as e:
        summary["errors"].append({"symbol": symbol, "error": str(e)})
        logger.exception("Failed to place equity order for %s", symbol)
        return None


async def _execute_options_trade(
    option: dict, trade_type: str, summary: dict, regime: str = "",
) -> Optional[dict]:
    from app.services.order_manager import place_order

    symbol = option.get("nfo_symbol", "")
    underlying = option.get("symbol", "")
    entry_price = option.get("entry_price", 0)
    qty = option.get("quantity", option.get("lot_size", 1))
    eq_score = option.get("equity_score", 0)
    confidence = eq_score / 100.0
    volatility = option.get("volatility", 0)

    signal_quality = option.get("signal_quality", "C")
    if _ai_features.get("signal_quality_filter") and signal_quality == "D":
        summary["skipped"].append({
            "symbol": symbol, "side": "BUY",
            "reason": f"Low signal quality: {signal_quality}",
        })
        return None

    risk = check_risk(
        underlying, "BUY", qty, entry_price, confidence,
        volatility=volatility, regime=regime, signal_quality=signal_quality,
    )
    if not risk["allowed"]:
        summary["skipped"].append({
            "symbol": symbol, "side": "BUY", "reason": risk["reason"],
        })
        return None

    product = option.get("product", "MIS")
    sl_tp = compute_dynamic_sl_tp(
        entry_price=entry_price, side="BUY",
        volatility=volatility, regime=regime,
    )

    try:
        result = await place_order(
            symbol=symbol, side="BUY", quantity=qty,
            order_type="MARKET", price=entry_price,
            product=product, exchange="NFO", trade_type=trade_type,
        )
        record_trade_open(symbol, "BUY", qty, entry_price, confidence=confidence, volatility=volatility)
        result["option_type"] = option.get("option_type", "")
        result["strike"] = option.get("strike", 0)
        result["underlying"] = underlying
        result["direction"] = option.get("direction", "")
        result["sl_price"] = sl_tp["stop_loss"]
        result["target_price"] = sl_tp["take_profit"]
        result["equity_score"] = eq_score
        result["regime"] = regime
        result["signal_quality"] = signal_quality
        summary["orders_placed"].append(result)
        _log("OPTIONS", {
            "symbol": symbol, "underlying": underlying,
            "type": option.get("option_type"),
            "strike": option.get("strike"),
            "entry": entry_price, "qty": qty,
            "regime": regime,
            "sl": sl_tp["stop_loss"], "tp": sl_tp["take_profit"],
        })
        return result
    except Exception as e:
        summary["errors"].append({"symbol": symbol, "error": str(e)})
        logger.exception("Failed to place options order for %s", symbol)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Position monitoring — AI-enhanced with dynamic SL/TP
# ─────────────────────────────────────────────────────────────────────────────

async def _monitor_positions(summary: dict, regime: str = "") -> None:
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
        pos_volatility = pos.get("volatility", 0)

        # Use dynamic SL/TP based on position metadata and current regime
        if _ai_features.get("dynamic_sl_tp"):
            if should_stop_loss(entry, current, side=side, regime=regime):
                to_close.append((symbol, current, entry, "stop_loss", pos))
            elif should_take_profit(entry, current, side=side, regime=regime):
                to_close.append((symbol, current, entry, "take_profit", pos))
        else:
            # Fallback to static SL/TP
            if side == "BUY":
                if current <= entry * (1 - settings.stop_loss_pct):
                    to_close.append((symbol, current, entry, "stop_loss", pos))
                elif current >= entry * (1 + settings.take_profit_pct):
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
            record_trade_close(symbol, pnl, confidence=pos.get("confidence", 0))
            result["close_reason"] = reason
            result["pnl"] = round(pnl, 2)
            result["regime"] = regime
            summary["positions_closed"].append(result)
            _log("CLOSE", {
                "symbol": symbol, "reason": reason,
                "entry": round(entry, 2), "exit": round(current, 2),
                "pnl": round(pnl, 2), "regime": regime,
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
            record_trade_close(symbol, pnl, confidence=pos.get("confidence", 0))
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
    """Run one full AI-powered scan-and-trade cycle manually."""
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

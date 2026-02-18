"""Options chain analysis — fetch option chains, select strikes, analyse Greeks.

Supports both live (Kite API) and paper (synthetic) modes.
Designed for F&O stocks and index options on NSE/NFO.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.kite import get_kite, is_logged_in

logger = logging.getLogger(__name__)
settings = get_settings()

IST = timezone(timedelta(hours=5, minutes=30))

# Standard lot sizes for major F&O stocks (approximate — Kite instruments have exact values)
LOT_SIZES: Dict[str, int] = {
    "RELIANCE": 250, "TCS": 175, "INFY": 400, "HDFCBANK": 550,
    "ICICIBANK": 700, "SBIN": 750, "BAJFINANCE": 125, "ITC": 1600,
    "HINDUNILVR": 300, "KOTAKBANK": 400, "TATAMOTORS": 575,
    "MARUTI": 100, "AXISBANK": 600, "LT": 150, "SUNPHARMA": 350,
    "TITAN": 375, "ADANIENT": 250, "BHARTIARTL": 475, "WIPRO": 1500,
    "HCLTECH": 350, "TATASTEEL": 550, "POWERGRID": 2700,
    "NTPC": 2325, "ONGC": 3075, "BPCL": 1800, "COALINDIA": 2100,
    "DRREDDY": 125, "DIVISLAB": 200, "CIPLA": 650, "APOLLOHOSP": 250,
    "TECHM": 600, "ULTRACEMCO": 100, "GRASIM": 475, "HEROMOTOCO": 150,
    "EICHERMOT": 175, "M&M": 350, "BAJAJ-AUTO": 250, "JSWSTEEL": 450,
    "ASIANPAINT": 300, "NESTLEIND": 50,
    "NIFTY": 50, "BANKNIFTY": 15, "FINNIFTY": 40,
}

# Strike intervals for major symbols
STRIKE_INTERVALS: Dict[str, float] = {
    "NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50,
    "RELIANCE": 20, "TCS": 50, "INFY": 20, "HDFCBANK": 20,
    "ICICIBANK": 20, "SBIN": 10, "BAJFINANCE": 100, "ITC": 5,
    "HINDUNILVR": 50, "KOTAKBANK": 20, "TATAMOTORS": 10,
    "MARUTI": 200, "AXISBANK": 20, "LT": 50, "SUNPHARMA": 20,
    "TITAN": 50, "ADANIENT": 50, "BHARTIARTL": 20, "WIPRO": 10,
    "HCLTECH": 20,
}

DEFAULT_STRIKE_INTERVAL = 10.0


def get_lot_size(symbol: str) -> int:
    """Get the lot size for a symbol."""
    return LOT_SIZES.get(symbol, 100)


def get_strike_interval(symbol: str) -> float:
    """Get the strike price interval for a symbol."""
    return STRIKE_INTERVALS.get(symbol, DEFAULT_STRIKE_INTERVAL)


def round_to_strike(price: float, interval: float) -> float:
    """Round a price to the nearest strike price."""
    return round(price / interval) * interval


def get_atm_strike(spot_price: float, symbol: str) -> float:
    """Get the ATM (At The Money) strike for a given spot price."""
    interval = get_strike_interval(symbol)
    return round_to_strike(spot_price, interval)


def get_strike_range(spot_price: float, symbol: str, n_strikes: int = 5) -> List[float]:
    """Get a range of strikes around ATM."""
    interval = get_strike_interval(symbol)
    atm = get_atm_strike(spot_price, symbol)
    strikes = []
    for i in range(-n_strikes, n_strikes + 1):
        strikes.append(atm + i * interval)
    return strikes


def _synthetic_option_price(
    spot: float, strike: float, option_type: str,
    days_to_expiry: int = 7, iv: float = 0.20
) -> Dict[str, Any]:
    """Generate synthetic option price data for paper mode using Black-Scholes approximation."""
    t = max(days_to_expiry / 365.0, 0.001)
    r = 0.065  # risk-free rate ~6.5%

    d1 = (math.log(spot / strike) + (r + 0.5 * iv**2) * t) / (iv * math.sqrt(t))
    d2 = d1 - iv * math.sqrt(t)

    from statistics import NormalDist
    nd = NormalDist()

    if option_type == "CE":
        price = spot * nd.cdf(d1) - strike * math.exp(-r * t) * nd.cdf(d2)
        delta = nd.cdf(d1)
    else:  # PE
        price = strike * math.exp(-r * t) * nd.cdf(-d2) - spot * nd.cdf(-d1)
        delta = nd.cdf(d1) - 1

    gamma = nd.pdf(d1) / (spot * iv * math.sqrt(t))
    theta = -(spot * nd.pdf(d1) * iv) / (2 * math.sqrt(t)) / 365
    vega = spot * nd.pdf(d1) * math.sqrt(t) / 100

    # Add some randomness for realism
    noise = np.random.uniform(0.95, 1.05)
    price = max(0.05, price * noise)

    return {
        "last_price": round(price, 2),
        "bid": round(price * 0.98, 2),
        "ask": round(price * 1.02, 2),
        "volume": int(np.random.uniform(1000, 50000)),
        "oi": int(np.random.uniform(50000, 500000)),
        "iv": round(iv * 100, 2),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
    }


async def get_option_chain(
    symbol: str, spot_price: float, n_strikes: int = 5,
    days_to_expiry: int = 7
) -> Dict[str, Any]:
    """Fetch or generate option chain for a symbol.

    Returns chain with CE and PE data for strikes around ATM.
    """
    strikes = get_strike_range(spot_price, symbol, n_strikes)
    atm_strike = get_atm_strike(spot_price, symbol)
    lot_size = get_lot_size(symbol)

    chain = []
    for strike in strikes:
        moneyness = "ATM" if strike == atm_strike else ("ITM" if (
            strike < spot_price) else "OTM")

        iv_base = np.random.uniform(0.15, 0.35)

        ce_data = _synthetic_option_price(
            spot_price, strike, "CE", days_to_expiry, iv_base
        )
        pe_data = _synthetic_option_price(
            spot_price, strike, "PE", days_to_expiry, iv_base
        )

        chain.append({
            "strike": strike,
            "moneyness_ce": moneyness if strike <= spot_price else "OTM",
            "moneyness_pe": moneyness if strike >= spot_price else "OTM",
            "ce": ce_data,
            "pe": pe_data,
        })

    return {
        "symbol": symbol,
        "spot_price": round(spot_price, 2),
        "atm_strike": atm_strike,
        "lot_size": lot_size,
        "days_to_expiry": days_to_expiry,
        "chain": chain,
    }


async def select_best_option(
    symbol: str, spot_price: float, direction: str,
    trade_type: str = "intraday", days_to_expiry: int = 7
) -> Dict[str, Any]:
    """Select the best option strike for a given trade direction.

    Args:
        symbol: Stock/index symbol
        spot_price: Current spot price
        direction: "SELL" (bearish — buy PUT) or "BUY" (bullish — buy CALL)
        trade_type: "intraday" or "btst"
        days_to_expiry: Days to nearest expiry

    Returns:
        Dict with recommended option details
    """
    offset = settings.options_strike_offset
    interval = get_strike_interval(symbol)
    atm_strike = get_atm_strike(spot_price, symbol)
    lot_size = get_lot_size(symbol)

    if direction == "SELL":
        # Bearish: Buy PUT at ATM or slightly OTM
        option_type = "PE"
        strike = atm_strike + offset * interval  # +offset = OTM for puts
    else:
        # Bullish: Buy CALL at ATM or slightly OTM
        option_type = "CE"
        strike = atm_strike + offset * interval  # +offset = OTM for calls

    iv_base = np.random.uniform(0.15, 0.30)
    opt_data = _synthetic_option_price(spot_price, strike, option_type, days_to_expiry, iv_base)

    product = "MIS" if trade_type == "intraday" else settings.btst_product
    qty = lot_size * settings.options_lot_size

    # Compute option SL/target
    if trade_type == "intraday":
        sl_pct = settings.stop_loss_pct
        tp_pct = settings.take_profit_pct
    else:
        sl_pct = settings.btst_stoploss_pct
        tp_pct = settings.btst_target_pct

    entry_price = opt_data["last_price"]
    sl_price = round(entry_price * (1 - sl_pct), 2)
    target_price = round(entry_price * (1 + tp_pct), 2)

    # Build NFO trading symbol (approximate format)
    now = datetime.now(IST)
    expiry_date = now + timedelta(days=days_to_expiry)
    # Adjust to Thursday (NFO weekly expiry)
    while expiry_date.weekday() != 3:  # Thursday
        expiry_date += timedelta(days=1)

    month_map = {1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
                 7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"}

    if symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY"):
        # Weekly format: NIFTY2502727000CE
        nfo_symbol = f"{symbol}{expiry_date.strftime('%y')}{expiry_date.strftime('%m')}{expiry_date.strftime('%d')}{int(strike)}{option_type}"
    else:
        # Monthly format: RELIANCE25FEB2500CE
        nfo_symbol = f"{symbol}{expiry_date.strftime('%y')}{month_map[expiry_date.month]}{int(strike)}{option_type}"

    return {
        "symbol": symbol,
        "nfo_symbol": nfo_symbol,
        "option_type": option_type,
        "strike": strike,
        "spot_price": round(spot_price, 2),
        "atm_strike": atm_strike,
        "entry_price": entry_price,
        "sl_price": sl_price,
        "target_price": target_price,
        "lot_size": lot_size,
        "quantity": qty,
        "product": product,
        "exchange": "NFO",
        "trade_type": trade_type,
        "direction": direction,
        "days_to_expiry": days_to_expiry,
        "greeks": {
            "delta": opt_data["delta"],
            "gamma": opt_data["gamma"],
            "theta": opt_data["theta"],
            "vega": opt_data["vega"],
            "iv": opt_data["iv"],
        },
        "volume": opt_data["volume"],
        "oi": opt_data["oi"],
    }


async def scan_options_opportunities(
    symbols: Optional[List[str]] = None, top_n: int = 10,
    trade_type: str = "intraday"
) -> List[Dict[str, Any]]:
    """Scan symbols and return the best options trading opportunities.

    Combines the F&O equity scanner with options chain analysis.
    """
    from app.services.fno_scanner import scan_fno_symbols

    # Get equity signals first
    equity_signals = await scan_fno_symbols(
        symbols=symbols, top_n=20, use_live_data=False
    )

    opportunities = []
    for sig in equity_signals:
        if sig["action"] in ("STRONG SELL", "SELL"):
            direction = "SELL"
        elif sig["score"] < 20:
            direction = "BUY"
        else:
            continue

        try:
            opt = await select_best_option(
                symbol=sig["symbol"],
                spot_price=sig["price"],
                direction=direction,
                trade_type=trade_type,
            )
            opt["equity_score"] = sig["score"]
            opt["equity_action"] = sig["action"]
            opt["equity_reasons"] = sig["reasons"]
            opportunities.append(opt)
        except Exception as exc:
            logger.warning("Options scan failed for %s: %s", sig["symbol"], exc)

    # Sort by equity score (higher = stronger signal)
    opportunities.sort(key=lambda x: x["equity_score"], reverse=True)
    return opportunities[:top_n]

#!/usr/bin/env python3
"""DOOLSH TRADING AI — Terminal Dashboard.

Full trading interface that runs inside Userland terminal.
No browser needed — everything works right here.

Usage:
    python trade.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import signal
import time
from datetime import datetime, timedelta, timezone

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IST = timezone(timedelta(hours=5, minutes=30))

# ── ANSI Colours ──────────────────────────────────────────────────────────────
G = "\033[0;32m"   # green
R = "\033[0;31m"   # red
Y = "\033[1;33m"   # yellow
C = "\033[0;36m"   # cyan
B = "\033[1;34m"   # blue
M = "\033[0;35m"   # magenta
W = "\033[1;37m"   # white bold
DIM = "\033[2m"    # dim
N = "\033[0m"      # reset


def clear():
    os.system("clear" if os.name != "nt" else "cls")


def banner():
    print(f"""
{C}╔══════════════════════════════════════════════════════════════╗
║{W}           DOOLSH TRADING AI  v2.1{C}                            ║
║{DIM}     F&O Options + Stocks | Intraday + BTST{C}                  ║
╚══════════════════════════════════════════════════════════════╝{N}
""")


def header(text: str):
    w = 60
    pad = max(0, w - len(text) - 4)
    print(f"\n{B}┌{'─'*w}┐{N}")
    print(f"{B}│{N}  {W}{text}{N}{' '*pad}{B}│{N}")
    print(f"{B}└{'─'*w}┘{N}")


def row(label: str, value: str, color: str = N):
    print(f"  {DIM}{label:<22}{N} {color}{value}{N}")


def table_header(*cols):
    fmt = ""
    for c, w in cols:
        fmt += f"  {W}{c:<{w}}{N}"
    print(fmt)
    total = sum(w for _, w in cols) + 2 * len(cols)
    print(f"  {DIM}{'─'*total}{N}")


def signal_row(r: dict):
    """Print a single scan result row."""
    sym = r.get("symbol", "?")
    score = r.get("score", 0)
    action = r.get("action", "?")
    direction = r.get("direction", "?")
    price = r.get("price", 0)
    rsi = r.get("rsi", 0)
    ret1 = r.get("return_1d_pct", 0)

    # Color the action
    if "STRONG" in action:
        ac = f"{R if 'SELL' in action else G}{action}{N}"
    elif action in ("SELL", "BUY"):
        ac = f"{Y}{action}{N}"
    else:
        ac = f"{DIM}{action}{N}"

    # Color the score
    if score >= 60:
        sc = f"{G}{score:5.1f}{N}"
    elif score >= 40:
        sc = f"{Y}{score:5.1f}{N}"
    else:
        sc = f"{DIM}{score:5.1f}{N}"

    # Color return
    rc = f"{G if ret1 >= 0 else R}{ret1:+6.2f}%{N}"

    print(f"  {W}{sym:<14}{N} {sc}  {ac:<22}  {C}{price:>10.2f}{N}  RSI {rsi:5.1f}  {rc}")


def option_row(o: dict):
    """Print a single option recommendation row."""
    sym = o.get("symbol", "?")
    otype = o.get("option_type", "?")
    strike = o.get("strike", 0)
    entry = o.get("entry_price", 0)
    sl = o.get("sl_price", 0)
    target = o.get("target_price", 0)
    lot = o.get("lot_size", 0)
    product = o.get("product", "?")
    direction = o.get("direction", "?")
    eq_score = o.get("equity_score", 0)

    dc = G if direction == "BUY" else R
    print(f"  {W}{sym:<12}{N} {dc}{direction:<5}{N} {Y}{otype}{N} {C}{strike:>8.0f}{N}  "
          f"Entry {G}{entry:>8.2f}{N}  SL {R}{sl:>8.2f}{N}  Tgt {G}{target:>8.2f}{N}  "
          f"Lot {lot}  {DIM}{product}{N}  Score {eq_score:.0f}")


def order_row(o: dict):
    """Print a single order row."""
    oid = o.get("order_id", "?")
    sym = o.get("symbol", "?")
    side = o.get("side", "?")
    qty = o.get("quantity", 0)
    status = o.get("status", "?")
    price = o.get("price", 0)
    mode = o.get("mode", "?")
    ttype = o.get("trade_type", "?")

    sc = G if side == "BUY" else R
    stc = G if status == "COMPLETE" else Y
    print(f"  {DIM}{oid:<14}{N} {W}{sym:<12}{N} {sc}{side:<5}{N} qty={qty:<4} "
          f"{stc}{status:<10}{N} {C}{price:>10.2f}{N}  {DIM}{mode}/{ttype}{N}")


# ── Async Scanner Calls ──────────────────────────────────────────────────────

async def run_scan(scan_type: str = "sell", top_n: int = 15):
    from app.services.fno_scanner import scan_fno_symbols
    return await scan_fno_symbols(top_n=top_n, scan_type=scan_type)


async def run_options_scan(trade_type: str = "intraday", top_n: int = 10):
    from app.services.options_chain import scan_options_opportunities
    return await scan_options_opportunities(top_n=top_n, trade_type=trade_type)


async def run_options_chain(symbol: str, spot_price: float, n_strikes: int = 5):
    from app.services.options_chain import get_option_chain
    return await get_option_chain(symbol, spot_price, n_strikes)


async def get_orders():
    from app.services.order_manager import get_paper_orders
    return get_paper_orders()


async def place_trade(symbol: str, side: str, qty: int, trade_type: str = "intraday"):
    from app.services.order_manager import place_order
    return await place_order(
        symbol=symbol, side=side, quantity=qty,
        trade_type=trade_type,
    )


async def try_kite_login():
    """Attempt Kite auto-login. Returns status string."""
    try:
        from app.core.kite import auto_login, is_logged_in
        if is_logged_in():
            return "CONNECTED"
        token = await auto_login()
        return "CONNECTED" if token else "FAILED"
    except Exception as e:
        return f"OFFLINE ({e})"


def kite_status():
    try:
        from app.core.kite import is_logged_in
        return is_logged_in()
    except Exception:
        return False


# ── Menu Actions ─────────────────────────────────────────────────────────────

def show_intraday_sells():
    header("INTRADAY SELL SIGNALS (F&O Stocks)")
    print(f"  {DIM}Scanning 40 F&O stocks for short-sell opportunities...{N}\n")
    results = asyncio.get_event_loop().run_until_complete(run_scan("sell", 15))
    if not results:
        print(f"  {Y}No strong sell signals found right now.{N}")
        return
    table_header(("SYMBOL", 14), ("SCORE", 7), ("ACTION", 18), ("PRICE", 12), ("RSI", 9), ("1D RET", 8))
    for r in results:
        signal_row(r)
    print(f"\n  {DIM}Top {len(results)} results shown. Higher score = stronger sell signal.{N}")


def show_btst_buys():
    header("BTST BUY SIGNALS (Buy Today Sell Tomorrow)")
    print(f"  {DIM}Scanning 40 F&O stocks for bullish BTST opportunities...{N}\n")
    results = asyncio.get_event_loop().run_until_complete(run_scan("buy", 15))
    if not results:
        print(f"  {Y}No strong buy signals found right now.{N}")
        return
    table_header(("SYMBOL", 14), ("SCORE", 7), ("ACTION", 18), ("PRICE", 12), ("RSI", 9), ("1D RET", 8))
    for r in results:
        signal_row(r)
    print(f"\n  {DIM}Top {len(results)} results shown. Higher score = stronger buy signal.{N}")


def show_all_signals():
    header("ALL SIGNALS (Intraday + BTST)")
    print(f"  {DIM}Scanning for both SELL and BUY signals...{N}\n")
    results = asyncio.get_event_loop().run_until_complete(run_scan("both", 20))
    if not results:
        print(f"  {Y}No signals found.{N}")
        return
    table_header(("SYMBOL", 14), ("SCORE", 7), ("ACTION", 18), ("PRICE", 12), ("RSI", 9), ("1D RET", 8))
    for r in results:
        signal_row(r)
    print(f"\n  {DIM}{len(results)} signals shown.{N}")


def show_options_intraday():
    header("OPTIONS — INTRADAY RECOMMENDATIONS")
    print(f"  {DIM}Finding best option strikes for intraday trading...{N}\n")
    results = asyncio.get_event_loop().run_until_complete(run_options_scan("intraday", 10))
    if not results:
        print(f"  {Y}No strong options signals found.{N}")
        return
    for o in results:
        option_row(o)
        reasons = o.get("equity_reasons", [])
        if reasons:
            print(f"    {DIM}→ {', '.join(reasons[:3])}{N}")
    print(f"\n  {DIM}{len(results)} option recommendations shown.{N}")


def show_options_btst():
    header("OPTIONS — BTST RECOMMENDATIONS")
    print(f"  {DIM}Finding best option strikes for BTST carry-forward...{N}\n")
    results = asyncio.get_event_loop().run_until_complete(run_options_scan("btst", 10))
    if not results:
        print(f"  {Y}No BTST options signals found.{N}")
        return
    for o in results:
        option_row(o)
        reasons = o.get("equity_reasons", [])
        if reasons:
            print(f"    {DIM}→ {', '.join(reasons[:3])}{N}")
    print(f"\n  {DIM}{len(results)} BTST option recommendations shown.{N}")


def show_option_chain():
    header("OPTION CHAIN VIEWER")
    symbol = input(f"  {C}Symbol{N} (e.g. RELIANCE): ").strip().upper()
    if not symbol:
        return
    try:
        spot_str = input(f"  {C}Spot price{N} (e.g. 2500): ").strip()
        spot = float(spot_str) if spot_str else 2500.0
    except ValueError:
        spot = 2500.0

    print(f"\n  {DIM}Generating option chain for {symbol} @ {spot}...{N}\n")
    chain = asyncio.get_event_loop().run_until_complete(run_options_chain(symbol, spot, 5))

    lot = chain.get("lot_size", 100)
    atm = chain.get("atm_strike", 0)
    print(f"  {W}Symbol:{N} {symbol}  {W}Spot:{N} {spot}  {W}ATM:{N} {atm}  {W}Lot:{N} {lot}\n")

    # Table header
    print(f"  {G}{'CE OI':>8} {'CE Vol':>8} {'CE Price':>9} {'CE Δ':>7}{N}  "
          f"{W}{'STRIKE':>8}{N}  "
          f"{R}{'PE Δ':>7} {'PE Price':>9} {'PE Vol':>8} {'PE OI':>8}{N}")
    print(f"  {DIM}{'─'*80}{N}")

    for entry in chain.get("chain", []):
        strike = entry["strike"]
        ce = entry["ce"]
        pe = entry["pe"]
        atm_mark = " ◄" if strike == atm else ""

        print(f"  {G}{ce['oi']:>8,} {ce['volume']:>8,} {ce['last_price']:>9.2f} {ce['delta']:>7.4f}{N}  "
              f"{W}{strike:>8.0f}{N}{Y}{atm_mark}{N}  "
              f"{R}{pe['delta']:>7.4f} {pe['last_price']:>9.2f} {pe['volume']:>8,} {pe['oi']:>8,}{N}")


def show_orders():
    header("ORDER BOOK")
    orders = asyncio.get_event_loop().run_until_complete(get_orders())
    if not orders:
        print(f"  {DIM}No orders placed in this session.{N}")
        return
    for o in orders:
        order_row(o)
    print(f"\n  {DIM}Total: {len(orders)} orders{N}")


def place_order_interactive():
    header("PLACE ORDER")
    from app.core.config import get_settings
    s = get_settings()
    print(f"  {W}Mode:{N} {G if s.trading_mode == 'paper' else R}{s.trading_mode.upper()}{N}")
    if s.trading_mode == "live":
        print(f"  {R}WARNING: LIVE MODE — real money will be used!{N}")

    symbol = input(f"\n  {C}Symbol{N} (e.g. RELIANCE): ").strip().upper()
    if not symbol:
        return

    side = input(f"  {C}Side{N} (BUY/SELL): ").strip().upper()
    if side not in ("BUY", "SELL"):
        print(f"  {R}Invalid side. Must be BUY or SELL.{N}")
        return

    try:
        qty = int(input(f"  {C}Quantity{N} (default {s.default_quantity}): ").strip() or s.default_quantity)
    except ValueError:
        qty = s.default_quantity

    ttype = input(f"  {C}Trade type{N} (intraday/btst, default intraday): ").strip().lower()
    if ttype not in ("intraday", "btst"):
        ttype = "intraday"

    product = "MIS" if ttype == "intraday" else "NRML"
    print(f"\n  {W}Confirm:{N} {side} {symbol} x{qty} ({ttype}/{product})")
    confirm = input(f"  {Y}Place order? (y/n):{N} ").strip().lower()
    if confirm != "y":
        print(f"  {DIM}Order cancelled.{N}")
        return

    result = asyncio.get_event_loop().run_until_complete(
        place_trade(symbol, side, qty, ttype)
    )
    status = result.get("status", "?")
    oid = result.get("order_id", "?")
    print(f"\n  {G}Order placed!{N} ID: {oid}  Status: {status}")


def show_risk_config():
    header("RISK MANAGEMENT & CONFIG")
    from app.core.config import get_settings
    s = get_settings()
    now = datetime.now(IST)
    market_open = now.replace(hour=s.market_open_hour, minute=s.market_open_minute, second=0)
    market_close = now.replace(hour=s.market_close_hour, minute=s.market_close_minute, second=0)
    is_market = market_open <= now <= market_close and now.weekday() < 5

    row("Trading Mode", s.trading_mode.upper(), G if s.trading_mode == "paper" else R)
    row("Kite User", s.kite_user_id or "Not set")
    row("Kite Connected", "Yes" if kite_status() else "No", G if kite_status() else R)
    row("Market Hours", f"{s.market_open_hour}:{s.market_open_minute:02d} - {s.market_close_hour}:{s.market_close_minute:02d}")
    row("Market Status", "OPEN" if is_market else "CLOSED", G if is_market else R)
    row("Current Time (IST)", now.strftime("%H:%M:%S  %d-%b-%Y"))
    print()
    row("Max Daily Loss", f"₹{s.max_daily_loss:,.0f}")
    row("Max Position Value", f"₹{s.max_position_value:,.0f}")
    row("Max Open Positions", str(s.max_open_positions))
    row("Stop Loss %", f"{s.stop_loss_pct*100:.1f}%")
    row("Take Profit %", f"{s.take_profit_pct*100:.1f}%")
    row("Max Trades/Day", str(s.max_trade_count_per_day))
    row("Min Confidence", f"{s.min_confidence_threshold*100:.0f}%")
    print()
    row("Options Enabled", str(s.options_enabled), G if s.options_enabled else DIM)
    row("Options Lot Size", str(s.options_lot_size))
    row("BTST Enabled", str(s.btst_enabled), G if s.btst_enabled else DIM)
    row("BTST Product", s.btst_product)
    row("BTST Target %", f"{s.btst_target_pct*100:.1f}%")
    row("BTST StopLoss %", f"{s.btst_stoploss_pct*100:.1f}%")


def try_connect_kite():
    header("KITE CONNECTION")
    from app.core.config import get_settings
    s = get_settings()
    row("Kite User", s.kite_user_id or "Not set")
    row("API Key", (s.kite_api_key[:6] + "***") if s.kite_api_key else "Not set")
    row("TOTP Key", "Configured" if s.kite_totp_secret else "Not set", G if s.kite_totp_secret else R)
    print()

    if not all([s.kite_user_id, s.kite_password, s.kite_totp_secret, s.kite_api_key]):
        print(f"  {R}Missing Kite credentials in .env file.{N}")
        print(f"  {DIM}Set KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET, KITE_API_KEY{N}")
        return

    print(f"  {C}Step 1:{N} Sending login credentials...")
    print(f"  {C}Step 2:{N} Submitting TOTP...")
    print(f"  {C}Step 3:{N} Extracting request token...")
    print(f"  {C}Step 4:{N} Generating session...")
    print()
    status = asyncio.get_event_loop().run_until_complete(try_kite_login())
    if status == "CONNECTED":
        print(f"  {G}Connected to Kite successfully!{N}")
        print(f"  {G}You can now trade in LIVE mode.{N}")
    else:
        print(f"  {Y}{status}{N}")
        print()
        print(f"  {DIM}Possible causes:{N}")
        print(f"  {DIM}  - Invalid credentials or TOTP key{N}")
        print(f"  {DIM}  - Kite API might be down or rate-limited{N}")
        print(f"  {DIM}  - Network connectivity issue{N}")
        print()
        print(f"  {DIM}Paper mode still works without Kite connection.{N}")


def show_watchlist():
    header("WATCHLIST")
    from app.services.fno_scanner import FNO_SYMBOLS
    for i, sym in enumerate(FNO_SYMBOLS, 1):
        print(f"  {DIM}{i:>3}.{N} {W}{sym}{N}")
    print(f"\n  {DIM}Total: {len(FNO_SYMBOLS)} F&O stocks{N}")


# ── Main Menu ────────────────────────────────────────────────────────────────

MENU = f"""
{W}  ── SCAN ─────────────────────────────────────────{N}
{C}  1{N}  Intraday SELL signals (short-sell candidates)
{C}  2{N}  BTST BUY signals (buy today, sell tomorrow)
{C}  3{N}  All signals (both directions)

{W}  ── OPTIONS ───────────────────────────────────────{N}
{C}  4{N}  Options — intraday recommendations
{C}  5{N}  Options — BTST recommendations
{C}  6{N}  Option chain viewer

{W}  ── TRADING ───────────────────────────────────────{N}
{C}  7{N}  Place order
{C}  8{N}  View orders

{W}  ── SYSTEM ────────────────────────────────────────{N}
{C}  9{N}  Risk & config
{C}  w{N}  Watchlist (F&O symbols)
{C}  k{N}  Connect to Kite
{C}  0{N}  Exit
"""

ACTIONS = {
    "1": show_intraday_sells,
    "2": show_btst_buys,
    "3": show_all_signals,
    "4": show_options_intraday,
    "5": show_options_btst,
    "6": show_option_chain,
    "7": place_order_interactive,
    "8": show_orders,
    "9": show_risk_config,
    "w": show_watchlist,
    "k": try_connect_kite,
}


def main():
    # Graceful exit on Ctrl+C
    signal.signal(signal.SIGINT, lambda *_: (print(f"\n{Y}Bye!{N}"), sys.exit(0)))

    # Create event loop once
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    clear()
    banner()

    from app.core.config import get_settings
    s = get_settings()
    now = datetime.now(IST)
    print(f"  {W}Mode:{N}  {G if s.trading_mode == 'paper' else R}{s.trading_mode.upper()}{N}")
    print(f"  {W}User:{N}  {s.kite_user_id or 'Not configured'}")
    print(f"  {W}Time:{N}  {now.strftime('%H:%M IST  %d-%b-%Y')}")
    print()

    while True:
        print(MENU)
        choice = input(f"  {C}>{N} ").strip().lower()

        if choice == "0" or choice == "q":
            print(f"\n  {Y}Goodbye! Happy trading.{N}\n")
            break

        action = ACTIONS.get(choice)
        if action:
            try:
                action()
            except KeyboardInterrupt:
                print(f"\n  {Y}Cancelled.{N}")
            except Exception as e:
                print(f"\n  {R}Error: {e}{N}")
        else:
            print(f"  {R}Invalid choice. Try 1-9, w, k, or 0.{N}")

        print()
        input(f"  {DIM}Press Enter to continue...{N}")
        clear()
        banner()


if __name__ == "__main__":
    main()

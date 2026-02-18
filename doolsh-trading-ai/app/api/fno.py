"""F&O trading API — scan stocks, analyse options, get BTST signals."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.core.config import get_settings
from app.core.deps import get_current_user
from app.services.fno_scanner import FNO_SYMBOLS, scan_fno_symbols
from app.services.options_chain import (
    get_option_chain,
    scan_options_opportunities,
    select_best_option,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/fno", tags=["fno"], dependencies=[Depends(get_current_user)])


@router.get("/symbols")
async def list_fno_symbols():
    """Return the list of F&O symbols we scan."""
    return {"symbols": FNO_SYMBOLS, "count": len(FNO_SYMBOLS)}


@router.get("/scan")
async def scan(
    top_n: int = Query(default=10, ge=1, le=40),
    use_live: bool = Query(default=False),
    scan_type: str = Query(default="sell", pattern="^(sell|buy|both)$"),
    symbols: Optional[str] = Query(default=None, description="Comma-separated symbols"),
):
    """Scan F&O symbols for trading signals.

    scan_type: "sell" (intraday short), "buy" (BTST long), "both" (all signals)
    """
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    use_live_data = use_live and settings.trading_mode == "live"
    results = await scan_fno_symbols(
        symbols=symbol_list,
        top_n=top_n,
        use_live_data=use_live_data,
        scan_type=scan_type,
    )

    sells = [r for r in results if "SELL" in r.get("action", "")]
    buys = [r for r in results if "BUY" in r.get("action", "")]

    return {
        "mode": settings.trading_mode,
        "scan_type": scan_type,
        "scanned": len(symbol_list or FNO_SYMBOLS),
        "top_n": top_n,
        "sell_signals": len(sells),
        "buy_signals": len(buys),
        "results": results,
    }


@router.get("/scan/options")
async def scan_options(
    top_n: int = Query(default=10, ge=1, le=20),
    trade_type: str = Query(default="intraday", pattern="^(intraday|btst)$"),
    symbols: Optional[str] = Query(default=None),
):
    """Scan for options trading opportunities.

    Combines equity analysis with options chain to recommend specific strikes.
    """
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    results = await scan_options_opportunities(
        symbols=symbol_list, top_n=top_n, trade_type=trade_type,
    )

    return {
        "mode": settings.trading_mode,
        "trade_type": trade_type,
        "count": len(results),
        "results": results,
    }


@router.get("/options/chain")
async def options_chain(
    symbol: str = Query(..., min_length=1, max_length=20),
    spot_price: float = Query(..., gt=0),
    n_strikes: int = Query(default=5, ge=1, le=15),
):
    """Get option chain for a symbol around a spot price."""
    chain = await get_option_chain(symbol.upper(), spot_price, n_strikes)
    return chain


@router.get("/scan/btst")
async def scan_btst(
    top_n: int = Query(default=10, ge=1, le=20),
    symbols: Optional[str] = Query(default=None),
):
    """Scan for BTST (Buy Today Sell Tomorrow) opportunities.

    Finds stocks with bullish momentum for overnight holds.
    """
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    results = await scan_fno_symbols(
        symbols=symbol_list,
        top_n=top_n,
        use_live_data=False,
        scan_type="buy",
    )

    return {
        "mode": settings.trading_mode,
        "trade_type": "btst",
        "count": len(results),
        "results": results,
    }

"""F&O symbol scanner API — scan NSE F&O stocks for intraday short-sell signals."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from app.core.config import get_settings
from app.core.deps import get_current_user
from app.services.fno_scanner import FNO_SYMBOLS, scan_fno_symbols

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
    symbols: Optional[str] = Query(default=None, description="Comma-separated symbols to scan"),
):
    """Scan F&O symbols and return top short-sell candidates.

    Returns ranked list scored on RSI, MACD, Bollinger Bands, momentum,
    volatility, and trend indicators. Higher score = stronger SELL signal.
    """
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    use_live_data = use_live and settings.trading_mode == "live"
    results = await scan_fno_symbols(
        symbols=symbol_list,
        top_n=top_n,
        use_live_data=use_live_data,
    )

    strong_sells = [r for r in results if r["action"] == "STRONG SELL"]
    sells = [r for r in results if r["action"] == "SELL"]

    return {
        "mode": settings.trading_mode,
        "scanned": len(symbol_list or FNO_SYMBOLS),
        "top_n": top_n,
        "strong_sell_count": len(strong_sells),
        "sell_count": len(sells),
        "results": results,
    }

"""Live trading control endpoints — start/stop auto-trading, ticker, cycles."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import get_settings
from app.core.deps import get_current_user, require_role
from app.core.kite import is_logged_in
from app.models.user import User, UserRole
from app.schemas.trading import AutoTradeConfig
from app.services.auto_trader import (
    disable_auto_trading,
    enable_auto_trading,
    is_auto_trading_enabled,
    is_market_open,
    trading_cycle,
)
from app.services.live_feed import get_all_ticks, is_ticker_running, start_ticker, stop_ticker
from app.services.market_data import _instrument_cache, _load_instruments
from app.services.risk_manager import get_risk_state

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/live", tags=["live"], dependencies=[Depends(get_current_user)])


@router.get("/status")
async def live_status():
    state = get_risk_state()
    return {
        "kite_logged_in": is_logged_in(),
        "market_open": is_market_open(),
        "auto_trading": is_auto_trading_enabled(),
        "ticker_running": is_ticker_running(),
        "trading_mode": settings.trading_mode,
        "open_positions": len(state.open_positions),
        "day_pnl": round(state.realized_pnl, 2),
        "day_trades": state.trade_count,
    }


@router.post("/enable")
async def enable(
    config: AutoTradeConfig,
    user: User = Depends(require_role(UserRole.ADMIN)),
):
    """Enable the automated trading loop."""
    if not is_logged_in():
        raise HTTPException(status_code=400, detail="Login to Kite first")

    model_dir = Path(settings.model_save_dir)
    if config.model_type == "rf":
        check = model_dir / f"rf_{settings.watchlist_symbols[0]}_{config.model_version}.pkl"
    else:
        check = model_dir / f"lstm_{settings.watchlist_symbols[0]}_{config.model_version}.pt"

    enable_auto_trading()
    return {"status": "auto_trading_enabled", "model_type": config.model_type}


@router.post("/disable")
async def disable(user: User = Depends(require_role(UserRole.ADMIN))):
    disable_auto_trading()
    return {"status": "auto_trading_disabled"}


@router.post("/cycle")
async def run_one_cycle(
    model_type: str = "rf",
    model_version: str = "v1",
    user: User = Depends(require_role(UserRole.ADMIN)),
):
    """Manually trigger one trading cycle."""
    model_dir = Path(settings.model_save_dir)
    if model_type == "rf":
        mp = str(model_dir / f"rf_{settings.watchlist_symbols[0]}_{model_version}.pkl")
    else:
        mp = str(model_dir / f"lstm_{settings.watchlist_symbols[0]}_{model_version}.pt")

    result = await trading_cycle(model_path=mp, model_type=model_type)
    return result


@router.post("/ticker/start")
async def start_live_ticker(user: User = Depends(require_role(UserRole.ADMIN))):
    """Start the Kite WebSocket ticker for watchlist symbols."""
    if not is_logged_in():
        raise HTTPException(status_code=400, detail="Login to Kite first")

    await _load_instruments()
    tokens = []
    for sym in settings.watchlist_symbols:
        t = _instrument_cache.get(sym)
        if t:
            tokens.append(t)
    if not tokens:
        raise HTTPException(status_code=400, detail="No instrument tokens found")

    start_ticker(tokens)
    return {"status": "ticker_started", "instruments": len(tokens)}


@router.post("/ticker/stop")
async def stop_live_ticker(user: User = Depends(require_role(UserRole.ADMIN))):
    stop_ticker()
    return {"status": "ticker_stopped"}


@router.get("/ticks")
async def latest_ticks():
    return get_all_ticks()


@router.get("/risk")
async def risk_status():
    state = get_risk_state()
    return {
        "trade_date": str(state.trade_date),
        "realized_pnl": round(state.realized_pnl, 2),
        "trade_count": state.trade_count,
        "open_positions": {
            sym: {**pos, "entry_price": round(pos["entry_price"], 2)}
            for sym, pos in state.open_positions.items()
        },
        "max_daily_loss": settings.max_daily_loss,
        "max_open_positions": settings.max_open_positions,
        "max_trade_count": settings.max_trade_count_per_day,
    }

"""Live trading control endpoints — start/stop auto-trading, ticker, cycles.

Includes AI-powered endpoints for regime detection, risk analytics,
AI feature management, and ensemble model status.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.deps import get_current_user, require_role
from app.core.kite import is_logged_in
from app.models.user import User, UserRole
from app.schemas.trading import AutoTradeConfig
from app.services.auto_trader import (
    disable_auto_trading,
    enable_auto_trading,
    get_ai_features,
    get_current_regime,
    is_auto_trading_enabled,
    is_market_open,
    set_ai_features,
    trading_cycle,
)
from app.services.live_feed import get_all_ticks, is_ticker_running, start_ticker, stop_ticker
from app.services.market_data import _instrument_cache, _load_instruments
from app.services.risk_manager import get_ai_risk_analytics, get_risk_state

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/live", tags=["live"], dependencies=[Depends(get_current_user)])


# ─────────────────────────────────────────────────────────────────────────────
# Status & control
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/status")
async def live_status():
    state = get_risk_state()
    regime = get_current_regime()
    return {
        "kite_logged_in": is_logged_in(),
        "market_open": is_market_open(),
        "auto_trading": is_auto_trading_enabled(),
        "ticker_running": is_ticker_running(),
        "trading_mode": settings.trading_mode,
        "open_positions": len(state.open_positions),
        "day_pnl": round(state.realized_pnl, 2),
        "day_trades": state.trade_count,
        # AI additions
        "ai_features": get_ai_features(),
        "current_regime": regime.get("regime", "unknown") if regime else "not_detected",
        "regime_confidence": regime.get("confidence", 0) if regime else 0,
        "consecutive_losses": state.consecutive_losses,
        "consecutive_wins": state.consecutive_wins,
        "max_drawdown": round(state.max_drawdown, 2),
    }


@router.post("/enable")
async def enable(
    config: AutoTradeConfig,
    user: User = Depends(require_role(UserRole.ADMIN)),
):
    """Enable the AI-powered automated trading loop."""
    if not is_logged_in():
        raise HTTPException(status_code=400, detail="Login to Kite first")

    model_dir = Path(settings.model_save_dir)
    if config.model_type == "rf":
        check = model_dir / f"rf_{settings.watchlist_symbols[0]}_{config.model_version}.pkl"
    elif config.model_type == "transformer":
        check = model_dir / f"transformer_{settings.watchlist_symbols[0]}_{config.model_version}.pt"
    else:
        check = model_dir / f"lstm_{settings.watchlist_symbols[0]}_{config.model_version}.pt"

    enable_auto_trading()
    return {
        "status": "auto_trading_enabled",
        "model_type": config.model_type,
        "ai_features": get_ai_features(),
    }


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
    """Manually trigger one AI-powered trading cycle."""
    model_dir = Path(settings.model_save_dir)
    if model_type == "rf":
        mp = str(model_dir / f"rf_{settings.watchlist_symbols[0]}_{model_version}.pkl")
    elif model_type == "transformer":
        mp = str(model_dir / f"transformer_{settings.watchlist_symbols[0]}_{model_version}.pt")
    else:
        mp = str(model_dir / f"lstm_{settings.watchlist_symbols[0]}_{model_version}.pt")

    result = await trading_cycle(model_path=mp, model_type=model_type)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Ticker
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Risk & analytics
# ─────────────────────────────────────────────────────────────────────────────


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


@router.get("/ai/analytics")
async def ai_risk_analytics():
    """Get AI-computed risk analytics: win rate, drawdown, streaks, profit factor."""
    return get_ai_risk_analytics()


# ─────────────────────────────────────────────────────────────────────────────
# AI Feature management
# ─────────────────────────────────────────────────────────────────────────────


class AIFeatureConfig(BaseModel):
    regime_detection: Optional[bool] = None
    adaptive_ensemble: Optional[bool] = None
    ai_confidence: Optional[bool] = None
    dynamic_sl_tp: Optional[bool] = None
    volatility_sizing: Optional[bool] = None
    signal_quality_filter: Optional[bool] = None
    multi_timeframe: Optional[bool] = None
    tilt_protection: Optional[bool] = None


@router.get("/ai/features")
async def get_ai_feature_status():
    """Get current AI feature toggles."""
    return {
        "ai_features": get_ai_features(),
        "config": {
            "ai_regime_detection": settings.ai_regime_detection,
            "ai_adaptive_ensemble": settings.ai_adaptive_ensemble,
            "ai_transformer_enabled": settings.ai_transformer_enabled,
            "ai_ensemble_mode": settings.ai_ensemble_mode,
            "transformer_d_model": settings.transformer_d_model,
            "transformer_n_heads": settings.transformer_n_heads,
            "transformer_n_layers": settings.transformer_n_layers,
        },
    }


@router.post("/ai/features")
async def update_ai_features(
    config: AIFeatureConfig,
    user: User = Depends(require_role(UserRole.ADMIN)),
):
    """Toggle individual AI features on/off."""
    updates = {k: v for k, v in config.model_dump().items() if v is not None}
    set_ai_features(**updates)
    return {"status": "ai_features_updated", "ai_features": get_ai_features()}


# ─────────────────────────────────────────────────────────────────────────────
# Market regime
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/ai/regime")
async def get_regime():
    """Get the current detected market regime."""
    regime = get_current_regime()
    if not regime:
        return {"status": "not_detected", "message": "Run a trading cycle to detect regime"}
    return regime


@router.post("/ai/regime/detect")
async def detect_regime_now(
    symbol: str = "RELIANCE",
    user: User = Depends(require_role(UserRole.ADMIN)),
):
    """Manually trigger regime detection for a specific symbol."""
    from app.services.regime_detector import detect_regime

    use_live = is_logged_in()
    try:
        if use_live:
            from app.services.market_data import fetch_daily_prices
            df = await fetch_daily_prices(symbol, days=300, use_cache=True)
        else:
            from app.services.fno_scanner import _generate_synthetic_ohlcv
            df = _generate_synthetic_ohlcv(symbol)

        result = detect_regime(df, symbol=symbol)
        regime_val = result.get("regime", "unknown")
        if hasattr(regime_val, "value"):
            result["regime"] = regime_val.value
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regime detection failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble model status
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/ai/ensemble")
async def ensemble_status():
    """Get adaptive ensemble model weights and performance tracking."""
    try:
        from ml.training.adaptive_ensemble import get_performance_tracker, compute_dynamic_weights
        tracker = get_performance_tracker()
        accuracies = tracker.get_all_accuracies()

        available = list(accuracies.keys()) if accuracies else ["rf"]
        regime = get_current_regime()
        regime_str = regime.get("regime", "mean_reverting") if regime else "mean_reverting"
        if hasattr(regime_str, "value"):
            regime_str = regime_str.value

        weights = compute_dynamic_weights(available, regime=regime_str) if available else {}

        return {
            "model_accuracies": accuracies,
            "dynamic_weights": weights,
            "current_regime": regime_str,
            "available_models": available,
        }
    except ImportError:
        return {"status": "ensemble_not_available", "reason": "adaptive_ensemble module not loaded"}

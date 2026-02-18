"""KiteAI — FastAPI REST backend for Zerodha AI Auto-Trading.

Endpoints for trading, signals, models, backtesting, risk, and Kite auth.
Swagger UI: http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.config import load_config

logger = logging.getLogger(__name__)

# Scheduler reference
_scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB and scheduler on startup."""
    from core.database import get_engine
    get_engine()
    logger.info("KiteAI API started")
    yield
    if _scheduler:
        _scheduler.shutdown(wait=False)
    logger.info("KiteAI API shutdown")


app = FastAPI(
    title="KiteAI",
    description="Personal AI Auto-Trading Addon for Zerodha Kite",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ────────────────────────────────────────────────

class LoginRequest(BaseModel):
    method: str = "auto"  # auto, token
    token: str = ""


class SignalRequest(BaseModel):
    symbol: str = "RELIANCE"
    model_type: str = "random_forest"
    days: int = 180


class TrainRequest(BaseModel):
    symbol: str = "RELIANCE"
    model_type: str = "random_forest"
    days: int = 180


class BacktestRequest(BaseModel):
    symbol: str = "RELIANCE"
    model_type: str = "random_forest"
    days: int = 365
    initial_capital: float = 100000


class TradeRequest(BaseModel):
    symbol: str
    side: str  # BUY or SELL
    quantity: int = 1
    order_type: str = "MARKET"
    price: float = 0.0
    product: str = "MIS"
    exchange: str = "NSE"


# ─── System Endpoints ───────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """System health and overview."""
    from core.kite_auth import is_logged_in
    from core.risk_engine import get_risk_analytics, get_risk_state
    from core.auto_trader import is_auto_trading_enabled, get_cycle_count, get_current_regime

    cfg = load_config()
    state = get_risk_state()
    return {
        "status": "running",
        "mode": cfg.mode,
        "kite_connected": is_logged_in(),
        "auto_trading": is_auto_trading_enabled(),
        "cycle_count": get_cycle_count(),
        "regime": get_current_regime(),
        "risk": {
            "realized_pnl": state.realized_pnl,
            "trade_count": state.trade_count,
            "open_positions": len(state.open_positions),
            "kill_switch": cfg.kill_switch,
        },
        "analytics": get_risk_analytics(),
    }


@app.get("/api/config")
async def get_config():
    """Current configuration (no secrets)."""
    return load_config().to_dict()


# ─── Kite Auth Endpoints ────────────────────────────────────────────────────

@app.get("/api/kite/status")
async def kite_status():
    from core.kite_auth import is_logged_in, get_login_url
    return {
        "logged_in": is_logged_in(),
        "login_url": get_login_url(),
    }


@app.post("/api/kite/login")
async def kite_login(req: LoginRequest):
    """Login to Zerodha Kite."""
    from core.kite_auth import auto_login, token_login
    try:
        if req.method == "auto":
            token = await auto_login()
        elif req.method == "token" and req.token:
            token = await token_login(req.token)
        else:
            raise HTTPException(400, "Invalid login method")
        return {"status": "ok", "message": "Logged in successfully"}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/kite/margins")
async def kite_margins():
    from core.kite_data import get_margins
    return await get_margins()


@app.get("/api/kite/positions")
async def kite_positions():
    from core.kite_data import get_positions
    return await get_positions()


@app.get("/api/kite/holdings")
async def kite_holdings():
    from core.kite_data import get_holdings
    return await get_holdings()


# ─── Data Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/data/{symbol}")
async def get_data(symbol: str, days: int = 180):
    """Fetch historical OHLCV data."""
    from core.kite_data import fetch_historical
    df = await fetch_historical(symbol, days=days)
    return df.tail(100).to_dict(orient="records")


@app.get("/api/ltp")
async def get_ltp(symbols: str = "RELIANCE,TCS,INFY"):
    """Get last traded prices."""
    from core.kite_data import get_ltp as _get_ltp
    sym_list = [s.strip() for s in symbols.split(",")]
    return await _get_ltp(sym_list)


# ─── Scanner Endpoints ──────────────────────────────────────────────────────

@app.get("/api/scan")
async def scan(scan_type: str = "all", top_n: int = 10):
    """Scan watchlist for trading opportunities."""
    from core.scanner import scan_stocks
    return await scan_stocks(scan_type=scan_type, top_n=top_n)


@app.get("/api/scan/fno")
async def scan_fno(top_n: int = 5):
    """Scan for F&O opportunities."""
    from core.scanner import scan_fno_opportunities
    return await scan_fno_opportunities(top_n=top_n)


# ─── Signal Endpoints ───────────────────────────────────────────────────────

@app.post("/api/signal")
async def generate_signal(req: SignalRequest):
    """Generate AI trading signal for a symbol."""
    from core.kite_data import fetch_historical
    from core.feature_engine import build_features
    from core.model_engine import get_latest_model
    from core.signal_engine import generate_signal as _gen_signal

    df = await fetch_historical(req.symbol, days=req.days)
    model_path = get_latest_model(req.symbol, req.model_type)
    if not model_path:
        # Train a model first
        from core.model_engine import train_model
        featured = build_features(df)
        result = train_model(featured, req.symbol, req.model_type)
        model_path = result["file_path"]

    signal = _gen_signal(df, model_path, req.model_type)
    return {"symbol": req.symbol, **signal}


@app.get("/api/signals/history")
async def signal_history(limit: int = 50):
    from core.database import get_signals
    return get_signals(limit=limit)


# ─── Model Endpoints ────────────────────────────────────────────────────────

@app.post("/api/train")
async def train_model(req: TrainRequest):
    """Train an ML model on historical data."""
    from core.kite_data import fetch_historical
    from core.feature_engine import build_features
    from core.model_engine import train_model as _train
    from core.database import save_model_record

    df = await fetch_historical(req.symbol, days=req.days)
    featured = build_features(df)
    result = _train(featured, req.symbol, req.model_type)
    save_model_record(result)
    return result


@app.get("/api/models")
async def list_models():
    from core.model_engine import list_models as _list
    return _list()


# ─── Trading Endpoints ──────────────────────────────────────────────────────

@app.post("/api/trade")
async def execute_trade(req: TradeRequest):
    """Place a manual trade."""
    from core.kite_orders import place_order
    from core.database import save_trade

    result = await place_order(
        symbol=req.symbol, side=req.side, quantity=req.quantity,
        order_type=req.order_type, price=req.price,
        product=req.product, exchange=req.exchange,
    )
    save_trade(result)
    return result


@app.get("/api/trades")
async def get_trades(limit: int = 50, symbol: str = ""):
    from core.database import get_trades as _get
    return _get(limit=limit, symbol=symbol)


@app.get("/api/orders")
async def get_orders():
    from core.kite_orders import get_orders as _get
    return await _get()


# ─── Backtest Endpoints ─────────────────────────────────────────────────────

@app.post("/api/backtest")
async def backtest(req: BacktestRequest):
    """Run a backtest on historical data."""
    from core.kite_data import fetch_historical
    from core.feature_engine import build_features
    from core.model_engine import get_latest_model, train_model as _train
    from core.backtester import run_backtest

    df = await fetch_historical(req.symbol, days=req.days)
    model_path = get_latest_model(req.symbol, req.model_type)
    if not model_path:
        featured = build_features(df)
        result = _train(featured, req.symbol, req.model_type)
        model_path = result["file_path"]

    return run_backtest(df, model_path, req.model_type, req.initial_capital)


# ─── Risk Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/risk")
async def get_risk():
    from core.risk_engine import get_risk_analytics, get_risk_state
    state = get_risk_state()
    return {
        "analytics": get_risk_analytics(),
        "open_positions": {
            k: v for k, v in state.open_positions.items()
        },
    }


@app.post("/api/risk/kill-switch")
async def toggle_kill_switch(enable: bool = True):
    """Toggle the emergency kill switch."""
    import os
    os.environ["KILL_SWITCH"] = "true" if enable else "false"
    return {"kill_switch": enable}


# ─── Auto-Trading Endpoints ─────────────────────────────────────────────────

@app.post("/api/auto/start")
async def start_auto_trading():
    """Start the auto-trading scheduler."""
    global _scheduler
    from core.auto_trader import enable_auto_trading, trading_cycle

    enable_auto_trading()
    cfg = load_config()

    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        if _scheduler is None:
            _scheduler = AsyncIOScheduler()
            _scheduler.add_job(
                trading_cycle,
                "interval",
                seconds=cfg.scan_interval,
                id="trading_cycle",
                replace_existing=True,
            )
            _scheduler.start()
    except ImportError:
        # Fallback: run single cycle
        asyncio.create_task(trading_cycle())

    return {"status": "auto-trading started", "interval": cfg.scan_interval}


@app.post("/api/auto/stop")
async def stop_auto_trading():
    global _scheduler
    from core.auto_trader import disable_auto_trading
    disable_auto_trading()
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
    return {"status": "auto-trading stopped"}


@app.post("/api/auto/cycle")
async def run_single_cycle():
    """Run one auto-trading cycle manually."""
    from core.auto_trader import trading_cycle
    return await trading_cycle()


@app.get("/api/auto/log")
async def get_auto_log():
    from core.auto_trader import get_trade_log, get_cycle_count
    return {"cycles": get_cycle_count(), "log": get_trade_log()}


# ─── Daily P&L ──────────────────────────────────────────────────────────────

@app.get("/api/pnl/daily")
async def daily_pnl(days: int = 30):
    from core.database import get_daily_pnl
    return get_daily_pnl(days)


# ─── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    logging.basicConfig(level=cfg.log_level)
    uvicorn.run(app, host=cfg.api_host, port=cfg.api_port)

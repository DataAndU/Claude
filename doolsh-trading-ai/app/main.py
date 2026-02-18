"""FastAPI application entry point — Zerodha Kite automated trading."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import admin, auth, kite, live, orders, strategies, trading
from app.core.config import get_settings
from app.core.database import init_db
from app.core.logging import setup_logging
from app.core.redis import close_redis
from app.services.scheduler import start_scheduler, stop_scheduler

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    await init_db()
    start_scheduler()
    yield
    stop_scheduler()
    await close_redis()


app = FastAPI(
    title="Doolsh Trading AI",
    description=(
        "AI-powered automated trading engine for Zerodha Kite. "
        "Runs on Userland / Android. "
        "For educational purposes only. Not financial advice."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PREFIX = f"/api/{settings.api_version}"
app.include_router(auth.router, prefix=PREFIX)
app.include_router(kite.router, prefix=PREFIX)
app.include_router(trading.router, prefix=PREFIX)
app.include_router(orders.router, prefix=PREFIX)
app.include_router(live.router, prefix=PREFIX)
app.include_router(strategies.router, prefix=PREFIX)
app.include_router(admin.router, prefix=PREFIX)


@app.get("/health")
async def health():
    from app.core.kite import is_logged_in
    from app.services.auto_trader import is_auto_trading_enabled, is_market_open
    return {
        "status": "ok",
        "app": settings.app_name,
        "mode": settings.trading_mode,
        "kite_connected": is_logged_in(),
        "market_open": is_market_open(),
        "auto_trading": is_auto_trading_enabled(),
    }

"""FastAPI application entry point — Zerodha Kite automated trading."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import admin, auth, kite, live, orders, strategies, trading
from app.api import fno
from app.core.config import get_settings
from app.core.database import init_db
from app.core.logging import setup_logging
from app.core.redis import close_redis
from app.services.scheduler import start_scheduler, stop_scheduler

settings = get_settings()

STATIC_DIR = Path(__file__).parent / "static"


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
    version="2.1.0",
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
app.include_router(fno.router, prefix=PREFIX)

# Serve static files (CSS/JS if we ever split them out)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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


@app.get("/")
async def dashboard():
    """Serve the main HTML dashboard."""
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return FileResponse(str(html_file), media_type="text/html")
    return {"message": "Dashboard not found. Place index.html in app/static/"}

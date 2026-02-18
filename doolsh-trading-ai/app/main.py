"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import admin, auth, strategies, trading
from app.core.config import get_settings
from app.core.database import init_db
from app.core.logging import setup_logging
from app.core.redis import close_redis

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    await init_db()
    yield
    await close_redis()


app = FastAPI(
    title="Doolsh Trading AI",
    description=(
        "AI-powered market analysis engine for signal generation, "
        "backtesting, and strategy comparison. "
        "For educational purposes only. Not financial advice."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
PREFIX = f"/api/{settings.api_version}"
app.include_router(auth.router, prefix=PREFIX)
app.include_router(trading.router, prefix=PREFIX)
app.include_router(strategies.router, prefix=PREFIX)
app.include_router(admin.router, prefix=PREFIX)


@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}

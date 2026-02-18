"""Application configuration for Zerodha Kite + Userland environment.

Uses SQLite (no external DB), in-memory caching (no Redis),
and APScheduler (no Celery). Designed to run on minimal ARM hardware.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ---- Application ----
    app_name: str = "doolsh-trading-ai"
    app_env: str = "development"
    debug: bool = False
    secret_key: str = "change-me-to-a-random-64-char-string"
    api_version: str = "v1"
    base_dir: str = str(Path(__file__).resolve().parent.parent.parent)

    # ---- Database (SQLite — zero config) ----
    db_path: str = "data/doolsh.db"

    @property
    def database_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.base_dir}/{self.db_path}"

    @property
    def database_url_sync(self) -> str:
        return f"sqlite:///{self.base_dir}/{self.db_path}"

    # ---- Zerodha Kite Connect ----
    kite_api_key: str = ""
    kite_api_secret: str = ""
    kite_user_id: str = ""
    kite_password: str = ""
    kite_totp_secret: str = ""       # Base32 TOTP secret for auto-login
    kite_access_token: str = ""       # Populated at runtime after login
    kite_request_token: str = ""      # Populated during OAuth callback

    # ---- Trading ----
    trading_mode: str = "paper"       # "paper" or "live"
    trading_exchange: str = "NSE"
    trading_product: str = "MIS"      # MIS (intraday) / CNC (delivery) / NRML
    default_quantity: int = 1
    watchlist: str = '["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","SBIN","BAJFINANCE","ITC","HINDUNILVR","KOTAKBANK"]'
    market_open_hour: int = 9
    market_open_minute: int = 15
    market_close_hour: int = 15
    market_close_minute: int = 15
    auto_square_off_minute: int = 10  # minutes before close

    @property
    def watchlist_symbols(self) -> List[str]:
        return json.loads(self.watchlist)

    # ---- Risk Management ----
    max_daily_loss: float = 5000.0           # INR
    max_position_value: float = 100000.0     # INR per position
    max_open_positions: int = 5
    stop_loss_pct: float = 0.02              # 2 %
    take_profit_pct: float = 0.04            # 4 %
    max_trade_count_per_day: int = 20
    min_confidence_threshold: float = 0.65

    # ---- JWT ----
    jwt_secret_key: str = "change-me-jwt-secret"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 720  # 12 h for bot sessions

    # ---- ML ----
    model_save_dir: str = "ml/saved_models"
    default_train_test_split: float = 0.8
    cross_validation_folds: int = 5

    # ---- Scheduler ----
    scheduler_interval_seconds: int = 60  # main loop tick

    # ---- Logging ----
    log_level: str = "INFO"
    log_format: str = "text"  # text is friendlier on Userland terminal

    # ---- CORS ----
    cors_origins: str = '["http://localhost:3000","http://localhost:8000"]'

    @property
    def cors_origin_list(self) -> List[str]:
        return json.loads(self.cors_origins)


@lru_cache
def get_settings() -> Settings:
    return Settings()

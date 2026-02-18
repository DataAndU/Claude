"""Application configuration for Zerodha Kite + Userland environment.

Supports equity + options trading, intraday (MIS) + BTST (NRML).
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
    kite_totp_secret: str = ""
    kite_access_token: str = ""
    kite_request_token: str = ""

    # ---- Trading ----
    trading_mode: str = "paper"       # "paper" or "live"
    trading_exchange: str = "NSE"     # NSE for equity, NFO for options
    trading_product: str = "MIS"      # MIS (intraday) / NRML (BTST/options)
    default_quantity: int = 1
    watchlist: str = '["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","SBIN","BAJFINANCE","ITC","HINDUNILVR","KOTAKBANK","TATAMOTORS","MARUTI","AXISBANK","LT","SUNPHARMA","TITAN","ADANIENT","BHARTIARTL","WIPRO","HCLTECH"]'
    market_open_hour: int = 9
    market_open_minute: int = 15
    market_close_hour: int = 15
    market_close_minute: int = 15
    auto_square_off_minute: int = 10

    # ---- Options Trading ----
    options_enabled: bool = True
    options_lot_size: int = 1         # Number of lots
    options_strike_offset: int = 0    # 0 = ATM, +1 = 1 OTM, -1 = 1 ITM
    options_expiry_preference: str = "weekly"  # "weekly" or "monthly"

    # ---- BTST (Buy Today Sell Tomorrow) ----
    btst_enabled: bool = True
    btst_product: str = "NRML"       # NRML for carry-forward
    btst_max_positions: int = 3
    btst_target_pct: float = 0.03    # 3% target for BTST
    btst_stoploss_pct: float = 0.015 # 1.5% stoploss for BTST

    @property
    def watchlist_symbols(self) -> List[str]:
        return json.loads(self.watchlist)

    # ---- Risk Management ----
    max_daily_loss: float = 5000.0
    max_position_value: float = 100000.0
    max_open_positions: int = 5
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_trade_count_per_day: int = 20
    min_confidence_threshold: float = 0.60

    # ---- JWT ----
    jwt_secret_key: str = "change-me-jwt-secret"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 1440

    # ---- ML ----
    model_save_dir: str = "ml/saved_models"
    default_train_test_split: float = 0.8
    cross_validation_folds: int = 5

    # ---- AI Features ----
    ai_regime_detection: bool = True        # Auto-detect market regime each cycle
    ai_adaptive_ensemble: bool = True       # Use dynamic model weighting
    ai_confidence_scoring: bool = True      # Multi-indicator AI confidence
    ai_dynamic_sl_tp: bool = True           # ATR/regime-based SL/TP
    ai_volatility_sizing: bool = True       # Volatility-adjusted position sizing
    ai_signal_quality_filter: bool = True   # Filter out D-quality signals
    ai_multi_timeframe: bool = True         # Multi-timeframe signal alignment
    ai_tilt_protection: bool = True         # Pause after consecutive losses
    ai_transformer_enabled: bool = True     # Enable Transformer attention model
    ai_ensemble_mode: str = "adaptive"      # "adaptive", "simple", "single"

    # Transformer model hyperparameters
    transformer_d_model: int = 64
    transformer_n_heads: int = 4
    transformer_n_layers: int = 3
    transformer_seq_len: int = 30
    transformer_dropout: float = 0.2

    # ---- Scheduler ----
    scheduler_interval_seconds: int = 60

    # ---- Logging ----
    log_level: str = "INFO"
    log_format: str = "text"

    # ---- CORS ----
    cors_origins: str = '["http://localhost:3000","http://localhost:8000","*"]'

    @property
    def cors_origin_list(self) -> List[str]:
        return json.loads(self.cors_origins)


@lru_cache
def get_settings() -> Settings:
    return Settings()

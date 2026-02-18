"""Application configuration loaded from environment variables."""

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

    # ---- Database ----
    postgres_host: str = "db"
    postgres_port: int = 5432
    postgres_user: str = "doolsh"
    postgres_password: str = "change-me"
    postgres_db: str = "doolsh_trading"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ---- Redis ----
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ---- Celery ----
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"

    # ---- JWT ----
    jwt_secret_key: str = "change-me-jwt-secret"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # ---- Market Data ----
    alpha_vantage_api_key: str = "demo"
    data_cache_ttl_seconds: int = 300

    # ---- ML ----
    model_save_dir: str = "ml/saved_models"
    default_train_test_split: float = 0.8
    cross_validation_folds: int = 5

    # ---- Rate Limiting ----
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000

    # ---- Logging ----
    log_level: str = "INFO"
    log_format: str = "json"

    # ---- CORS ----
    cors_origins: str = '["http://localhost:3000","http://localhost:8000"]'

    @property
    def cors_origin_list(self) -> List[str]:
        return json.loads(self.cors_origins)


@lru_cache
def get_settings() -> Settings:
    return Settings()

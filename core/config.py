"""Configuration loader — reads config.yaml and environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base recursively."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class Config:
    """Typed access to config.yaml values with env-var overrides."""

    def __init__(self, data: dict):
        self._data = data

    def _get(self, *keys, default=None):
        val = self._data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val

    # ── System ──
    @property
    def mode(self) -> str:
        return os.getenv("TRADING_MODE", self._get("system", "mode", default="paper"))

    @property
    def log_level(self) -> str:
        return self._get("system", "log_level", default="INFO")

    @property
    def db_path(self) -> str:
        return self._get("system", "db_path", default="database/trading.db")

    # ── Trading ──
    @property
    def exchange(self) -> str:
        return self._get("trading", "exchange", default="binance")

    @property
    def symbols(self) -> List[str]:
        return self._get("trading", "symbols", default=["BTC/USDT"])

    @property
    def primary_timeframe(self) -> str:
        return self._get("trading", "timeframes", "primary", default="1h")

    @property
    def confirmation_timeframes(self) -> List[str]:
        return self._get("trading", "timeframes", "confirmation", default=["4h"])

    @property
    def default_capital(self) -> float:
        return float(self._get("trading", "default_capital", default=10000.0))

    # ── Risk ──
    @property
    def max_position_pct(self) -> float:
        return float(self._get("risk", "max_position_pct", default=0.02))

    @property
    def atr_sl_multiplier(self) -> float:
        return float(self._get("risk", "atr_sl_multiplier", default=2.0))

    @property
    def atr_tp_multiplier(self) -> float:
        return float(self._get("risk", "atr_tp_multiplier", default=3.0))

    @property
    def max_daily_loss_pct(self) -> float:
        return float(self._get("risk", "max_daily_loss_pct", default=0.05))

    @property
    def max_drawdown_pct(self) -> float:
        return float(self._get("risk", "max_drawdown_pct", default=0.15))

    @property
    def cooldown_minutes(self) -> int:
        return int(self._get("risk", "cooldown_minutes", default=30))

    @property
    def max_open_trades(self) -> int:
        return int(self._get("risk", "max_open_trades", default=3))

    @property
    def kill_switch(self) -> bool:
        ks = os.getenv("KILL_SWITCH", "").lower()
        if ks in ("true", "1", "yes"):
            return True
        return bool(self._get("risk", "kill_switch", default=False))

    # ── Model ──
    @property
    def primary_model(self) -> str:
        return self._get("model", "primary", default="random_forest")

    @property
    def fallback_model(self) -> str:
        return self._get("model", "fallback", default="xgboost")

    @property
    def retrain_hours(self) -> int:
        return int(self._get("model", "retrain_hours", default=6))

    @property
    def lookback_days(self) -> int:
        return int(self._get("model", "lookback_days", default=90))

    @property
    def prediction_horizon(self) -> int:
        return int(self._get("model", "prediction_horizon", default=5))

    @property
    def min_confidence(self) -> float:
        return float(self._get("model", "min_confidence", default=0.60))

    @property
    def rf_params(self) -> Dict[str, Any]:
        return self._get("model", "random_forest", default={"n_estimators": 100, "max_depth": 10})

    @property
    def xgb_params(self) -> Dict[str, Any]:
        return self._get("model", "xgboost", default={"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1})

    @property
    def lstm_enabled(self) -> bool:
        return bool(self._get("model", "lstm", "enabled", default=False))

    @property
    def lstm_params(self) -> Dict[str, Any]:
        return self._get("model", "lstm", default={"seq_len": 20, "hidden_size": 64, "epochs": 30, "batch_size": 32})

    # ── Backtest ──
    @property
    def backtest_capital(self) -> float:
        return float(self._get("backtest", "initial_capital", default=10000.0))

    @property
    def commission_pct(self) -> float:
        return float(self._get("backtest", "commission_pct", default=0.001))

    @property
    def slippage_pct(self) -> float:
        return float(self._get("backtest", "slippage_pct", default=0.0005))

    # ── API ──
    @property
    def api_host(self) -> str:
        return self._get("api", "host", default="0.0.0.0")

    @property
    def api_port(self) -> int:
        return int(self._get("api", "port", default=8000))

    @property
    def dashboard_port(self) -> int:
        return int(self._get("dashboard", "port", default=8501))


def _validate_config(cfg: Config) -> None:
    """Warn about dangerous configuration at startup."""
    import logging
    _logger = logging.getLogger(__name__)

    if cfg.mode == "live":
        _logger.warning(
            "*** LIVE TRADING MODE ACTIVE *** Real money will be used for orders. "
            "Set TRADING_MODE=paper to disable."
        )
        if not os.getenv("EXCHANGE_API_KEY"):
            _logger.error(
                "CRITICAL: TRADING_MODE is 'live' but EXCHANGE_API_KEY is not set. "
                "Live trading will fail."
            )

    # Validate risk parameter bounds
    if not (0 < cfg.max_position_pct < 0.5):
        _logger.warning(
            "Risk: max_position_pct=%.4f is outside safe bounds (0, 0.5). "
            "This could lead to excessive position sizes.",
            cfg.max_position_pct,
        )
    if not (0 < cfg.max_daily_loss_pct < 1.0):
        _logger.warning(
            "Risk: max_daily_loss_pct=%.4f is outside safe bounds (0, 1.0).",
            cfg.max_daily_loss_pct,
        )
    if cfg.atr_sl_multiplier <= 0:
        _logger.warning("Risk: atr_sl_multiplier=%.2f must be > 0.", cfg.atr_sl_multiplier)


@lru_cache
def load_config(path: str = "config/config.yaml") -> Config:
    """Load and cache configuration."""
    cfg_path = Path(path)
    if cfg_path.exists():
        with open(cfg_path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    cfg = Config(data)
    _validate_config(cfg)
    return cfg

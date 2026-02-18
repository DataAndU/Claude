"""Configuration loader for KiteAI.

Reads config/config.yaml for trading parameters and .env for secrets.
Environment variables always override config file values.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class Config:
    """Typed access to config values with env-var overrides."""

    def __init__(self, data: dict):
        self._data = data
        self.base_dir = Path(__file__).resolve().parent.parent

    def _get(self, *keys, default=None):
        val = self._data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val

    # ── System ──────────────────────────────────────────────────────────────
    @property
    def mode(self) -> str:
        return os.getenv("TRADING_MODE", self._get("system", "mode", default="paper"))

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    @property
    def log_level(self) -> str:
        return os.getenv("LOG_LEVEL", self._get("system", "log_level", default="INFO"))

    @property
    def db_path(self) -> str:
        return str(self.base_dir / self._get("system", "db_path", default="data/kiteai.db"))

    @property
    def timezone(self) -> str:
        return self._get("system", "timezone", default="Asia/Kolkata")

    # ── Zerodha Kite Credentials ────────────────────────────────────────────
    @property
    def kite_api_key(self) -> str:
        return os.getenv("KITE_API_KEY", "")

    @property
    def kite_api_secret(self) -> str:
        return os.getenv("KITE_API_SECRET", "")

    @property
    def kite_user_id(self) -> str:
        return os.getenv("KITE_USER_ID", "")

    @property
    def kite_password(self) -> str:
        return os.getenv("KITE_PASSWORD", "")

    @property
    def kite_totp_secret(self) -> str:
        return os.getenv("KITE_TOTP_SECRET", "")

    # ── Trading ─────────────────────────────────────────────────────────────
    @property
    def exchange(self) -> str:
        return self._get("trading", "exchange", default="NSE")

    @property
    def watchlist(self) -> List[str]:
        return self._get("trading", "watchlist", default=[
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        ])

    @property
    def intraday_enabled(self) -> bool:
        return bool(self._get("trading", "intraday", "enabled", default=True))

    @property
    def intraday_product(self) -> str:
        return self._get("trading", "intraday", "product", default="MIS")

    @property
    def intraday_max_trades(self) -> int:
        return int(self._get("trading", "intraday", "max_trades", default=5))

    @property
    def btst_enabled(self) -> bool:
        return bool(self._get("trading", "btst", "enabled", default=True))

    @property
    def btst_product(self) -> str:
        return self._get("trading", "btst", "product", default="NRML")

    @property
    def btst_max_positions(self) -> int:
        return int(self._get("trading", "btst", "max_positions", default=3))

    @property
    def btst_target_pct(self) -> float:
        return float(self._get("trading", "btst", "target_pct", default=0.03))

    @property
    def btst_stoploss_pct(self) -> float:
        return float(self._get("trading", "btst", "stoploss_pct", default=0.015))

    @property
    def fno_enabled(self) -> bool:
        return bool(self._get("trading", "fno", "enabled", default=True))

    @property
    def fno_exchange(self) -> str:
        return self._get("trading", "fno", "exchange", default="NFO")

    @property
    def fno_lot_size(self) -> int:
        return int(self._get("trading", "fno", "lot_size", default=1))

    @property
    def market_open(self) -> str:
        return self._get("trading", "market_open", default="09:15")

    @property
    def market_close(self) -> str:
        return self._get("trading", "market_close", default="15:15")

    @property
    def square_off_minutes(self) -> int:
        return int(self._get("trading", "square_off_minutes_before", default=10))

    # ── Risk ────────────────────────────────────────────────────────────────
    @property
    def max_daily_loss(self) -> float:
        return float(self._get("risk", "max_daily_loss", default=5000))

    @property
    def max_position_value(self) -> float:
        return float(self._get("risk", "max_position_value", default=100000))

    @property
    def max_open_positions(self) -> int:
        return int(self._get("risk", "max_open_positions", default=5))

    @property
    def stop_loss_pct(self) -> float:
        return float(self._get("risk", "stop_loss_pct", default=0.02))

    @property
    def take_profit_pct(self) -> float:
        return float(self._get("risk", "take_profit_pct", default=0.04))

    @property
    def max_trades_per_day(self) -> int:
        return int(self._get("risk", "max_trades_per_day", default=20))

    @property
    def min_confidence(self) -> float:
        return float(self._get("risk", "min_confidence", default=0.60))

    @property
    def kill_switch(self) -> bool:
        ks = os.getenv("KILL_SWITCH", "").lower()
        if ks in ("true", "1", "yes"):
            return True
        return bool(self._get("risk", "kill_switch", default=False))

    @property
    def cooldown_minutes(self) -> int:
        return int(self._get("risk", "cooldown_minutes", default=5))

    # ── AI / ML ─────────────────────────────────────────────────────────────
    @property
    def primary_model(self) -> str:
        return self._get("ai", "primary_model", default="random_forest")

    @property
    def fallback_model(self) -> str:
        return self._get("ai", "fallback_model", default="xgboost")

    @property
    def ensemble_mode(self) -> str:
        return self._get("ai", "ensemble_mode", default="adaptive")

    @property
    def retrain_hours(self) -> int:
        return int(self._get("ai", "retrain_hours", default=24))

    @property
    def lookback_days(self) -> int:
        return int(self._get("ai", "lookback_days", default=180))

    @property
    def ai_min_confidence(self) -> float:
        return float(self._get("ai", "min_confidence", default=0.60))

    @property
    def prediction_horizon(self) -> int:
        return int(self._get("ai", "prediction_horizon", default=5))

    @property
    def regime_detection(self) -> bool:
        return bool(self._get("ai", "regime_detection", default=True))

    @property
    def adaptive_ensemble(self) -> bool:
        return bool(self._get("ai", "adaptive_ensemble", default=True))

    @property
    def multi_timeframe(self) -> bool:
        return bool(self._get("ai", "multi_timeframe", default=True))

    @property
    def dynamic_sl_tp(self) -> bool:
        return bool(self._get("ai", "dynamic_sl_tp", default=True))

    @property
    def volatility_sizing(self) -> bool:
        return bool(self._get("ai", "volatility_sizing", default=True))

    @property
    def signal_quality_filter(self) -> bool:
        return bool(self._get("ai", "signal_quality_filter", default=True))

    @property
    def tilt_protection(self) -> bool:
        return bool(self._get("ai", "tilt_protection", default=True))

    @property
    def rf_params(self) -> Dict[str, Any]:
        return self._get("ai", "random_forest", default={"n_estimators": 200, "max_depth": 12})

    @property
    def xgb_params(self) -> Dict[str, Any]:
        return self._get("ai", "xgboost", default={
            "n_estimators": 150, "max_depth": 8, "learning_rate": 0.05,
        })

    @property
    def lstm_enabled(self) -> bool:
        return bool(self._get("ai", "lstm", "enabled", default=False))

    @property
    def lstm_params(self) -> Dict[str, Any]:
        return self._get("ai", "lstm", default={
            "seq_len": 30, "hidden_size": 64, "epochs": 50,
        })

    @property
    def transformer_enabled(self) -> bool:
        return bool(self._get("ai", "transformer", "enabled", default=True))

    @property
    def transformer_params(self) -> Dict[str, Any]:
        return self._get("ai", "transformer", default={
            "d_model": 64, "n_heads": 4, "n_layers": 3,
            "seq_len": 30, "dropout": 0.2,
        })

    # ── Backtest ────────────────────────────────────────────────────────────
    @property
    def backtest_capital(self) -> float:
        return float(self._get("backtest", "initial_capital", default=100000))

    @property
    def commission_pct(self) -> float:
        return float(self._get("backtest", "commission_pct", default=0.0003))

    @property
    def slippage_pct(self) -> float:
        return float(self._get("backtest", "slippage_pct", default=0.001))

    @property
    def stt_pct(self) -> float:
        return float(self._get("backtest", "stt_pct", default=0.001))

    # ── Scheduler ───────────────────────────────────────────────────────────
    @property
    def scan_interval(self) -> int:
        return int(self._get("scheduler", "scan_interval_seconds", default=60))

    # ── API ──────────────────────────────────────────────────────────────────
    @property
    def api_host(self) -> str:
        return self._get("api", "host", default="0.0.0.0")

    @property
    def api_port(self) -> int:
        return int(self._get("api", "port", default=8000))

    @property
    def dashboard_port(self) -> int:
        return int(self._get("dashboard", "port", default=8501))

    def to_dict(self) -> dict:
        """Return safe (no secrets) config summary."""
        return {
            "mode": self.mode,
            "exchange": self.exchange,
            "watchlist": self.watchlist,
            "intraday_enabled": self.intraday_enabled,
            "btst_enabled": self.btst_enabled,
            "fno_enabled": self.fno_enabled,
            "max_daily_loss": self.max_daily_loss,
            "max_position_value": self.max_position_value,
            "max_open_positions": self.max_open_positions,
            "primary_model": self.primary_model,
            "ensemble_mode": self.ensemble_mode,
            "kite_connected": bool(self.kite_api_key),
        }


@lru_cache
def load_config(path: str = "config/config.yaml") -> Config:
    cfg_path = Path(path)
    if cfg_path.exists():
        with open(cfg_path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    return Config(data)

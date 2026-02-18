"""Pydantic schemas for trading, signal, backtest, and ML endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---- Training ----
class TrainRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20, examples=["AAPL"])
    model_type: str = Field("rf", pattern="^(rf|lstm)$")
    horizon: int = Field(5, ge=1, le=60)
    version: str = "v1"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class TrainResponse(BaseModel):
    model_type: str
    symbol: str
    version: str
    file_path: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    hyperparameters: Dict[str, Any]
    extra: Dict[str, Any] = Field(default_factory=dict)


# ---- Prediction ----
class PredictRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    model_type: str = Field("rf", pattern="^(rf|lstm|ensemble)$")
    model_version: str = "v1"


class SignalOut(BaseModel):
    index: int
    signal: str
    confidence: float
    price: float
    risk_score: float
    rsi: float
    macd_hist: float


class PredictResponse(BaseModel):
    symbol: str
    model_type: str
    total_signals: int
    signals: List[SignalOut]


# ---- Backtest ----
class BacktestRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    model_type: str = Field("rf", pattern="^(rf|lstm|ensemble)$")
    model_version: str = "v1"
    initial_capital: float = 100_000.0
    commission_rate: float = 0.001
    slippage_bps: float = 5.0
    position_size_pct: float = 0.1
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10


class BacktestResponse(BaseModel):
    symbol: str
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_pnl: float
    total_commission: float
    total_slippage: float
    equity_curve: List[float]
    trades: List[Dict[str, Any]]


# ---- Metrics ----
class ModelMetricsResponse(BaseModel):
    models: List[Dict[str, Any]]


# ---- Strategy ----
class StrategyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)


class StrategyOut(BaseModel):
    id: int
    name: str
    description: str
    parameters: Dict[str, Any]
    user_id: int

    model_config = {"from_attributes": True}

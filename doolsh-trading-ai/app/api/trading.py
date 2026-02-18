"""Trading API endpoints: train, predict, backtest, metrics."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import get_settings
from app.core.deps import get_current_user, rate_limit, require_role
from app.models.user import User, UserRole
from app.schemas.trading import (
    BacktestRequest,
    BacktestResponse,
    ModelMetricsResponse,
    PredictRequest,
    PredictResponse,
    SignalOut,
    TrainRequest,
    TrainResponse,
)
from app.services.market_data import fetch_daily_prices
from app.services.signal_generator import generate_signals
from backtesting.engine import BacktestConfig, BacktestEngine
from ml.features.engineering import build_features
from ml.training.lstm_model import train_lstm
from ml.training.random_forest import train_random_forest

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/trading", tags=["trading"], dependencies=[Depends(rate_limit)])


@router.post("/train", response_model=TrainResponse)
async def train_model(
    body: TrainRequest,
    user: User = Depends(require_role(UserRole.ANALYST, UserRole.ADMIN)),
):
    """Train a model on historical daily data for the given symbol."""
    try:
        df = await fetch_daily_prices(body.symbol)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Market data fetch failed: {exc}")

    featured = build_features(df)
    if featured.empty or len(featured) < 100:
        raise HTTPException(status_code=422, detail="Insufficient data after feature engineering")

    hp = body.hyperparameters
    if body.model_type == "rf":
        result = train_random_forest(
            featured,
            symbol=body.symbol,
            horizon=body.horizon,
            n_estimators=hp.get("n_estimators", 200),
            max_depth=hp.get("max_depth", 12),
            version=body.version,
        )
    else:
        result = train_lstm(
            featured,
            symbol=body.symbol,
            horizon=body.horizon,
            seq_len=hp.get("seq_len", 30),
            hidden_size=hp.get("hidden_size", 128),
            num_layers=hp.get("num_layers", 2),
            epochs=hp.get("epochs", 50),
            batch_size=hp.get("batch_size", 64),
            lr=hp.get("lr", 1e-3),
            version=body.version,
        )

    extra = {k: v for k, v in result.items() if k not in TrainResponse.model_fields}
    return TrainResponse(
        model_type=result["model_type"],
        symbol=result["symbol"],
        version=result["version"],
        file_path=result["file_path"],
        accuracy=result["accuracy"],
        precision=result["precision"],
        recall=result["recall"],
        f1=result["f1"],
        hyperparameters=result["hyperparameters"],
        extra=extra,
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(
    body: PredictRequest,
    user: User = Depends(get_current_user),
):
    """Generate signals for the latest data using a trained model."""
    save_dir = Path(settings.model_save_dir)

    if body.model_type == "ensemble":
        rf_path = save_dir / f"rf_{body.symbol}_{body.model_version}.pkl"
        lstm_path = save_dir / f"lstm_{body.symbol}_{body.model_version}.pt"
        if not rf_path.exists() or not lstm_path.exists():
            raise HTTPException(status_code=404, detail="Both RF and LSTM models required for ensemble")
        model_path = str(rf_path)
    elif body.model_type == "rf":
        model_path = str(save_dir / f"rf_{body.symbol}_{body.model_version}.pkl")
    else:
        model_path = str(save_dir / f"lstm_{body.symbol}_{body.model_version}.pt")

    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    try:
        df = await fetch_daily_prices(body.symbol)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Market data fetch failed: {exc}")

    sigs = generate_signals(
        df,
        model_path=model_path,
        model_type=body.model_type if body.model_type != "ensemble" else "rf",
        ensemble=body.model_type == "ensemble",
        rf_path=str(save_dir / f"rf_{body.symbol}_{body.model_version}.pkl") if body.model_type == "ensemble" else None,
        lstm_path=str(save_dir / f"lstm_{body.symbol}_{body.model_version}.pt") if body.model_type == "ensemble" else None,
    )

    return PredictResponse(
        symbol=body.symbol,
        model_type=body.model_type,
        total_signals=len(sigs),
        signals=[SignalOut(**s) for s in sigs[-50:]],  # last 50 signals
    )


@router.post("/backtest", response_model=BacktestResponse)
async def backtest(
    body: BacktestRequest,
    user: User = Depends(get_current_user),
):
    """Run a backtest for a trained model on historical data."""
    save_dir = Path(settings.model_save_dir)
    if body.model_type == "rf":
        model_path = str(save_dir / f"rf_{body.symbol}_{body.model_version}.pkl")
    elif body.model_type == "lstm":
        model_path = str(save_dir / f"lstm_{body.symbol}_{body.model_version}.pt")
    else:
        model_path = str(save_dir / f"rf_{body.symbol}_{body.model_version}.pkl")

    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        df = await fetch_daily_prices(body.symbol)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Market data fetch failed: {exc}")

    sigs = generate_signals(
        df,
        model_path=model_path,
        model_type=body.model_type if body.model_type != "ensemble" else "rf",
        ensemble=body.model_type == "ensemble",
        rf_path=str(save_dir / f"rf_{body.symbol}_{body.model_version}.pkl") if body.model_type == "ensemble" else None,
        lstm_path=str(save_dir / f"lstm_{body.symbol}_{body.model_version}.pt") if body.model_type == "ensemble" else None,
    )

    config = BacktestConfig(
        initial_capital=body.initial_capital,
        commission_rate=body.commission_rate,
        slippage_bps=body.slippage_bps,
        position_size_pct=body.position_size_pct,
        stop_loss_pct=body.stop_loss_pct,
        take_profit_pct=body.take_profit_pct,
    )
    engine = BacktestEngine(config)
    featured = build_features(df)
    result = engine.run(featured, sigs)

    return BacktestResponse(symbol=body.symbol, **asdict(result))


@router.get("/metrics", response_model=ModelMetricsResponse)
async def list_model_metrics(
    user: User = Depends(get_current_user),
):
    """List all saved models with their file paths and sizes."""
    save_dir = Path(settings.model_save_dir)
    models = []
    if save_dir.exists():
        for p in sorted(save_dir.iterdir()):
            if p.suffix in (".pkl", ".pt"):
                parts = p.stem.split("_")
                models.append(
                    {
                        "file": p.name,
                        "model_type": parts[0] if parts else "unknown",
                        "symbol": parts[1] if len(parts) > 1 else "unknown",
                        "version": parts[2] if len(parts) > 2 else "unknown",
                        "size_kb": round(p.stat().st_size / 1024, 2),
                    }
                )
    return ModelMetricsResponse(models=models)

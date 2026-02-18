"""FastAPI backend — REST API for the AI auto-trading system.

Endpoints: system status, trading signals, model training,
backtesting, risk management, data fetching.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.config import load_config
from core.database import Signal, Trade, get_session, init_db

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Auto-Trading System", version="1.0.0")

_cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8000,http://localhost:8501",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Models ─────────────────────────────────────────────────────────


class TrainRequest(BaseModel):
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    model_type: str = "random_forest"
    limit: int = 500


class BacktestRequest(BaseModel):
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    model_type: str = "random_forest"
    initial_capital: float = 10000.0
    limit: int = 500


class SignalRequest(BaseModel):
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    model_type: str = "random_forest"
    limit: int = 500


class ExecuteRequest(BaseModel):
    symbol: str = "BTC/USDT"
    side: str = "BUY"
    confidence: float = 0.7
    model_type: str = "random_forest"


class KillSwitchRequest(BaseModel):
    enabled: bool


# ─── Startup ──────────────────────────────────────────────────────────────────


_scheduler = None


@app.on_event("startup")
def startup():
    cfg = load_config()
    init_db(cfg.db_path)
    logger.info("API started — mode=%s", cfg.mode)


@app.on_event("shutdown")
def shutdown():
    global _auto_trading, _scheduler
    logger.info("Shutting down API — stopping auto-trading and cleaning up...")
    _auto_trading = False
    if _scheduler is not None:
        try:
            _scheduler.shutdown(wait=False)
        except Exception as e:
            logger.warning("Scheduler shutdown error: %s", e)
        _scheduler = None
    from core.database import get_engine
    engine = get_engine()
    if engine:
        engine.dispose()
    logger.info("Shutdown complete.")


# ─── System ───────────────────────────────────────────────────────────────────


@app.get("/api/status")
def get_status():
    """System health and configuration overview."""
    cfg = load_config()
    from core.risk_engine import get_risk_summary

    risk = get_risk_summary()
    return {
        "status": "running",
        "mode": cfg.mode,
        "exchange": cfg.exchange,
        "symbols": cfg.symbols,
        "primary_timeframe": cfg.primary_timeframe,
        "kill_switch": cfg.kill_switch,
        "risk": risk,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/config")
def get_config():
    """Return current configuration (non-sensitive)."""
    cfg = load_config()
    return {
        "mode": cfg.mode,
        "exchange": cfg.exchange,
        "symbols": cfg.symbols,
        "primary_timeframe": cfg.primary_timeframe,
        "confirmation_timeframes": cfg.confirmation_timeframes,
        "risk": {
            "max_position_pct": cfg.max_position_pct,
            "atr_sl_multiplier": cfg.atr_sl_multiplier,
            "atr_tp_multiplier": cfg.atr_tp_multiplier,
            "max_daily_loss_pct": cfg.max_daily_loss_pct,
            "max_drawdown_pct": cfg.max_drawdown_pct,
            "cooldown_minutes": cfg.cooldown_minutes,
            "max_open_trades": cfg.max_open_trades,
            "kill_switch": cfg.kill_switch,
        },
        "model": {
            "primary": cfg.primary_model,
            "fallback": cfg.fallback_model,
            "min_confidence": cfg.min_confidence,
        },
    }


# ─── Data ─────────────────────────────────────────────────────────────────────


@app.get("/api/data/{symbol}/{timeframe}")
def fetch_data(symbol: str, timeframe: str = "1h", limit: int = 200):
    """Fetch OHLCV candles for a symbol."""
    from core.data_engine import fetch_ohlcv, generate_synthetic

    cfg = load_config()
    sym = symbol.replace("-", "/")

    try:
        df = fetch_ohlcv(sym, timeframe=timeframe, limit=limit, exchange_id=cfg.exchange)
    except Exception as e:
        logger.warning("fetch_ohlcv failed for %s, falling back to synthetic: %s", sym, e)
        df = generate_synthetic(sym, bars=limit)

    if df.empty:
        df = generate_synthetic(sym, bars=limit)

    records = df.tail(100).to_dict(orient="records")
    return {"symbol": sym, "timeframe": timeframe, "count": len(records), "candles": records}


@app.get("/api/ticker/{symbol}")
def get_ticker(symbol: str):
    """Get current ticker price."""
    from core.data_engine import fetch_ticker

    cfg = load_config()
    sym = symbol.replace("-", "/")
    try:
        ticker = fetch_ticker(sym, exchange_id=cfg.exchange)
        return {"symbol": sym, "price": ticker.get("last", 0), "bid": ticker.get("bid", 0), "ask": ticker.get("ask", 0)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Signals ──────────────────────────────────────────────────────────────────


@app.post("/api/signal")
def generate_signal(req: SignalRequest):
    """Generate a trading signal for a symbol."""
    from core.data_engine import fetch_ohlcv, generate_synthetic
    from core.feature_engine import build_features
    from core.strategy_engine import generate_signal as gen_sig

    cfg = load_config()
    try:
        df = fetch_ohlcv(req.symbol, timeframe=req.timeframe, limit=req.limit, exchange_id=cfg.exchange)
    except Exception as e:
        logger.warning("fetch_ohlcv failed for signal %s, falling back to synthetic: %s", req.symbol, e)
        df = generate_synthetic(req.symbol, bars=req.limit)

    if df.empty:
        df = generate_synthetic(req.symbol, bars=req.limit)

    sig = gen_sig(df, req.symbol, req.timeframe, model_type=req.model_type)

    # Log to DB
    session = get_session()
    try:
        session.add(Signal(
            symbol=req.symbol, timeframe=req.timeframe,
            signal=sig["signal"], confidence=sig["confidence"],
            model_type=req.model_type,
        ))
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error("Failed to log signal to DB: %s", e)
    finally:
        session.close()

    return sig


@app.get("/api/signals/history")
def signal_history(limit: int = 50):
    """Get recent signal history from database."""
    session = get_session()
    try:
        signals = session.query(Signal).order_by(Signal.id.desc()).limit(limit).all()
        return [{
            "id": s.id, "timestamp": s.timestamp.isoformat() if s.timestamp else "",
            "symbol": s.symbol, "timeframe": s.timeframe,
            "signal": s.signal, "confidence": s.confidence,
            "model_type": s.model_type,
        } for s in signals]
    finally:
        session.close()


# ─── Model Training ──────────────────────────────────────────────────────────


@app.post("/api/train")
def train_model(req: TrainRequest):
    """Train a model on historical data."""
    from core.data_engine import fetch_ohlcv, generate_synthetic
    from core.feature_engine import build_features
    from core.model_engine import train_model as do_train

    cfg = load_config()
    try:
        df = fetch_ohlcv(req.symbol, timeframe=req.timeframe, limit=req.limit, exchange_id=cfg.exchange)
    except Exception as e:
        logger.warning("fetch_ohlcv failed for training %s, falling back to synthetic: %s", req.symbol, e)
        df = generate_synthetic(req.symbol, bars=req.limit)

    if df.empty:
        df = generate_synthetic(req.symbol, bars=req.limit)

    featured = build_features(df)
    if len(featured) < 100:
        raise HTTPException(status_code=400, detail=f"Not enough data: {len(featured)} rows (need 100+)")

    try:
        result = do_train(featured, req.symbol, req.timeframe, req.model_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


@app.get("/api/models")
def list_models():
    """List all trained models."""
    from core.model_engine import MODEL_DIR

    models = []
    for f in MODEL_DIR.glob("*"):
        if f.suffix in (".pkl", ".pt"):
            parts = f.stem.split("_")
            models.append({
                "file": f.name,
                "type": parts[0] if parts else "unknown",
                "size_kb": round(f.stat().st_size / 1024, 1),
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
    return {"models": models}


# ─── Backtesting ──────────────────────────────────────────────────────────────


@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    """Run a backtest on historical data."""
    from core.backtester import run_backtest as do_backtest
    from core.data_engine import fetch_ohlcv, generate_synthetic
    from core.model_engine import get_latest_model

    cfg = load_config()
    try:
        df = fetch_ohlcv(req.symbol, timeframe=req.timeframe, limit=req.limit, exchange_id=cfg.exchange)
    except Exception as e:
        logger.warning("fetch_ohlcv failed for backtest %s, falling back to synthetic: %s", req.symbol, e)
        df = generate_synthetic(req.symbol, bars=req.limit)

    if df.empty:
        df = generate_synthetic(req.symbol, bars=req.limit)

    model_path = get_latest_model(req.symbol, req.timeframe, req.model_type)
    if not model_path:
        raise HTTPException(status_code=404, detail="No trained model found. Train a model first via /api/train")

    result = do_backtest(
        df, model_path=model_path, model_type=req.model_type,
        initial_capital=req.initial_capital,
    )
    return {
        "initial_capital": result.initial_capital,
        "final_capital": result.final_capital,
        "total_return_pct": result.total_return_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown_pct": result.max_drawdown_pct,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "winning_trades": result.winning_trades,
        "losing_trades": result.losing_trades,
        "expectancy": result.expectancy,
        "cagr": result.cagr,
        "profit_factor": result.profit_factor,
        "avg_win": result.avg_win,
        "avg_loss": result.avg_loss,
        "equity_curve": result.equity_curve[-200:],  # limit for response size
    }


# ─── Trading / Execution ─────────────────────────────────────────────────────


@app.post("/api/trade/execute")
def execute_trade(req: ExecuteRequest):
    """Execute a trade signal (paper or live)."""
    from core.execution_engine import execute_signal

    signal = {
        "symbol": req.symbol,
        "signal": req.side,
        "confidence": req.confidence,
        "model_type": req.model_type,
        "atr": 0,
    }
    result = execute_signal(signal)
    if result is None:
        raise HTTPException(status_code=400, detail="Trade rejected by risk engine or already positioned")
    return result


@app.get("/api/trades")
def get_trades(status: str = "all", limit: int = 50):
    """Get trade history from database."""
    session = get_session()
    try:
        q = session.query(Trade)
        if status != "all":
            q = q.filter(Trade.status == status)
        trades = q.order_by(Trade.id.desc()).limit(limit).all()
        return [{
            "id": t.id,
            "timestamp": t.timestamp.isoformat() if t.timestamp else "",
            "symbol": t.symbol, "side": t.side,
            "price": t.price, "quantity": t.quantity,
            "pnl": t.pnl, "status": t.status, "mode": t.mode,
            "stop_loss": t.stop_loss, "take_profit": t.take_profit,
            "model_type": t.model_type, "confidence": t.confidence,
            "close_reason": t.close_reason,
            "closed_at": t.closed_at.isoformat() if t.closed_at else None,
        } for t in trades]
    finally:
        session.close()


@app.post("/api/trade/close/{symbol}")
def close_trade(symbol: str):
    """Manually close an open paper trade."""
    from core.data_engine import fetch_ticker
    from core.execution_engine import paper_close

    cfg = load_config()
    sym = symbol.replace("-", "/")
    try:
        ticker = fetch_ticker(sym, exchange_id=cfg.exchange)
        price = ticker.get("last", 0)
    except Exception as e:
        logger.warning("Failed to fetch ticker for %s: %s", sym, e)
        price = 0

    if price <= 0:
        raise HTTPException(status_code=400, detail="Cannot determine current price")

    result = paper_close(sym, price, reason="manual_api")
    if result is None:
        raise HTTPException(status_code=404, detail="No open position for this symbol")
    return result


# ─── Risk ─────────────────────────────────────────────────────────────────────


@app.get("/api/risk")
def get_risk():
    """Get current risk state and analytics."""
    from core.risk_engine import get_risk_summary

    return get_risk_summary()


@app.post("/api/risk/init")
def init_risk_capital(capital: float = 10000.0):
    """Initialize risk engine with starting capital."""
    from core.risk_engine import init_risk

    init_risk(capital)
    return {"status": "ok", "capital": capital}


@app.post("/api/risk/kill-switch")
def set_kill_switch(req: KillSwitchRequest):
    """Toggle the kill switch."""
    val = "true" if req.enabled else "false"
    os.environ["KILL_SWITCH"] = val
    return {"kill_switch": req.enabled}


# ─── Auto-Trading Loop ───────────────────────────────────────────────────────

_auto_trading = False
_consecutive_cycle_errors = 0
_MAX_CYCLE_ERRORS = 5  # Stop auto-trading after this many consecutive failures


@app.post("/api/auto/start")
def start_auto_trading():
    """Start the auto-trading loop."""
    global _auto_trading, _scheduler
    if _auto_trading:
        return {"status": "already_running"}

    from core.risk_engine import init_risk

    cfg = load_config()
    init_risk(cfg.default_capital)
    _auto_trading = True

    # Start background scheduler
    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        _scheduler = BackgroundScheduler()
        _scheduler.add_job(_trading_cycle, "interval", minutes=60, id="trading_cycle")
        _scheduler.start()
        logger.info("Auto-trading started (1h cycle)")
    except ImportError:
        logger.warning("APScheduler not installed — auto-trading requires manual trigger")

    return {"status": "started", "mode": cfg.mode, "interval": "1h"}


@app.post("/api/auto/stop")
def stop_auto_trading():
    """Stop the auto-trading loop."""
    global _auto_trading, _scheduler
    _auto_trading = False
    if _scheduler is not None:
        try:
            _scheduler.shutdown(wait=False)
        except Exception as e:
            logger.warning("Could not stop scheduler cleanly: %s", e)
        _scheduler = None
    return {"status": "stopped"}


@app.get("/api/auto/status")
def auto_status():
    return {"running": _auto_trading}


def _trading_cycle():
    """One auto-trading cycle: fetch data, generate signals, execute.

    Includes a circuit breaker that halts auto-trading after repeated failures.
    """
    global _auto_trading, _consecutive_cycle_errors

    if not _auto_trading:
        return

    from core.data_engine import fetch_multi_timeframe, fetch_ohlcv
    from core.execution_engine import execute_signal, paper_check_exits
    from core.strategy_engine import generate_multi_timeframe_signal

    cfg = load_config()
    all_tfs = [cfg.primary_timeframe] + cfg.confirmation_timeframes
    cycle_had_error = False

    for symbol in cfg.symbols:
        try:
            # Fetch data
            tf_data = fetch_multi_timeframe(symbol, all_tfs, limit=200, exchange_id=cfg.exchange)
            if not tf_data:
                continue

            # Generate signal
            sig = generate_multi_timeframe_signal(tf_data, symbol, model_type=cfg.primary_model)

            # Execute if actionable
            if sig["signal"] != "HOLD":
                execute_signal(sig)

            # Check exits on open positions
            from core.data_engine import fetch_ticker
            ticker = fetch_ticker(symbol, exchange_id=cfg.exchange)
            if ticker:
                paper_check_exits({symbol: ticker.get("last", 0)})

        except Exception as e:
            logger.error("Trading cycle error for %s: %s", symbol, e)
            cycle_had_error = True

    # Circuit breaker: track consecutive failures
    if cycle_had_error:
        _consecutive_cycle_errors += 1
        if _consecutive_cycle_errors >= _MAX_CYCLE_ERRORS:
            logger.error(
                "CIRCUIT BREAKER: %d consecutive cycle errors — halting auto-trading. "
                "Restart via /api/auto/start after investigating.",
                _consecutive_cycle_errors,
            )
            _auto_trading = False
    else:
        _consecutive_cycle_errors = 0


# ─── Entry Point ──────────────────────────────────────────────────────────────


def start_api():
    """Start the API server."""
    import uvicorn

    cfg = load_config()
    uvicorn.run(app, host=cfg.api_host, port=cfg.api_port, log_level="info")


if __name__ == "__main__":
    start_api()

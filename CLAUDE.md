# CLAUDE.md — AI Auto-Trading System

## Project Overview

AI-powered cryptocurrency and equity auto-trading system built in Python. Designed for low-resource environments (UserLand on Android, 2GB RAM minimum). Paper trading by default — no real money unless explicitly switched to live mode.

The repository contains two systems:
1. **Primary System** (root level): Standalone crypto auto-trading via CCXT/Binance
2. **Secondary System** (`doolsh-trading-ai/`): Zerodha Kite Connect integration for Indian equities and F&O trading

## Repository Structure

```
core/                   Core trading engine modules
  config.py               Config loader (YAML + env vars, LRU-cached)
  database.py             SQLite ORM (trades, signals, models, daily PnL)
  data_engine.py          Market data via ccxt (crypto) + yfinance (equities)
  feature_engine.py       Technical indicators + regime detection
  model_engine.py         ML models (RandomForest, XGBoost, LSTM)
  strategy_engine.py      Signal generation, multi-timeframe confirmation
  risk_engine.py          Position sizing, SL/TP, kill-switch, drawdown limits
  execution_engine.py     Paper + live order execution
  backtester.py           Backtesting with Sharpe, drawdown, CAGR

api/
  main.py                 FastAPI REST backend (all endpoints)

dashboard/
  app.py                  Streamlit web UI (real-time monitoring)

config/
  config.yaml             All settings (trading, risk, model, API)

scripts/
  install.sh              One-time installation script
  start.sh                Service launcher (API + Dashboard)

doolsh-trading-ai/        Secondary system (Zerodha/Indian equities)
  app/                    FastAPI app for Zerodha integration
  ml/                     ML models for equity trading
  backtesting/            Backtest engine for equities
  data/                   Data storage (gitignored)
  tests/                  pytest test suite
  trade.py                Main trading logic

database/                 Auto-created at runtime
  trading.db              SQLite database
  models/                 Trained ML model files (.pkl, .pt)
```

## Tech Stack

- **Language**: Python 3.8+
- **API**: FastAPI + Uvicorn (port 8000)
- **Dashboard**: Streamlit (port 8501)
- **Database**: SQLite via SQLAlchemy (single-file, zero config)
- **ML**: scikit-learn, XGBoost, LightGBM, PyTorch (CPU-only)
- **Market Data**: ccxt (crypto exchanges), yfinance (equities)
- **Technical Analysis**: `ta` library (RSI, MACD, Bollinger, ATR, ADX, etc.)
- **Scheduling**: APScheduler (replaces Celery)

## Development Commands

### Setup and Run

```bash
# First-time setup and start (auto-installs everything)
chmod +x start.sh && ./start.sh

# Manual install only
./scripts/install.sh

# Start services only (after install)
./scripts/start.sh
```

### Running Tests

Tests are in the secondary system (`doolsh-trading-ai/tests/`):

```bash
# Run all tests
cd doolsh-trading-ai && python -m pytest tests/ -v

# Run specific test file
cd doolsh-trading-ai && python -m pytest tests/test_features.py -v

# Run with async support (pytest-asyncio is pre-configured)
cd doolsh-trading-ai && python -m pytest tests/ -v --asyncio-mode=auto
```

### Linting

```bash
# Lint with ruff (configured in doolsh-trading-ai)
cd doolsh-trading-ai && ruff check .

# Auto-fix lint issues
cd doolsh-trading-ai && ruff check --fix .
```

### Dependencies

```bash
# Primary system
pip install -r requirements.txt

# Secondary system (Zerodha)
pip install -r doolsh-trading-ai/requirements.txt
```

## Architecture and Key Patterns

### Configuration Precedence

Configuration resolves in this order (highest to lowest priority):
1. Environment variables (`TRADING_MODE`, `KILL_SWITCH`, `EXCHANGE_API_KEY`, `EXCHANGE_API_SECRET`)
2. `config/config.yaml` values
3. Hard-coded defaults in `core/config.py`

Config is loaded via `load_config()` which is LRU-cached (singleton pattern). All modules receive config through this function.

### Module Pipeline

```
data_engine (fetch OHLCV) -> feature_engine (compute indicators) -> model_engine (ML predict)
    -> strategy_engine (generate signals) -> risk_engine (validate) -> execution_engine (execute)
```

### Database Schema (SQLite)

Four tables managed by SQLAlchemy ORM in `core/database.py`:
- `Trade` — open/closed positions with PnL, SL/TP, model info
- `Signal` — signal history with confidence and features JSON
- `ModelRecord` — trained model metadata, accuracy, feature importance
- `DailyPnL` — daily realized PnL and win/loss counts

Database auto-initializes on first access via `init_db()`. No migration system — schema is static.

### ML Label Map

Defined in `core/model_engine.py`:
- `0` = SELL
- `1` = HOLD
- `2` = BUY

### API Structure

All endpoints are defined in `api/main.py`. Key Pydantic request models:
- `TrainRequest` — symbol, timeframe, model_type, limit
- `BacktestRequest` — symbol, timeframe, model_type, initial_capital, limit
- `SignalRequest` — symbol, timeframe, model_type
- `ExecuteRequest` — symbol, side, quantity
- `KillSwitchRequest` — enabled (bool)

CORS is fully open (`allow_origins=["*"]`).

### Testing Conventions

- Framework: pytest + pytest-asyncio
- Fixtures in `tests/conftest.py` provide an async `client` via `httpx.AsyncClient` with `ASGITransport`
- Test files follow `test_*.py` naming
- Test categories: API health, backtesting, features, risk management, security, strategies, F&O scanner, options chain

## Code Conventions

- **Type hints** used throughout with `from __future__ import annotations`
- **Docstrings** at module and function level
- **Logging** via `logging.getLogger(__name__)` in each module
- **PEP 8** naming: snake_case for functions/variables, PascalCase for classes
- **Stateless modules** — all core engines are stateless except in-memory risk state tracking
- **Config injection** — modules call `load_config()` rather than accepting config parameters
- **No Docker** — designed to run directly on host with virtualenv

## Important Conventions for AI Assistants

1. **Never commit secrets.** API keys, TOTP secrets, and credentials belong in `.env` or environment variables, not in code or config files.
2. **Paper mode by default.** Always ensure `system.mode: paper` unless the user explicitly requests live trading. The kill switch (`risk.kill_switch`) must be respected.
3. **Low-resource awareness.** This system targets 2GB RAM, CPU-only environments. Avoid adding heavy dependencies or GPU-required code. LSTM is disabled by default to save memory.
4. **SQLite only.** Do not introduce PostgreSQL, Redis, or other external database dependencies. The single-file SQLite approach is intentional.
5. **No Docker/cloud assumptions.** The system runs on bare metal / UserLand. Do not add Dockerfiles or cloud deployment configs unless asked.
6. **Two separate systems.** The root-level system (CCXT/Binance) and `doolsh-trading-ai/` (Zerodha) have independent dependency files, configs, and entry points. Changes to one do not necessarily affect the other.
7. **Risk management is critical.** Any changes to trading logic must respect the risk engine: position size limits, drawdown protection, daily loss caps, cooldown timers, and the kill switch.
8. **Test before merging.** Run the pytest suite in `doolsh-trading-ai/tests/` to validate changes to the secondary system. The primary system does not have a dedicated test suite yet.

## Services and Ports

| Service         | Port | Command                                    |
|-----------------|------|--------------------------------------------|
| FastAPI (API)   | 8000 | `uvicorn api.main:app --host 0.0.0.0`     |
| Streamlit (UI)  | 8501 | `streamlit run dashboard/app.py`           |
| Swagger Docs    | 8000 | http://localhost:8000/docs                 |

## Environment Variables

### Primary System (Crypto)
| Variable             | Values          | Purpose                    |
|----------------------|-----------------|----------------------------|
| `TRADING_MODE`       | `paper` / `live`| Override trading mode       |
| `KILL_SWITCH`        | `true` / `false`| Emergency stop all trading  |
| `EXCHANGE_API_KEY`   | string          | Binance API key (live mode) |
| `EXCHANGE_API_SECRET`| string          | Binance API secret          |

### Secondary System (Zerodha)
See `doolsh-trading-ai/.env.example` for the full list of Zerodha-specific environment variables including Kite API credentials, TOTP secrets, market hours, and ML feature flags.

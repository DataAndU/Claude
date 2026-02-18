# AI Auto-Trading System

Fully modular, AI-powered crypto auto-trading system. Runs on low-resource environments (UserLAnd / Android / 2GB RAM). Paper trading by default — no real money unless you switch to live mode.

## Quick Start

```bash
chmod +x start.sh
./start.sh
```

First run auto-installs everything. After startup:
- **API**: http://localhost:8000/docs (Swagger UI)
- **Dashboard**: http://localhost:8501

## Architecture

```
core/           # Engine modules
  config.py       Config loader (YAML + env vars)
  database.py     SQLite ORM (trades, signals, models)
  data_engine.py  Market data via ccxt (crypto) + yfinance (equities)
  feature_engine.py  Technical indicators, regime detection
  model_engine.py    ML models (RandomForest, XGBoost, LSTM)
  strategy_engine.py Signal generation, multi-timeframe confirmation
  risk_engine.py     Position sizing, SL/TP, kill-switch
  execution_engine.py Paper + live order execution
  backtester.py      Backtesting with Sharpe, drawdown, CAGR

api/            # FastAPI REST backend
  main.py         All endpoints

dashboard/      # Streamlit web UI
  app.py          Real-time monitoring dashboard

config/         # Configuration
  config.yaml     All settings (trading, risk, model, API)

scripts/        # Setup and launcher scripts
  install.sh      One-time installation
  start.sh        Service launcher

database/       # Auto-created at runtime
  trading.db      SQLite database
  models/         Trained ML model files
```

## Features

- **ML Models**: RandomForest, XGBoost, lightweight LSTM (CPU-only)
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic RSI, OBV, moving averages
- **Regime Detection**: Classifies market as trending up/down, ranging, or high volatility
- **Multi-Timeframe**: Primary signal confirmed across higher timeframes
- **Risk Management**: ATR-based SL/TP, max drawdown protection, daily loss cap, cooldown timer, kill-switch
- **Backtesting**: Sharpe ratio, max drawdown, CAGR, profit factor, win rate, equity curves
- **Paper Trading**: Default mode — simulates all trades with no real money
- **Live Trading**: Sends real orders via ccxt (Binance) when enabled

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/status | System health and risk overview |
| GET | /api/config | Current configuration |
| GET | /api/data/{symbol}/{timeframe} | Fetch OHLCV candles |
| POST | /api/signal | Generate trading signal |
| GET | /api/signals/history | Signal history |
| POST | /api/train | Train ML model |
| GET | /api/models | List trained models |
| POST | /api/backtest | Run backtest |
| POST | /api/trade/execute | Execute a trade |
| GET | /api/trades | Trade history |
| GET | /api/risk | Risk state |
| POST | /api/risk/kill-switch | Toggle kill switch |
| POST | /api/auto/start | Start auto-trading loop |
| POST | /api/auto/stop | Stop auto-trading loop |

## Configuration

Edit `config/config.yaml`:

```yaml
system:
  mode: paper           # paper or live
trading:
  exchange: binance
  symbols: [BTC/USDT, ETH/USDT]
risk:
  max_position_pct: 0.02
  kill_switch: false
```

Environment variables override config:
- `TRADING_MODE=paper|live`
- `KILL_SWITCH=true|false`
- `EXCHANGE_API_KEY=...` (live mode only)
- `EXCHANGE_API_SECRET=...` (live mode only)

## Workflow

1. **Train a model**: Dashboard > Models > Train (or `POST /api/train`)
2. **Backtest**: Dashboard > Backtest > Run (or `POST /api/backtest`)
3. **Generate signals**: Dashboard > Signals > Generate (or `POST /api/signal`)
4. **Auto-trade**: Dashboard > Dashboard > Start Auto-Trading

## Requirements

- Python 3.8+
- 2GB RAM minimum
- Internet connection (for exchange data)
- No Docker, no cloud, no GPU needed

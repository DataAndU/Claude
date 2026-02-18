# KiteAI — Personal AI Auto-Trading Addon for Zerodha Kite

AI-powered auto-trading system for Indian equity markets (NSE/BSE) via Zerodha Kite Connect. Runs locally on low-resource environments. Paper trading by default — no real money unless you explicitly switch to live mode.

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your Zerodha Kite API credentials

# 2. Run (auto-installs on first run)
chmod +x start.sh
./start.sh
```

After startup:
- **API**: http://localhost:8000/docs (Swagger UI)
- **Dashboard**: http://localhost:8501

## Architecture

```
core/                  # Engine modules
  config.py              Config loader (YAML + .env)
  database.py            SQLite ORM (trades, signals, models, P&L)
  kite_auth.py           Zerodha Kite login (TOTP auto-login, token)
  kite_data.py           Historical data (Kite -> yfinance -> synthetic fallback)
  kite_orders.py         Order execution (paper + live via Kite)
  kite_ticker.py         WebSocket live price feed
  feature_engine.py      30+ technical indicators, regime features
  model_engine.py        ML models (RandomForest, XGBoost, LSTM)
  signal_engine.py       AI confidence scoring, quality grading
  scanner.py             NSE stock scanner with composite scoring
  risk_engine.py         Position sizing, SL/TP, kill switch, tilt protection
  auto_trader.py         Auto-trading orchestrator (scan -> signal -> trade)
  backtester.py          Backtesting with Zerodha-like costs

api/                   # FastAPI REST backend
  main.py                All endpoints (Kite, scanner, signals, trading, risk)

dashboard/             # Streamlit web UI
  app.py                 8-page monitoring dashboard

config/                # Configuration
  config.yaml            All settings (trading, risk, AI, backtest)

scripts/               # Setup and launcher scripts
  install.sh             One-time installation
  start.sh               Service launcher
```

## Features

### Trading
- **Markets**: NSE, BSE equities + NFO (F&O)
- **Modes**: Intraday (MIS), BTST (NRML), F&O
- **Auto Square-Off**: MIS positions closed before 3:15 PM IST
- **Paper Trading**: Default mode -- simulates all trades with realistic slippage

### AI / ML
- **Models**: RandomForest, XGBoost, LSTM (CPU-only PyTorch)
- **30+ Indicators**: RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic RSI, OBV, Ichimoku Cloud, VWAP, candlestick patterns
- **Regime Detection**: Trending up/down, mean-reverting, high volatility
- **Signal Quality**: A/B/C/D grading based on 6-dimension confidence scoring
- **Multi-Timeframe**: Confirmation across short/medium/long windows

### Risk Management
- **Dynamic Position Sizing**: ATR + volatility + confidence adjusted
- **Dynamic SL/TP**: Regime-aware stop-loss and take-profit
- **Tilt Protection**: Reduces size after consecutive losses, requires higher quality signals
- **Daily Loss Cap**: Configurable INR limit per day
- **Kill Switch**: Emergency stop for all trading

### Backtesting
- **Zerodha-Like Costs**: Commission, STT, slippage simulation
- **Metrics**: Sharpe ratio, CAGR, max drawdown, profit factor, win rate
- **Equity Curves**: Visual performance tracking

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/status | System health and overview |
| GET | /api/config | Current configuration |
| GET | /api/kite/status | Kite connection status |
| POST | /api/kite/login | Login to Zerodha Kite |
| GET | /api/kite/margins | Account margins |
| GET | /api/kite/positions | Open positions |
| GET | /api/scan | Scan watchlist for opportunities |
| GET | /api/scan/fno | Scan F&O opportunities |
| POST | /api/signal | Generate AI trading signal |
| POST | /api/train | Train ML model |
| GET | /api/models | List trained models |
| POST | /api/backtest | Run backtest |
| POST | /api/trade | Execute a trade |
| GET | /api/trades | Trade history |
| GET | /api/risk | Risk analytics |
| POST | /api/risk/kill-switch | Toggle kill switch |
| POST | /api/auto/start | Start auto-trading loop |
| POST | /api/auto/stop | Stop auto-trading loop |
| POST | /api/auto/cycle | Run single trading cycle |
| GET | /api/pnl/daily | Daily P&L history |

## Configuration

### Environment Variables (.env)

```bash
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret
KITE_USER_ID=your_user_id
KITE_PASSWORD=your_password
KITE_TOTP_SECRET=your_totp_secret
TRADING_MODE=paper  # paper or live
```

### Config File (config/config.yaml)

```yaml
system:
  mode: paper
trading:
  exchange: NSE
  watchlist: [RELIANCE, TCS, INFY, HDFCBANK, ...]
  intraday_enabled: true
  btst_enabled: true
  fno_enabled: false
risk:
  max_daily_loss: 5000    # INR
  max_trades_per_day: 10
  kill_switch: false
```

## Workflow

1. **Configure**: Set Kite credentials in `.env`, adjust `config/config.yaml`
2. **Train**: Dashboard > Models > Train (or `POST /api/train`)
3. **Backtest**: Dashboard > Backtest > Run (or `POST /api/backtest`)
4. **Scan**: Dashboard > Scanner > Scan Now
5. **Auto-Trade**: Dashboard > Start Auto-Trading

## Requirements

- Python 3.8+
- 2GB RAM minimum
- Internet connection (for Kite API / market data)
- Zerodha Kite Connect API subscription
- No Docker, no cloud, no GPU needed

## Disclaimer

This is a personal-use tool for educational and research purposes. Trading in financial markets involves risk. Use paper mode to test strategies before considering live trading. The authors are not responsible for any financial losses.

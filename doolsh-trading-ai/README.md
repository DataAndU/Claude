# Doolsh Trading AI

Automated AI-powered trading engine for **Zerodha Kite**, designed to run on **Userland (Android)** or any Linux environment. No Docker, no PostgreSQL, no Redis — just Python and SQLite.

> **DISCLAIMER:** This software is for **educational and research purposes only**. It does **not** constitute financial advice. Automated trading involves substantial risk of financial loss. Past performance does not guarantee future results. **Use at your own risk.** The authors are not responsible for any losses incurred.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                  Userland / Android / Linux VPS                    │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    FastAPI Application                       │  │
│  │                                                              │  │
│  │  /auth/*     /kite/*      /trading/*    /orders/*            │  │
│  │  /live/*     /strategies/* /admin/*                          │  │
│  └──────┬───────────┬──────────────┬───────────────────────────┘  │
│         │           │              │                               │
│  ┌──────▼───┐ ┌─────▼──────┐ ┌────▼─────────────────────────┐    │
│  │ Auth +   │ │  Kite      │ │  Trading Engine               │    │
│  │ JWT      │ │  Session   │ │                               │    │
│  │          │ │  Manager   │ │  Signal Generator             │    │
│  │          │ │  + TOTP    │ │  Risk Manager                 │    │
│  │          │ │  Auto-login│ │  Order Manager (Paper + Live) │    │
│  └──────────┘ └─────┬──────┘ │  Auto-Trader Loop             │    │
│                     │        │  APScheduler                   │    │
│                     │        └────┬──────────────────────────┘    │
│                     │             │                                │
│  ┌──────────────────▼─────────────▼──────────────────────────┐    │
│  │                  Zerodha Kite Connect API                  │    │
│  │  Historical Data │ Live Quotes │ Order Placement           │    │
│  │  WebSocket Ticker│ Positions   │ Portfolio                 │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                    │
│  ┌──────────────┐  ┌──────────────────────────────────────────┐    │
│  │ SQLite DB    │  │  ML Pipeline                             │    │
│  │ (data/       │  │  Random Forest │ LSTM (PyTorch)          │    │
│  │  doolsh.db)  │  │  Feature Engineering │ Backtesting       │    │
│  └──────────────┘  └──────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Features

### Trading Engine
- **Zerodha Kite Connect** — full API integration (orders, positions, holdings, margins)
- **Auto-login via TOTP** — no browser needed, fully headless
- **Paper trading mode** — test strategies with simulated orders
- **Live trading mode** — real order execution on Zerodha
- **Auto square-off** — closes all MIS positions before market close
- **WebSocket live feed** — real-time tick data via Kite Ticker

### Risk Management
- Max daily loss cap (INR)
- Max position value per trade
- Max concurrent open positions
- Per-position stop-loss and take-profit
- Max trades per day
- Minimum confidence threshold filter
- Dynamic position sizing

### ML & Signals
- **Feature Engineering** — RSI, MACD, Bollinger Bands, ATR, OBV, moving averages, volatility
- **Random Forest classifier** — with time-series cross-validation
- **LSTM (PyTorch)** — sequential signal prediction
- **Ensemble mode** — majority vote across RF + LSTM
- **Risk-scored signals** — BUY / SELL / HOLD with confidence

### Backtesting
- Portfolio simulation with slippage and commission
- Sharpe ratio, Sortino ratio, max drawdown
- Equity curve and per-trade breakdown

### Userland Optimized
- **SQLite** — zero-config database, no server needed
- **In-memory cache** — replaces Redis entirely
- **APScheduler** — replaces Celery, runs in-process
- **CPU-only PyTorch** — works on ARM / Android
- **Single process** — everything runs in one `uvicorn` process

---

## Project Structure

```
doolsh-trading-ai/
├── app/
│   ├── api/
│   │   ├── auth.py          # Register, login, JWT refresh
│   │   ├── kite.py          # Kite login, auto-login, status, margins
│   │   ├── trading.py       # /train, /predict, /backtest, /metrics
│   │   ├── orders.py        # Place, cancel, list orders + positions
│   │   ├── live.py          # Auto-trade enable/disable, ticker, risk
│   │   ├── strategies.py    # Strategy CRUD
│   │   └── admin.py         # User management, dashboard
│   ├── core/
│   │   ├── config.py        # All settings (Kite, trading, risk, ML)
│   │   ├── database.py      # SQLite + async SQLAlchemy
│   │   ├── kite.py          # Kite session manager + TOTP auto-login
│   │   ├── security.py      # JWT + bcrypt
│   │   ├── redis.py         # In-memory cache (no Redis needed)
│   │   ├── deps.py          # FastAPI dependencies
│   │   └── logging.py       # Logging config
│   ├── models/              # SQLAlchemy ORM (SQLite-compatible)
│   ├── schemas/             # Pydantic request/response models
│   ├── services/
│   │   ├── market_data.py   # Kite historical + live data
│   │   ├── order_manager.py # Paper + live order placement
│   │   ├── risk_manager.py  # Risk controls and position sizing
│   │   ├── auto_trader.py   # Automated trading loop
│   │   ├── live_feed.py     # Kite WebSocket ticker
│   │   ├── signal_generator.py
│   │   └── scheduler.py     # APScheduler (replaces Celery)
│   ├── strategies/          # Pluggable strategy implementations
│   └── main.py              # FastAPI entry point
├── ml/                      # Feature engineering + model training
├── backtesting/             # Backtest engine
├── tests/                   # pytest suite
├── scripts/
│   ├── setup_userland.sh    # One-command Userland setup
│   ├── start.sh             # Quick start script
│   ├── seed_admin.py        # Create admin user
│   └── run_backtest.py      # CLI backtest tool
├── data/                    # SQLite DB + market data
├── requirements.txt
└── .env.example
```

---

## Quick Start on Userland

### Prerequisites

1. Install **Userland** from the Play Store
2. Create an **Ubuntu** session in Userland
3. Get a **Kite Connect API** subscription from https://developers.kite.trade

### Installation

```bash
# Clone the repo
git clone <repo-url> doolsh-trading-ai
cd doolsh-trading-ai

# Run the automated setup
chmod +x scripts/setup_userland.sh
./scripts/setup_userland.sh
```

This installs Python, creates a virtual environment, installs all dependencies, and seeds the admin user.

### Configuration

```bash
# Edit .env with your Kite credentials
nano .env
```

Required fields:
```
KITE_API_KEY=your-kite-api-key
KITE_API_SECRET=your-kite-api-secret
KITE_USER_ID=AB1234
KITE_PASSWORD=your-kite-password
KITE_TOTP_SECRET=your-base32-totp-secret
```

**Important:** Keep `TRADING_MODE=paper` until you've thoroughly tested your strategies.

### Start the Server

```bash
source .venv/bin/activate
./scripts/start.sh
# or manually:
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

---

## API Usage

### 1. Login to the API

```bash
# Login (get JWT token)
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"Admin@12345"}'

export TOKEN="eyJ..."
```

### 2. Connect to Zerodha Kite

```bash
# Auto-login (headless TOTP — no browser needed)
curl -X POST http://localhost:8000/api/v1/kite/auto-login \
  -H "Authorization: Bearer $TOKEN"

# Check status
curl http://localhost:8000/api/v1/kite/status \
  -H "Authorization: Bearer $TOKEN"

# Check margins / funds
curl http://localhost:8000/api/v1/kite/margins \
  -H "Authorization: Bearer $TOKEN"
```

### 3. Train a Model

```bash
curl -X POST http://localhost:8000/api/v1/trading/train \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"RELIANCE","model_type":"rf","horizon":5,"version":"v1"}'
```

### 4. Generate Predictions

```bash
curl -X POST http://localhost:8000/api/v1/trading/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"RELIANCE","model_type":"rf","model_version":"v1"}'
```

### 5. Place a Manual Order

```bash
# Paper order (or live, depending on TRADING_MODE)
curl -X POST http://localhost:8000/api/v1/orders/place \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"RELIANCE","side":"BUY","quantity":1,"order_type":"MARKET","price":2500}'
```

### 6. Enable Auto-Trading

```bash
# Enable the automated trading loop
curl -X POST http://localhost:8000/api/v1/live/enable \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_type":"rf","model_version":"v1","enabled":true}'

# Check live status
curl http://localhost:8000/api/v1/live/status \
  -H "Authorization: Bearer $TOKEN"

# Check risk state
curl http://localhost:8000/api/v1/live/risk \
  -H "Authorization: Bearer $TOKEN"

# Run one manual cycle
curl -X POST "http://localhost:8000/api/v1/live/cycle?model_type=rf&model_version=v1" \
  -H "Authorization: Bearer $TOKEN"

# Disable auto-trading
curl -X POST http://localhost:8000/api/v1/live/disable \
  -H "Authorization: Bearer $TOKEN"
```

### 7. Run a Backtest

```bash
curl -X POST http://localhost:8000/api/v1/trading/backtest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"RELIANCE","model_type":"rf","model_version":"v1","initial_capital":100000}'
```

---

## Trading Modes

| Mode | Orders | Risk | Use Case |
|------|--------|------|----------|
| `paper` | Simulated (logged to DB) | Full risk checks | Testing and development |
| `live` | **Real orders on Zerodha** | Full risk checks | Production trading |

Switch in `.env`:
```
TRADING_MODE=live   # BE VERY CAREFUL
```

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

---

## TOTP Auto-Login Setup

1. Enable TOTP 2FA on your Zerodha account (Kite → Settings → Security)
2. When scanning the QR code, also copy the **secret key** (Base32 string)
3. Add it to `.env` as `KITE_TOTP_SECRET=JBSWY3DPEHPK3PXP...`
4. The system will automatically generate TOTP codes for headless login

---

## VPS Deployment (Alternative to Userland)

```bash
# On Ubuntu 22.04+
sudo apt install python3 python3-pip python3-venv git
git clone <repo-url> /opt/doolsh-trading-ai
cd /opt/doolsh-trading-ai
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
python -m scripts.seed_admin
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For persistent running, use a systemd service or `tmux`/`screen`.

---

## Safety Checklist

- [ ] Start with `TRADING_MODE=paper` and test thoroughly
- [ ] Set conservative risk limits (`MAX_DAILY_LOSS`, `STOP_LOSS_PCT`)
- [ ] Change default admin password immediately
- [ ] Never commit `.env` to git
- [ ] Monitor the system — don't leave live trading unattended initially
- [ ] Use a separate Zerodha account for bot trading
- [ ] Keep `MIN_CONFIDENCE_THRESHOLD` high (0.65+)

---

## License

MIT

---

**For educational purposes only. Not financial advice. Use at your own risk.**

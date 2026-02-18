# Doolsh Trading AI

AI-powered market analysis platform for signal generation, backtesting, and strategy comparison.

> **Disclaimer:** This software is for **educational and research purposes only**. It does **not** constitute financial advice, and should **not** be used for real-money trading decisions. Past performance of any model or strategy does not guarantee future results. Use at your own risk.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT (curl / UI)                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │  HTTPS
┌──────────────────────────▼──────────────────────────────────────┐
│                     FastAPI Application                         │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ Auth API │  │Trading API│  │Strategy   │  │  Admin API   │  │
│  │ /auth/*  │  │/trading/* │  │/strategies│  │  /admin/*    │  │
│  └────┬─────┘  └─────┬─────┘  └─────┬─────┘  └──────┬───────┘  │
│       │              │              │               │           │
│  ┌────▼──────────────▼──────────────▼───────────────▼────────┐  │
│  │                   Service Layer                           │  │
│  │  Market Data  │  Signal Generator  │  Strategy Engine     │  │
│  └───────┬───────────────┬────────────────────┬──────────────┘  │
│          │               │                    │                 │
│  ┌───────▼───────┐ ┌─────▼─────────┐ ┌───────▼──────────────┐  │
│  │ ML Pipeline   │ │  Backtesting  │ │  Celery Workers      │  │
│  │ RF │ LSTM     │ │  Engine       │ │  (background train)  │  │
│  └───────┬───────┘ └───────────────┘ └───────────────────────┘  │
└──────────┼──────────────────────────────────────────────────────┘
           │
┌──────────▼─────────┐  ┌────────────┐  ┌───────────────────────┐
│  PostgreSQL 16     │  │  Redis 7   │  │ Alpha Vantage API     │
│  (users, signals,  │  │  (cache,   │  │ (market data source)  │
│   trades, models)  │  │   broker)  │  │                       │
└────────────────────┘  └────────────┘  └───────────────────────┘
```

---

## Features

- **Market Data Integration** — Alpha Vantage daily and intraday OHLCV with Redis caching
- **Feature Engineering** — RSI, MACD, Bollinger Bands, ATR, OBV, moving averages, volatility, risk scores
- **ML Model Training** — Random Forest and LSTM classifiers with time-series cross-validation
- **Signal Generation** — BUY / SELL / HOLD with confidence and risk scoring
- **Backtesting Engine** — Portfolio simulation with slippage, commission, stop-loss, take-profit, equity curve, Sharpe/Sortino ratios, max drawdown
- **Strategy System** — Pluggable strategies (MACD crossover, RSI mean reversion, dual MA crossover)
- **REST API** — JWT authentication, role-based access, rate limiting
- **Admin Dashboard** — User management, system metrics, API key provisioning
- **Background Tasks** — Celery workers for async model training and data refresh
- **Dockerized** — Single-command deployment with PostgreSQL, Redis, API, and workers

---

## Project Structure

```
doolsh-trading-ai/
├── app/
│   ├── api/             # FastAPI route modules
│   │   ├── auth.py      # Register, login, refresh, profile
│   │   ├── trading.py   # /train, /predict, /backtest, /metrics
│   │   ├── strategies.py# Strategy CRUD
│   │   └── admin.py     # Admin-only endpoints
│   ├── core/            # Configuration, DB, security, Redis, Celery
│   ├── models/          # SQLAlchemy ORM models
│   ├── schemas/         # Pydantic request/response schemas
│   ├── services/        # Business logic (market data, signals, tasks)
│   ├── strategies/      # Pluggable strategy implementations
│   └── main.py          # FastAPI app entry point
├── ml/
│   ├── features/        # Feature engineering pipeline
│   ├── training/        # Random Forest + LSTM trainers
│   └── saved_models/    # Persisted model artifacts
├── backtesting/
│   └── engine.py        # Portfolio simulation engine
├── tests/               # pytest test suite
├── scripts/             # CLI utilities (seed admin, run backtest)
├── data/                # Market data storage
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── alembic.ini
└── .env.example
```

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- An Alpha Vantage API key (free tier at https://www.alphavantage.co/support/#api-key)

### 1. Clone and configure

```bash
git clone <repo-url> && cd doolsh-trading-ai
cp .env.example .env
# Edit .env — set SECRET_KEY, JWT_SECRET_KEY, POSTGRES_PASSWORD, ALPHA_VANTAGE_API_KEY
```

### 2. Start all services

```bash
docker compose up -d --build
```

This launches PostgreSQL, Redis, the FastAPI server on `:8000`, a Celery worker, and a Celery beat scheduler.

### 3. Seed an admin user

```bash
docker compose exec api python -m scripts.seed_admin
```

Default credentials: `admin` / `Admin@12345` — change immediately after first login.

### 4. Verify

```bash
curl http://localhost:8000/health
```

---

## API Documentation

Interactive docs are available at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### Authentication

```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","username":"trader1","password":"MyP@ss1234"}'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"trader1","password":"MyP@ss1234"}'
# → {"access_token":"eyJ...","refresh_token":"eyJ...","token_type":"bearer"}

# Use the access_token for all subsequent requests
export TOKEN="eyJ..."
```

### Train a model

```bash
curl -X POST http://localhost:8000/api/v1/trading/train \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "model_type": "rf",
    "horizon": 5,
    "version": "v1",
    "hyperparameters": {"n_estimators": 200, "max_depth": 12}
  }'
```

### Generate predictions

```bash
curl -X POST http://localhost:8000/api/v1/trading/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "model_type": "rf", "model_version": "v1"}'
```

### Run a backtest

```bash
curl -X POST http://localhost:8000/api/v1/trading/backtest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "model_type": "rf",
    "model_version": "v1",
    "initial_capital": 100000,
    "commission_rate": 0.001,
    "slippage_bps": 5
  }'
```

### List saved models

```bash
curl http://localhost:8000/api/v1/trading/metrics \
  -H "Authorization: Bearer $TOKEN"
```

### Admin dashboard

```bash
curl http://localhost:8000/api/v1/admin/dashboard \
  -H "Authorization: Bearer $TOKEN"
```

---

## Running Tests

```bash
# With Docker
docker compose exec api pytest tests/ -v

# Locally (needs PostgreSQL + Redis running)
pip install -r requirements.txt
pytest tests/ -v
```

---

## Production Deployment

### VPS Deployment Steps

1. **Provision a VPS** (Ubuntu 22.04+ recommended, minimum 2 vCPU / 4 GB RAM).

2. **Install Docker and Docker Compose:**
   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER
   ```

3. **Clone the repository and configure `.env`:**
   ```bash
   git clone <repo-url> /opt/doolsh-trading-ai
   cd /opt/doolsh-trading-ai
   cp .env.example .env
   # Generate strong secrets:
   #   SECRET_KEY=$(openssl rand -hex 32)
   #   JWT_SECRET_KEY=$(openssl rand -hex 32)
   #   POSTGRES_PASSWORD=$(openssl rand -hex 16)
   ```

4. **Launch:**
   ```bash
   docker compose -f docker-compose.yml up -d --build
   ```

5. **Seed admin and verify:**
   ```bash
   docker compose exec api python -m scripts.seed_admin
   curl http://localhost:8000/health
   ```

### Connecting a Domain (Nginx reverse proxy)

```nginx
server {
    listen 80;
    server_name trading.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Then enable HTTPS with Certbot:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d trading.yourdomain.com
```

### Production Checklist

- [ ] Set `DEBUG=false` and `APP_ENV=production` in `.env`
- [ ] Use strong, unique values for `SECRET_KEY`, `JWT_SECRET_KEY`, `POSTGRES_PASSWORD`
- [ ] Set `CORS_ORIGINS` to your actual frontend domain
- [ ] Configure firewall: only expose ports 80/443
- [ ] Set up log rotation and monitoring
- [ ] Schedule database backups
- [ ] Change default admin password immediately after seeding

---

## License

MIT

---

**For educational purposes only. Not financial advice.**

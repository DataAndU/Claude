#!/usr/bin/env bash
# =============================================================================
# DOOLSH TRADING AI — One Command Setup & Run
# =============================================================================
# Usage:
#   bash run.sh          → Terminal dashboard (works in Userland directly)
#   bash run.sh web      → Web dashboard (open in phone browser)
# =============================================================================
set -e

G='\033[0;32m'; Y='\033[1;33m'; C='\033[0;36m'; R='\033[0;31m'; N='\033[0m'
ok(){ echo -e "${G}[OK]${N} $1"; }
info(){ echo -e "${C}[..]${N} $1"; }
warn(){ echo -e "${Y}[!!]${N} $1"; }

cd "$(dirname "$0")"

echo ""
echo "======================================================="
echo "   DOOLSH TRADING AI"
echo "   F&O Options + Stocks | Intraday + BTST"
echo "======================================================="
echo ""

# ---- Find Python ----
PY=""
for p in python3 python3.11 python3.10 python3.9; do
    command -v "$p" &>/dev/null && PY="$p" && break
done
if [ -z "$PY" ]; then
    info "Installing Python..."
    apt-get update -qq 2>/dev/null || true
    apt-get install -y -qq python3 python3-pip python3-venv 2>/dev/null || true
    command -v python3 &>/dev/null && PY="python3"
fi
[ -z "$PY" ] && echo "ERROR: python3 not found" && exit 1
ok "Python: $($PY --version)"

# ---- Virtual environment ----
if [ ! -d .venv ]; then
    info "Creating virtual environment..."
    $PY -m venv .venv 2>/dev/null || {
        apt-get install -y -qq python3-venv 2>/dev/null || true
        $PY -m venv .venv
    }
fi
source .venv/bin/activate
ok "Venv activated"

# ---- Install deps ----
if [ ! -f .venv/.ok ]; then
    info "Installing dependencies (first time only, takes a few minutes)..."
    pip install --upgrade pip setuptools wheel -q 2>/dev/null

    # Core deps only — no torch, no heavy ML
    pip install \
        kiteconnect==5.0.1 \
        pyotp==2.9.0 \
        httpx==0.28.1 \
        pandas \
        numpy \
        scikit-learn \
        "bcrypt>=4.0,<5" \
        fastapi \
        uvicorn \
        pydantic-settings \
        "sqlalchemy[asyncio]" \
        aiosqlite \
        PyJWT \
        "passlib[bcrypt]" \
        APScheduler \
        -q 2>&1 | tail -3

    touch .venv/.ok
    ok "Dependencies installed"
else
    ok "Dependencies ready"
fi

# ---- Create .env ----
if [ ! -f .env ]; then
    cat > .env << 'ENVEOF'
APP_NAME=doolsh-trading-ai
APP_ENV=production
DEBUG=false
SECRET_KEY=d00lsh-tr4d1ng-s3cr3t-k3y-2025
API_VERSION=v1
DB_PATH=data/doolsh.db
KITE_API_KEY=8ar6acbfss9x2dxx
KITE_API_SECRET=h55ppnhp2gig34z3ko4yxynzpxd8qymr
KITE_USER_ID=VQ4551
KITE_PASSWORD=Prince@2810
KITE_TOTP_SECRET=YGKGPI3TSTRKMHFRQKIJ5ELJI77E2DUO
TRADING_MODE=paper
TRADING_EXCHANGE=NSE
TRADING_PRODUCT=MIS
DEFAULT_QUANTITY=1
WATCHLIST=["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","SBIN","BAJFINANCE","ITC","HINDUNILVR","KOTAKBANK","TATAMOTORS","MARUTI","AXISBANK","LT","SUNPHARMA","TITAN","ADANIENT","BHARTIARTL","WIPRO","HCLTECH"]
MARKET_OPEN_HOUR=9
MARKET_OPEN_MINUTE=15
MARKET_CLOSE_HOUR=15
MARKET_CLOSE_MINUTE=15
AUTO_SQUARE_OFF_MINUTE=10
MAX_DAILY_LOSS=5000
MAX_POSITION_VALUE=100000
MAX_OPEN_POSITIONS=5
STOP_LOSS_PCT=0.02
TAKE_PROFIT_PCT=0.04
MAX_TRADE_COUNT_PER_DAY=20
MIN_CONFIDENCE_THRESHOLD=0.60
JWT_SECRET_KEY=d00lsh-jwt-s3cr3t-k3y
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440
MODEL_SAVE_DIR=ml/saved_models
DEFAULT_TRAIN_TEST_SPLIT=0.8
CROSS_VALIDATION_FOLDS=5
SCHEDULER_INTERVAL_SECONDS=60
LOG_LEVEL=INFO
LOG_FORMAT=text
CORS_ORIGINS=["*"]
OPTIONS_ENABLED=true
OPTIONS_LOT_SIZE=1
BTST_ENABLED=true
BTST_PRODUCT=NRML
ENVEOF
    ok "Config created with your Kite credentials"
fi

# ---- Create directories ----
mkdir -p data ml/saved_models logs

# ---- Seed DB ----
if [ ! -f data/doolsh.db ]; then
    info "Initializing database..."
    python -m scripts.seed_admin 2>/dev/null || true
    ok "Database ready"
fi

echo ""
echo "======================================================="
echo -e "${G}   READY!${N}"
echo "======================================================="
echo ""

MODE="${1:-terminal}"

if [ "$MODE" = "web" ]; then
    # ---- Web dashboard mode ----
    DEVICE_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "")
    if [ -z "$DEVICE_IP" ]; then
        DEVICE_IP=$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K\S+' || echo "127.0.0.1")
    fi

    echo "  Starting WEB DASHBOARD on port 8000..."
    echo ""
    if [ -n "$DEVICE_IP" ] && [ "$DEVICE_IP" != "127.0.0.1" ]; then
        echo -e "  Open in browser: ${G}http://${DEVICE_IP}:8000${N}"
    fi
    echo -e "  Or try:           ${C}http://localhost:8000${N}"
    echo ""
    echo "  Login: admin / Admin@12345"
    echo "  Mode:  PAPER (safe, no real trades)"
    echo ""
    echo "  Press Ctrl+C to stop"
    echo "======================================================="
    echo ""
    exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
else
    # ---- Terminal dashboard mode (default) ----
    echo "  Launching TERMINAL DASHBOARD..."
    echo ""
    echo "  Mode:  PAPER (safe, no real trades)"
    echo "  Tip:   Run 'bash run.sh web' for browser dashboard"
    echo "======================================================="
    echo ""
    exec python trade.py
fi

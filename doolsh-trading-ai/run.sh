#!/usr/bin/env bash
# =============================================================================
# DOOLSH TRADING AI — One Command Launch
# =============================================================================
# Just run: bash run.sh
# That's it. Everything else is automatic.
# =============================================================================
set -e

G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; C='\033[0;36m'; N='\033[0m'
ok()   { echo -e "${G}[OK]${N} $1"; }
warn() { echo -e "${Y}[!!]${N} $1"; }
err()  { echo -e "${R}[ERR]${N} $1"; }
info() { echo -e "${C}[..]${N} $1"; }

echo ""
echo "======================================================="
echo "   DOOLSH TRADING AI"
echo "   F&O Options + Stocks | Intraday + BTST"
echo "   Automated Trading for Zerodha Kite"
echo "======================================================="
echo ""

# ---- CD to project root ----
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

# ---- Create .env with credentials if missing ----
if [ ! -f .env ]; then
    info "Creating .env configuration..."
    cat > .env << 'ENVEOF'
APP_NAME=doolsh-trading-ai
APP_ENV=production
DEBUG=false
SECRET_KEY=d00lsh-tr4d1ng-s3cr3t-k3y-ch4ng3-m3-1n-pr0d
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
CORS_ORIGINS=["http://localhost:3000","http://localhost:8000","*"]
ENVEOF
    ok "Configuration created with your Kite credentials"
else
    ok "Configuration already exists"
fi

# ---- Find Python ----
PYTHON=""
for p in python3 python3.11 python3.10 python3.9; do
    if command -v "$p" &>/dev/null; then PYTHON="$p"; break; fi
done
if [ -z "$PYTHON" ]; then
    info "Installing Python..."
    if command -v apt-get &>/dev/null; then
        apt-get update -qq 2>/dev/null || true
        apt-get install -y -qq python3 python3-pip python3-venv python3-dev \
            build-essential libffi-dev libssl-dev 2>/dev/null || true
    elif command -v pkg &>/dev/null; then
        pkg install -y python3 2>/dev/null || true
    fi
    for p in python3 python3.11 python3.10; do
        if command -v "$p" &>/dev/null; then PYTHON="$p"; break; fi
    done
fi
if [ -z "$PYTHON" ]; then err "Python3 not found. Install python3 first."; exit 1; fi
ok "Python: $($PYTHON --version)"

# ---- Virtual environment ----
if [ ! -d .venv ]; then
    info "Creating virtual environment..."
    $PYTHON -m venv .venv 2>/dev/null || {
        info "Installing venv package..."
        apt-get install -y -qq python3-venv 2>/dev/null || true
        $PYTHON -m venv .venv
    }
    ok "Virtual environment created"
fi
source .venv/bin/activate
ok "Virtual environment activated"

# ---- Install deps ----
if [ ! -f .venv/.deps_installed ]; then
    info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel -q 2>/dev/null

    info "Installing dependencies (this takes a few minutes first time)..."

    # Try torch CPU-only first (skip if fails — RF model still works)
    pip install torch --index-url https://download.pytorch.org/whl/cpu -q 2>/dev/null || {
        pip install torch -q 2>/dev/null || warn "PyTorch skipped — LSTM unavailable, RF works fine"
    }

    pip install -r requirements.txt -q 2>/dev/null || {
        warn "Retrying with --no-cache-dir..."
        pip install -r requirements.txt --no-cache-dir 2>&1 | tail -3
    }

    touch .venv/.deps_installed
    ok "All dependencies installed"
else
    ok "Dependencies already installed"
fi

# ---- Create directories ----
mkdir -p data ml/saved_models logs

# ---- Seed database ----
if [ ! -f data/doolsh.db ]; then
    info "Initializing database..."
    python -m scripts.seed_admin 2>&1 || warn "DB seed had an issue"
    ok "Database ready (admin / Admin@12345)"
else
    ok "Database already exists"
fi

# ---- Launch ----
echo ""
echo "======================================================="
echo -e "${G}   READY TO TRADE!${N}"
echo "======================================================="
echo ""
echo "  Dashboard:  http://localhost:8000"
echo "  Login:      admin / Admin@12345"
echo ""
echo "  Quick start:"
echo "    1. Open http://localhost:8000 in browser"
echo "    2. Login → Click 'Connect Kite' → Click 'Scan'"
echo "    3. See F&O signals for options + stocks"
echo "    4. Switch to LIVE mode for real trading"
echo ""
echo "  Trading modes:"
echo "    PAPER = simulated (safe, no real money)"
echo "    LIVE  = real trades on Zerodha (be careful!)"
echo ""
echo "  Press Ctrl+C to stop the server"
echo "======================================================="
echo ""

exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

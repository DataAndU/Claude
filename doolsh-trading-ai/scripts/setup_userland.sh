#!/usr/bin/env bash
# =============================================================================
# Doolsh Trading AI — Userland One-Command Installer
# =============================================================================
# Installs everything and launches the dashboard on localhost:8000
#
# Usage:
#   chmod +x scripts/setup_userland.sh
#   ./scripts/setup_userland.sh
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!!]${NC} $1"; }
err()   { echo -e "${RED}[ERR]${NC} $1"; }
info()  { echo -e "${CYAN}[..]${NC} $1"; }

echo ""
echo "======================================================="
echo "   DOOLSH TRADING AI — Automated Installer"
echo "   F&O Intraday Short-Sell Analysis + Auto Trading"
echo "======================================================="
echo ""

# ---- Detect environment ----
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
info "Project directory: $PROJECT_DIR"

IS_USERLAND=false
if [ -f /support/.userland ] || [ -d /data/data/tech.ula ]; then
    IS_USERLAND=true
    info "Detected Userland environment"
fi

# ---- 1. System packages ----
info "Step 1/7 — Installing system packages..."

install_packages() {
    if command -v apt-get &>/dev/null; then
        apt-get update -qq 2>/dev/null || true
        apt-get install -y -qq \
            python3 python3-pip python3-venv python3-dev \
            git curl build-essential libffi-dev libssl-dev \
            2>/dev/null || {
            warn "Some packages may already be installed"
        }
    elif command -v apk &>/dev/null; then
        apk add --no-cache python3 py3-pip py3-virtualenv git curl \
            build-base libffi-dev openssl-dev 2>/dev/null || true
    elif command -v pkg &>/dev/null; then
        pkg install -y python3 git curl 2>/dev/null || true
    else
        warn "Unknown package manager — ensure python3, pip, venv are installed"
    fi
}

install_packages
log "System packages ready"

# ---- 2. Python version check ----
info "Step 2/7 — Checking Python..."

PYTHON=""
for p in python3.11 python3.10 python3.9 python3; do
    if command -v "$p" &>/dev/null; then
        PYTHON="$p"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    err "Python 3 not found. Install python3 and re-run."
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1 | grep -oP '\d+\.\d+' || $PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
    err "Python 3.9+ required. Found: $PYTHON ($PY_VERSION)"
    exit 1
fi

log "Using $PYTHON ($PY_VERSION)"

# ---- 3. Virtual environment ----
info "Step 3/7 — Setting up virtual environment..."

if [ ! -d .venv ]; then
    $PYTHON -m venv .venv
    log "Virtual environment created"
else
    log "Virtual environment already exists"
fi

# Activate
# shellcheck source=/dev/null
source .venv/bin/activate

# ---- 4. Upgrade pip ----
info "Step 4/7 — Upgrading pip..."
pip install --upgrade pip setuptools wheel -q 2>/dev/null
log "pip upgraded"

# ---- 5. Install Python packages ----
info "Step 5/7 — Installing Python dependencies (this may take a few minutes)..."

# PyTorch CPU-only (smaller, works on ARM)
info "Installing PyTorch (CPU-only)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu -q 2>/dev/null || {
    warn "CPU wheel not available for this arch — trying default torch"
    pip install torch -q 2>/dev/null || {
        warn "PyTorch install failed — LSTM model won't work, RF will still work"
    }
}

# Install remaining deps
pip install -r requirements.txt -q 2>/dev/null || {
    warn "Retrying with --no-cache-dir..."
    pip install -r requirements.txt --no-cache-dir 2>&1 | tail -5
}
log "Python dependencies installed"

# ---- 6. Configure environment ----
info "Step 6/7 — Configuring environment..."

mkdir -p data ml/saved_models logs

if [ ! -f .env ]; then
    cp .env.example .env
    warn "Created .env from template"
    warn "Edit .env with your Kite API credentials for live trading"
    warn "Paper mode (default) works without credentials"
else
    log ".env already exists"
fi

# ---- 7. Initialize database and seed admin ----
info "Step 7/7 — Initializing database..."

$PYTHON -m scripts.seed_admin 2>&1 || {
    warn "Seed script had an issue — database may already be initialized"
}
log "Database ready"

# ---- Done — Launch ----
echo ""
echo "======================================================="
echo -e "${GREEN}   INSTALLATION COMPLETE!${NC}"
echo "======================================================="
echo ""
echo "  Dashboard URL:  http://localhost:8000"
echo "  API Docs:       http://localhost:8000/docs"
echo ""
echo "  Default login:  admin / Admin@12345"
echo "  Mode:           PAPER (safe — no real trades)"
echo ""
echo "  How to use:"
echo "    1. Open http://localhost:8000 in your browser"
echo "    2. Login with admin / Admin@12345"
echo "    3. Click 'Scan F&O Symbols' to see sell signals"
echo "    4. For live trading: edit .env with Kite credentials"
echo "       then click 'Connect Kite' and 'Enable Auto-Trade'"
echo ""
echo "  To restart later:"
echo "    cd $PROJECT_DIR"
echo "    source .venv/bin/activate"
echo "    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "======================================================="
echo -e "${CYAN}   Starting server on http://localhost:8000 ...${NC}"
echo "   Press Ctrl+C to stop"
echo "======================================================="
echo ""

exec $PYTHON -m uvicorn app.main:app --host 0.0.0.0 --port 8000

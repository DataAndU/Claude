#!/usr/bin/env bash
# install.sh — One-time setup for AI Auto-Trading System
# Works on UserLAnd (Ubuntu on Android), Debian, Ubuntu

set -e

echo "============================================="
echo " AI Auto-Trading System — Installer"
echo "============================================="

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# Check Python
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "[!] Python not found. Installing..."
    apt-get update -qq && apt-get install -y -qq python3 python3-pip python3-venv
    PY=python3
fi

echo "[+] Using $($PY --version)"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "[+] Creating virtual environment..."
    $PY -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel --quiet

# Fix /tmp permission issues common in UserLand/restricted environments
export TMPDIR="${ROOT}/.tmp_pip"
mkdir -p "$TMPDIR"

# Install requirements
echo "[+] Installing Python packages..."
pip install -r requirements.txt --quiet 2>/dev/null || {
    echo "[!] Some packages failed. Installing core packages individually..."

    # Core data + framework packages (most reliable)
    pip install pandas numpy sqlalchemy pyyaml python-dotenv requests --quiet
    pip install fastapi uvicorn pydantic pydantic-settings --quiet
    pip install aiosqlite --quiet 2>/dev/null || echo "[!] aiosqlite install failed (optional)"

    # Technical analysis — ta can fail on restricted /tmp, try workarounds
    pip install ta --quiet 2>/dev/null || {
        echo "[!] ta package failed with default settings, retrying with --no-build-isolation..."
        pip install ta --no-build-isolation --quiet 2>/dev/null || {
            echo "[!] ta install failed — will use pandas-ta as fallback"
            pip install pandas-ta --quiet 2>/dev/null || echo "[!] pandas-ta also failed (feature engine will use built-in indicators)"
        }
    }

    # Exchange APIs
    pip install ccxt --quiet 2>/dev/null || echo "[!] ccxt install failed (exchange data unavailable, synthetic mode only)"
    pip install yfinance --quiet 2>/dev/null || echo "[!] yfinance install failed (optional)"

    # ML packages (may fail on ARM/low memory)
    pip install scikit-learn --quiet 2>/dev/null || echo "[!] scikit-learn install failed (ML features unavailable)"
    pip install xgboost --quiet 2>/dev/null || echo "[!] XGBoost install failed (optional)"
    pip install lightgbm --quiet 2>/dev/null || echo "[!] LightGBM install failed (optional)"
    pip install joblib --quiet 2>/dev/null || echo "[!] joblib install failed (optional)"

    # Scheduler
    pip install apscheduler --quiet 2>/dev/null || echo "[!] APScheduler install failed (optional)"

    # Dashboard (heavy dependency, optional)
    pip install streamlit --quiet 2>/dev/null || echo "[!] Streamlit install failed (dashboard unavailable)"

    # Deep learning (optional, likely to fail on low-memory ARM)
    pip install torch --quiet 2>/dev/null || echo "[!] PyTorch install failed (LSTM/Transformer unavailable — OK for basic trading)"

    echo "[+] Core packages installed. Some optional packages may have been skipped."
}

# Cleanup temp dir
rm -rf "$TMPDIR"

# Create directories
echo "[+] Creating directories..."
mkdir -p database/models
mkdir -p config
mkdir -p logs

# Copy config template if not exists
if [ ! -f "config/config.yaml" ]; then
    echo "[+] Config template already exists"
fi

# Initialize database
echo "[+] Initializing database..."
$PY -c "from core.database import init_db; init_db()" 2>/dev/null || echo "[!] DB init deferred to first run"

echo ""
echo "============================================="
echo " Installation complete!"
echo " Run: ./start.sh"
echo "============================================="

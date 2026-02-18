#!/usr/bin/env bash
# install.sh — One-time setup for KiteAI Auto-Trading System
# Works on UserLAnd (Ubuntu on Android), Debian, Ubuntu

set -e

echo "============================================="
echo " KiteAI — Zerodha AI Auto-Trader Installer"
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
pip install --upgrade pip --quiet

# Install requirements
echo "[+] Installing Python packages..."
pip install -r requirements.txt --quiet 2>/dev/null || {
    echo "[!] Some packages failed. Installing core packages only..."
    pip install kiteconnect pyotp pandas numpy yfinance ta scikit-learn fastapi uvicorn streamlit sqlalchemy httpx pyyaml python-dotenv --quiet
    # Optional packages (may fail on ARM/low memory)
    pip install xgboost --quiet 2>/dev/null || echo "[!] XGBoost install failed (optional)"
    pip install apscheduler --quiet 2>/dev/null || echo "[!] APScheduler install failed (optional)"
}

# Create directories
echo "[+] Creating directories..."
mkdir -p data/models
mkdir -p data/db
mkdir -p config
mkdir -p logs

# Create .env from template if not exists
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "[+] Creating .env from template..."
    cp .env.example .env
    echo "[!] Edit .env with your Zerodha Kite credentials before running!"
fi

# Initialize database
echo "[+] Initializing database..."
$PY -c "from core.database import get_engine; get_engine()" 2>/dev/null || echo "[!] DB init deferred to first run"

echo ""
echo "============================================="
echo " Installation complete!"
echo ""
echo " Next steps:"
echo "   1. Edit .env with your Zerodha Kite credentials"
echo "   2. Run: ./start.sh"
echo "============================================="

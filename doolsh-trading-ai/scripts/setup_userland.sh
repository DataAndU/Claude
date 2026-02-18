#!/usr/bin/env bash
# =============================================================================
# Doolsh Trading AI — Userland Setup Script
# =============================================================================
# Run this inside Userland (Ubuntu / Alpine) on Android.
#
# Usage:
#   chmod +x scripts/setup_userland.sh
#   ./scripts/setup_userland.sh
# =============================================================================

set -euo pipefail

echo "================================================"
echo " Doolsh Trading AI — Userland Setup"
echo "================================================"

# 1. System packages
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git curl build-essential \
    libffi-dev libssl-dev 2>/dev/null || {
    # Alpine fallback
    apk add --no-cache python3 py3-pip py3-virtualenv git curl build-base \
        libffi-dev openssl-dev
}

# 2. Create virtual environment
echo "[2/6] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip
echo "[3/6] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# 4. Install Python packages
echo "[4/6] Installing Python dependencies..."
# Install PyTorch CPU-only first (smaller, works on ARM)
pip install torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
    pip install torch

pip install -r requirements.txt

# 5. Configure environment
echo "[5/6] Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from template — edit it with your Kite API credentials!"
else
    echo "  .env already exists — skipping"
fi

mkdir -p data ml/saved_models

# 6. Seed the database
echo "[6/6] Initializing database and seeding admin user..."
python -m scripts.seed_admin

echo ""
echo "================================================"
echo " Setup complete!"
echo "================================================"
echo ""
echo " Next steps:"
echo "   1. Edit .env with your Zerodha Kite API credentials"
echo "   2. Start the server:"
echo "        source .venv/bin/activate"
echo "        python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo "   3. Open http://localhost:8000/docs in your browser"
echo "   4. Login and connect Kite via /api/v1/kite/auto-login"
echo ""
echo " Default admin credentials: admin / Admin@12345"
echo " CHANGE THESE IMMEDIATELY!"
echo ""

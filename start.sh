#!/usr/bin/env bash
# start.sh — One-command launcher for AI Auto-Trading System
# Usage: ./start.sh
# First run will auto-install everything.

set -e

cd "$(dirname "$0")"

echo "============================================="
echo " AI Auto-Trading System"
echo "============================================="

# Auto-install on first run
if [ ! -d "venv" ]; then
    echo "[+] First run — installing dependencies..."
    bash scripts/install.sh
fi

# Activate and launch
bash scripts/start.sh

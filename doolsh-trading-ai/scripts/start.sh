#!/usr/bin/env bash
# Quick start script for Userland
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo "Starting Doolsh Trading AI on port 8000..."
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

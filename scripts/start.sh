#!/usr/bin/env bash
# start.sh (scripts/) — Launch KiteAI API and Dashboard
# Called by root start.sh after setup

set -e

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

source venv/bin/activate

# Load .env if present
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "[+] Starting KiteAI Auto-Trading System..."
echo "    Mode: ${TRADING_MODE:-paper}"
echo ""

# Start FastAPI backend in background
echo "[+] Starting API server on port 8000..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info &
API_PID=$!
echo "    API PID: $API_PID"

# Wait for API to be ready
sleep 3

# Start Streamlit dashboard in background
echo "[+] Starting Dashboard on port 8501..."
python -m streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false &
DASH_PID=$!
echo "    Dashboard PID: $DASH_PID"

echo ""
echo "============================================="
echo " KiteAI Running!"
echo " API:       http://localhost:8000/docs"
echo " Dashboard: http://localhost:8501"
echo " Mode:      ${TRADING_MODE:-paper}"
echo "============================================="
echo ""
echo "Press Ctrl+C to stop all services"

# Trap Ctrl+C to kill both processes
cleanup() {
    echo ""
    echo "[+] Stopping services..."
    kill $API_PID 2>/dev/null
    kill $DASH_PID 2>/dev/null
    echo "[+] All services stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for either process to exit
wait $API_PID $DASH_PID

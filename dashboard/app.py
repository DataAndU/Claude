"""Streamlit dashboard — real-time monitoring UI for the AI auto-trading system.

Displays: system status, live signals, trade history, risk metrics,
equity curves, backtest results, and model management.

Run: streamlit run dashboard/app.py --server.port 8501
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000/api"


def api_get(endpoint: str, params: dict = None):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_post(endpoint: str, json_data: dict = None, params: dict = None):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=json_data, params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Auto-Trader",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("AI Auto-Trader")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Signals", "Trades", "Backtest", "Models", "Risk", "Settings"],
)

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto_refresh:
    time.sleep(30)
    st.rerun()


# ─── Dashboard Page ───────────────────────────────────────────────────────────

if page == "Dashboard":
    st.title("System Dashboard")

    status = api_get("/status")
    if "error" in status:
        st.error(f"API not reachable: {status['error']}")
        st.info("Start the API server first: python -m api.main")
        st.stop()

    # Status cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mode = status.get("mode", "unknown")
        st.metric("Mode", mode.upper())
    with col2:
        risk = status.get("risk", {})
        st.metric("Equity", f"${risk.get('current_equity', 0):,.2f}")
    with col3:
        st.metric("Open Trades", risk.get("open_trades", 0))
    with col4:
        dd = risk.get("drawdown_pct", 0)
        st.metric("Drawdown", f"{dd:.2%}")

    st.divider()

    # Risk overview
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Summary")
        if risk:
            st.write(f"Peak Equity: **${risk.get('peak_equity', 0):,.2f}**")
            st.write(f"Realized PnL: **${risk.get('realized_pnl', 0):,.4f}**")
            st.write(f"Win Rate: **{risk.get('win_rate', 0):.2%}**")
            st.write(f"Wins/Losses: **{risk.get('wins', 0)}/{risk.get('losses', 0)}**")
            st.write(f"Trade Count: **{risk.get('trade_count', 0)}**")

    with col2:
        st.subheader("Open Positions")
        positions = risk.get("open_positions", {})
        if positions:
            for sym, pos in positions.items():
                st.write(f"**{sym}** — {pos['side']} {pos['quantity']:.6f} @ {pos['entry_price']:.4f}")
                st.write(f"  SL: {pos['stop_loss']:.4f} | TP: {pos['take_profit']:.4f}")
        else:
            st.write("No open positions")

    st.divider()

    # Kill switch and auto-trading controls
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Kill Switch")
        ks = status.get("kill_switch", False)
        if ks:
            st.warning("KILL SWITCH IS ON — All trading halted")
        if st.button("Toggle Kill Switch"):
            api_post("/risk/kill-switch", {"enabled": not ks})
            st.rerun()

    with col2:
        st.subheader("Auto-Trading")
        auto_st = api_get("/auto/status")
        running = auto_st.get("running", False)
        st.write(f"Status: **{'Running' if running else 'Stopped'}**")
        if running:
            if st.button("Stop Auto-Trading"):
                api_post("/auto/stop")
                st.rerun()
        else:
            if st.button("Start Auto-Trading"):
                api_post("/auto/start")
                st.rerun()


# ─── Signals Page ─────────────────────────────────────────────────────────────

elif page == "Signals":
    st.title("Signal Generator")

    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Symbol", value="BTC/USDT")
    with col2:
        timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d", "15m"])
    with col3:
        model_type = st.selectbox("Model", ["random_forest", "xgboost", "lstm"])

    if st.button("Generate Signal"):
        with st.spinner("Generating signal..."):
            sig = api_post("/signal", {
                "symbol": symbol, "timeframe": timeframe,
                "model_type": model_type, "limit": 500,
            })

        if "error" in sig:
            st.error(sig["error"])
        else:
            signal_dir = sig.get("signal", "HOLD")
            conf = sig.get("confidence", 0)

            if signal_dir == "BUY":
                st.success(f"**{signal_dir}** — Confidence: {conf:.2%}")
            elif signal_dir == "SELL":
                st.error(f"**{signal_dir}** — Confidence: {conf:.2%}")
            else:
                st.info(f"**{signal_dir}** — {sig.get('reason', 'No action')}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Price", f"${sig.get('price', 0):,.4f}")
                st.metric("RSI", f"{sig.get('rsi', 50):.1f}")
            with col2:
                st.metric("ATR", f"{sig.get('atr', 0):.6f}")
                st.metric("ADX", f"{sig.get('adx', 25):.1f}")
            with col3:
                st.metric("Regime", sig.get("regime", "unknown"))
                st.metric("Volatility", f"{sig.get('volatility', 0):.4f}")

    st.divider()
    st.subheader("Signal History")
    history = api_get("/signals/history", {"limit": 30})
    if isinstance(history, list) and history:
        df = pd.DataFrame(history)
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No signal history yet")


# ─── Trades Page ──────────────────────────────────────────────────────────────

elif page == "Trades":
    st.title("Trade History")

    status_filter = st.selectbox("Filter", ["all", "open", "closed"])
    trades = api_get("/trades", {"status": status_filter, "limit": 100})

    if isinstance(trades, list) and trades:
        df = pd.DataFrame(trades)
        total_pnl = df["pnl"].sum()
        wins = len(df[df["pnl"] > 0])
        losses = len(df[df["pnl"] < 0])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total PnL", f"${total_pnl:,.4f}")
        with col2:
            st.metric("Wins/Losses", f"{wins}/{losses}")
        with col3:
            wr = wins / len(df) if len(df) > 0 else 0
            st.metric("Win Rate", f"{wr:.1%}")

        st.dataframe(df, use_container_width=True)

        # PnL chart
        if "pnl" in df.columns:
            st.subheader("PnL per Trade")
            st.bar_chart(df["pnl"])
    else:
        st.write("No trades yet")

    st.divider()
    st.subheader("Manual Trade Execution")
    col1, col2, col3 = st.columns(3)
    with col1:
        exec_symbol = st.text_input("Exec Symbol", value="BTC/USDT", key="exec_sym")
    with col2:
        exec_side = st.selectbox("Side", ["BUY", "SELL"])
    with col3:
        exec_conf = st.slider("Confidence", 0.5, 1.0, 0.7)

    if st.button("Execute Trade"):
        result = api_post("/trade/execute", {
            "symbol": exec_symbol, "side": exec_side, "confidence": exec_conf,
        })
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"Trade executed: {result}")


# ─── Backtest Page ────────────────────────────────────────────────────────────

elif page == "Backtest":
    st.title("Backtesting")

    col1, col2 = st.columns(2)
    with col1:
        bt_symbol = st.text_input("Symbol", value="BTC/USDT", key="bt_sym")
        bt_timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], key="bt_tf")
    with col2:
        bt_model = st.selectbox("Model", ["random_forest", "xgboost", "lstm"], key="bt_model")
        bt_capital = st.number_input("Capital", value=10000.0, step=1000.0)

    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            result = api_post("/backtest", {
                "symbol": bt_symbol, "timeframe": bt_timeframe,
                "model_type": bt_model, "initial_capital": bt_capital, "limit": 500,
            })

        if "error" in result:
            st.error(result["error"])
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Return", f"{result.get('total_return_pct', 0):.2f}%")
            with col2:
                st.metric("Sharpe", f"{result.get('sharpe_ratio', 0):.4f}")
            with col3:
                st.metric("Max DD", f"{result.get('max_drawdown_pct', 0):.2f}%")
            with col4:
                st.metric("Win Rate", f"{result.get('win_rate', 0):.2%}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("CAGR", f"{result.get('cagr', 0):.2f}%")
            with col2:
                st.metric("Total Trades", result.get("total_trades", 0))
            with col3:
                st.metric("Profit Factor", f"{result.get('profit_factor', 0):.2f}")
            with col4:
                st.metric("Expectancy", f"${result.get('expectancy', 0):.4f}")

            st.write(f"Final Capital: **${result.get('final_capital', 0):,.2f}** (from ${bt_capital:,.2f})")

            # Equity curve
            ec = result.get("equity_curve", [])
            if ec:
                st.subheader("Equity Curve")
                st.line_chart(ec)


# ─── Models Page ──────────────────────────────────────────────────────────────

elif page == "Models":
    st.title("Model Management")

    # Train new model
    st.subheader("Train New Model")
    col1, col2, col3 = st.columns(3)
    with col1:
        tr_symbol = st.text_input("Symbol", value="BTC/USDT", key="tr_sym")
    with col2:
        tr_tf = st.selectbox("Timeframe", ["1h", "4h", "1d"], key="tr_tf")
    with col3:
        tr_model = st.selectbox("Type", ["random_forest", "xgboost", "lstm"], key="tr_model")

    if st.button("Train Model"):
        with st.spinner(f"Training {tr_model}..."):
            result = api_post("/train", {
                "symbol": tr_symbol, "timeframe": tr_tf,
                "model_type": tr_model, "limit": 500,
            })
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"Model trained! Accuracy: {result.get('accuracy', 0):.4f} | F1: {result.get('f1', 0):.4f}")
            fi = result.get("feature_importance", {})
            if fi:
                st.subheader("Feature Importance")
                fi_df = pd.DataFrame({"feature": fi.keys(), "importance": fi.values()})
                fi_df = fi_df.sort_values("importance", ascending=False)
                st.bar_chart(fi_df.set_index("feature"))

    st.divider()
    st.subheader("Saved Models")
    models = api_get("/models")
    if isinstance(models, dict) and models.get("models"):
        df = pd.DataFrame(models["models"])
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No models trained yet")


# ─── Risk Page ────────────────────────────────────────────────────────────────

elif page == "Risk":
    st.title("Risk Management")

    risk = api_get("/risk")
    if "error" in risk:
        st.error(risk["error"])
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Equity", f"${risk.get('current_equity', 0):,.2f}")
        with col2:
            st.metric("Peak", f"${risk.get('peak_equity', 0):,.2f}")
        with col3:
            st.metric("Day PnL", f"${risk.get('realized_pnl', 0):,.4f}")
        with col4:
            st.metric("Drawdown", f"{risk.get('drawdown_pct', 0):.2%}")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Initialize Capital")
            cap = st.number_input("Starting Capital", value=10000.0, step=1000.0, key="risk_cap")
            if st.button("Set Capital"):
                api_post("/risk/init", params={"capital": cap})
                st.rerun()

        with col2:
            st.subheader("Open Positions")
            positions = risk.get("open_positions", {})
            if positions:
                for sym, pos in positions.items():
                    st.write(f"**{sym}** — {pos['side']} {pos['quantity']:.6f} @ {pos['entry_price']:.4f}")
                    if st.button(f"Close {sym}", key=f"close_{sym}"):
                        api_post(f"/trade/close/{sym.replace('/', '-')}")
                        st.rerun()
            else:
                st.write("No open positions")


# ─── Settings Page ────────────────────────────────────────────────────────────

elif page == "Settings":
    st.title("Configuration")

    config = api_get("/config")
    if "error" in config:
        st.error(config["error"])
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Trading")
            st.write(f"Mode: **{config.get('mode', 'paper')}**")
            st.write(f"Exchange: **{config.get('exchange', 'binance')}**")
            st.write(f"Symbols: **{', '.join(config.get('symbols', []))}**")
            st.write(f"Primary TF: **{config.get('primary_timeframe', '1h')}**")

        with col2:
            st.subheader("Risk Limits")
            risk_cfg = config.get("risk", {})
            st.write(f"Max Position: **{risk_cfg.get('max_position_pct', 0):.1%}**")
            st.write(f"ATR SL Mult: **{risk_cfg.get('atr_sl_multiplier', 0)}x**")
            st.write(f"ATR TP Mult: **{risk_cfg.get('atr_tp_multiplier', 0)}x**")
            st.write(f"Max Daily Loss: **{risk_cfg.get('max_daily_loss_pct', 0):.1%}**")
            st.write(f"Max Drawdown: **{risk_cfg.get('max_drawdown_pct', 0):.1%}**")
            st.write(f"Cooldown: **{risk_cfg.get('cooldown_minutes', 0)} min**")
            st.write(f"Max Open Trades: **{risk_cfg.get('max_open_trades', 0)}**")
            st.write(f"Kill Switch: **{risk_cfg.get('kill_switch', False)}**")

        st.subheader("Model")
        model_cfg = config.get("model", {})
        st.write(f"Primary Model: **{model_cfg.get('primary', 'random_forest')}**")
        st.write(f"Fallback Model: **{model_cfg.get('fallback', 'xgboost')}**")
        st.write(f"Min Confidence: **{model_cfg.get('min_confidence', 0):.0%}**")

    st.divider()
    st.info("Edit config/config.yaml to change settings, then restart the API server.")

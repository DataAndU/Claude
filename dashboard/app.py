"""KiteAI Dashboard — Streamlit monitoring UI for Zerodha AI Auto-Trading.

Displays: system status, scanner, signals, trades, risk, backtest, models.
Run: streamlit run dashboard/app.py --server.port 8501
"""

from __future__ import annotations

import json
import time

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="KiteAI - Zerodha Auto Trader",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_get(path: str, params: dict = None):
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, data: dict = None):
    try:
        r = httpx.post(f"{API_BASE}{path}", json=data or {}, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ─── Sidebar ────────────────────────────────────────────────────────────────

st.sidebar.title("KiteAI")
st.sidebar.caption("Personal AI Auto-Trader for Zerodha Kite")

page = st.sidebar.radio("Navigate", [
    "Dashboard",
    "Scanner",
    "Signals",
    "Trades",
    "Backtest",
    "Models",
    "Risk",
    "Settings",
])

# Quick status in sidebar
status = api_get("/api/status")
if status:
    st.sidebar.divider()
    mode_color = "red" if status.get("mode") == "live" else "blue"
    st.sidebar.markdown(f"**Mode:** :{mode_color}[{status.get('mode', 'paper').upper()}]")
    kite_icon = "white_check_mark" if status.get("kite_connected") else "x"
    st.sidebar.markdown(f"**Kite:** :{kite_icon}: {'Connected' if status.get('kite_connected') else 'Not connected'}")
    auto_icon = "arrow_forward" if status.get("auto_trading") else "stop_button"
    st.sidebar.markdown(f"**Auto-Trade:** :{auto_icon}: {'Running' if status.get('auto_trading') else 'Stopped'}")
    pnl = status.get("risk", {}).get("realized_pnl", 0)
    pnl_color = "green" if pnl >= 0 else "red"
    st.sidebar.metric("Day P&L", f"INR {pnl:,.2f}")
    st.sidebar.metric("Trades Today", status.get("risk", {}).get("trade_count", 0))


# ─── Dashboard Page ─────────────────────────────────────────────────────────

if page == "Dashboard":
    st.title("KiteAI Dashboard")

    if not status:
        st.warning("API not reachable. Start the backend with: `python -m uvicorn api.main:app`")
        st.stop()

    # Controls row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if status.get("auto_trading"):
            if st.button("Stop Auto-Trading", type="secondary"):
                api_post("/api/auto/stop")
                st.rerun()
        else:
            if st.button("Start Auto-Trading", type="primary"):
                api_post("/api/auto/start")
                st.rerun()
    with col2:
        if st.button("Run Single Cycle"):
            with st.spinner("Running cycle..."):
                result = api_post("/api/auto/cycle")
                if result:
                    st.json(result)
    with col3:
        if st.button("Kite Auto-Login"):
            with st.spinner("Logging in..."):
                result = api_post("/api/kite/login", {"method": "auto"})
                if result:
                    st.success("Logged in!")
    with col4:
        kill = status.get("risk", {}).get("kill_switch", False)
        if st.button("KILL SWITCH" if not kill else "Disable Kill Switch",
                      type="primary" if not kill else "secondary"):
            api_post(f"/api/risk/kill-switch?enable={str(not kill).lower()}")
            st.rerun()

    st.divider()

    # Metrics
    analytics = status.get("analytics", {})
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Win Rate", f"{analytics.get('win_rate', 0)*100:.1f}%")
    c2.metric("Profit Factor", f"{analytics.get('profit_factor', 0):.2f}")
    c3.metric("Max Drawdown", f"INR {analytics.get('max_drawdown', 0):,.0f}")
    c4.metric("Total Trades", analytics.get("total_trades", 0))
    c5.metric("Avg Win", f"INR {analytics.get('avg_win', 0):,.0f}")
    c6.metric("Avg Loss", f"INR {analytics.get('avg_loss', 0):,.0f}")

    # Regime
    regime = status.get("regime", {})
    if regime:
        st.subheader("Market Regime")
        st.info(f"**{regime.get('regime', 'unknown').replace('_', ' ').title()}** "
                f"(confidence: {regime.get('confidence', 0):.1%})")

    # Recent trade log
    st.subheader("Auto-Trade Log")
    log_data = api_get("/api/auto/log")
    if log_data and log_data.get("log"):
        for entry in reversed(log_data["log"][-20:]):
            action = entry.get("action", "")
            if action == "ORDER":
                st.success(f"[{entry.get('time')}] ORDER: {entry.get('side')} {entry.get('symbol')} x{entry.get('qty')} @ {entry.get('price')}")
            elif action == "CLOSE":
                pnl = entry.get("pnl", 0)
                fn = st.success if pnl > 0 else st.error
                fn(f"[{entry.get('time')}] CLOSE: {entry.get('symbol')} PnL={pnl:+,.2f} ({entry.get('reason')})")
            else:
                st.info(f"[{entry.get('time')}] {action}: {json.dumps({k: v for k, v in entry.items() if k not in ('time', 'action')})}")
    else:
        st.caption("No trades yet. Start auto-trading or run a cycle.")


# ─── Scanner Page ───────────────────────────────────────────────────────────

elif page == "Scanner":
    st.title("Stock Scanner")

    col1, col2 = st.columns(2)
    with col1:
        scan_type = st.selectbox("Scan Type", ["all", "buy", "sell"])
    with col2:
        top_n = st.slider("Top N Results", 3, 20, 10)

    if st.button("Scan Now", type="primary"):
        with st.spinner("Scanning watchlist..."):
            results = api_get(f"/api/scan?scan_type={scan_type}&top_n={top_n}")
            if results:
                for r in results:
                    action = r.get("action", "HOLD")
                    color = "green" if "BUY" in action else ("red" if "SELL" in action else "gray")
                    with st.expander(f"{r['symbol']} — {action} (Score: {r['score']:.0f}, Quality: {r.get('quality', '?')})"):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Price", f"INR {r.get('price', 0):,.2f}")
                        c2.metric("RSI", f"{r.get('rsi', 0):.1f}")
                        c3.metric("ADX", f"{r.get('adx', 0):.1f}")
                        c4.metric("Vol Ratio", f"{r.get('volume_ratio', 0):.2f}x")
                        st.write("**Reasons:**", ", ".join(r.get("reasons", [])))

    st.divider()
    st.subheader("F&O Scanner")
    if st.button("Scan F&O"):
        with st.spinner("Scanning F&O..."):
            fno = api_get("/api/scan/fno?top_n=5")
            if fno:
                for r in fno:
                    st.write(f"**{r['symbol']}** — {r.get('fno_action', '')} | Score: {r['score']:.0f} | {r.get('option_type', '')}")


# ─── Signals Page ───────────────────────────────────────────────────────────

elif page == "Signals":
    st.title("AI Signal Generator")

    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Symbol", "RELIANCE")
    with col2:
        model_type = st.selectbox("Model", ["random_forest", "xgboost", "lstm"])
    with col3:
        days = st.number_input("Lookback Days", 60, 365, 180)

    if st.button("Generate Signal", type="primary"):
        with st.spinner(f"Generating signal for {symbol}..."):
            result = api_post("/api/signal", {"symbol": symbol, "model_type": model_type, "days": days})
            if result:
                signal = result.get("signal", "HOLD")
                conf = result.get("confidence", 0)
                quality = result.get("quality", "?")

                col1, col2, col3 = st.columns(3)
                col1.metric("Signal", signal)
                col2.metric("Confidence", f"{conf:.1%}")
                col3.metric("Quality", quality)

                st.write("**Explanation:**", ", ".join(result.get("explanation", [])))
                st.json(result)


# ─── Trades Page ────────────────────────────────────────────────────────────

elif page == "Trades":
    st.title("Trade History")
    trades = api_get("/api/trades?limit=100")
    if trades:
        import pandas as pd
        df = pd.DataFrame(trades)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            total_pnl = df["pnl"].sum() if "pnl" in df.columns else 0
            st.metric("Total P&L", f"INR {total_pnl:,.2f}")
    else:
        st.info("No trades recorded yet.")


# ─── Backtest Page ──────────────────────────────────────────────────────────

elif page == "Backtest":
    st.title("Backtesting")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bt_symbol = st.text_input("Symbol", "RELIANCE", key="bt_sym")
    with col2:
        bt_model = st.selectbox("Model", ["random_forest", "xgboost"], key="bt_model")
    with col3:
        bt_days = st.number_input("Days", 90, 730, 365, key="bt_days")
    with col4:
        bt_capital = st.number_input("Capital (INR)", 50000, 1000000, 100000, key="bt_cap")

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            result = api_post("/api/backtest", {
                "symbol": bt_symbol, "model_type": bt_model,
                "days": bt_days, "initial_capital": bt_capital,
            })
            if result and "error" not in result:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Return", f"{result.get('total_return_pct', 0):.2f}%")
                c2.metric("Sharpe", f"{result.get('sharpe_ratio', 0):.4f}")
                c3.metric("Max DD", f"{result.get('max_drawdown_pct', 0):.2f}%")
                c4.metric("Win Rate", f"{result.get('win_rate', 0)*100:.1f}%")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("CAGR", f"{result.get('cagr_pct', 0):.2f}%")
                c6.metric("Profit Factor", f"{result.get('profit_factor', 0):.2f}")
                c7.metric("Total Trades", result.get("total_trades", 0))
                c8.metric("Final Equity", f"INR {result.get('final_equity', 0):,.0f}")

                # Equity curve
                eq = result.get("equity_curve", [])
                if eq:
                    import pandas as pd
                    st.line_chart(pd.DataFrame({"equity": eq}))
            elif result:
                st.error(result.get("error", "Backtest failed"))


# ─── Models Page ────────────────────────────────────────────────────────────

elif page == "Models":
    st.title("Model Management")

    col1, col2, col3 = st.columns(3)
    with col1:
        tr_symbol = st.text_input("Symbol", "RELIANCE", key="tr_sym")
    with col2:
        tr_model = st.selectbox("Type", ["random_forest", "xgboost", "lstm"], key="tr_type")
    with col3:
        tr_days = st.number_input("Days", 60, 365, 180, key="tr_days")

    if st.button("Train Model", type="primary"):
        with st.spinner(f"Training {tr_model} for {tr_symbol}..."):
            result = api_post("/api/train", {
                "symbol": tr_symbol, "model_type": tr_model, "days": tr_days,
            })
            if result:
                st.success(f"Trained! Accuracy: {result.get('accuracy', 0):.4f}, F1: {result.get('f1', 0):.4f}")
                if result.get("feature_importance"):
                    st.subheader("Feature Importance")
                    import pandas as pd
                    fi = result["feature_importance"]
                    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15])
                    st.bar_chart(pd.DataFrame({"importance": fi_sorted}))

    st.divider()
    st.subheader("Saved Models")
    models = api_get("/api/models")
    if models:
        for m in models:
            st.write(f"**{m['file']}** — {m['type']} ({m['size_kb']} KB)")


# ─── Risk Page ──────────────────────────────────────────────────────────────

elif page == "Risk":
    st.title("Risk Management")

    risk = api_get("/api/risk")
    if risk:
        analytics = risk.get("analytics", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Day P&L", f"INR {analytics.get('realized_pnl', 0):,.2f}")
        c2.metric("Win Rate", f"{analytics.get('win_rate', 0)*100:.1f}%")
        c3.metric("Profit Factor", f"{analytics.get('profit_factor', 0):.2f}")
        c4.metric("Max Drawdown", f"INR {analytics.get('max_drawdown', 0):,.0f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Total Trades", analytics.get("total_trades", 0))
        c6.metric("Consecutive Wins", analytics.get("consecutive_wins", 0))
        c7.metric("Consecutive Losses", analytics.get("consecutive_losses", 0))
        c8.metric("Open Positions", analytics.get("open_positions", 0))

        positions = risk.get("open_positions", {})
        if positions:
            st.subheader("Open Positions")
            for sym, pos in positions.items():
                st.write(f"**{sym}** — {pos.get('side')} x{pos.get('quantity')} @ {pos.get('entry_price')}")

    # Daily P&L
    st.subheader("Daily P&L History")
    daily = api_get("/api/pnl/daily?days=30")
    if daily:
        import pandas as pd
        df = pd.DataFrame(daily)
        if not df.empty and "realized_pnl" in df.columns:
            st.bar_chart(df.set_index("date")["realized_pnl"])


# ─── Settings Page ──────────────────────────────────────────────────────────

elif page == "Settings":
    st.title("Settings")

    config = api_get("/api/config")
    if config:
        st.json(config)

    st.divider()
    st.subheader("Kite Connection")
    kite = api_get("/api/kite/status")
    if kite:
        if kite.get("logged_in"):
            st.success("Connected to Zerodha Kite")
            margins = api_get("/api/kite/margins")
            if margins:
                st.json(margins)
        else:
            st.warning("Not connected to Kite")
            if st.button("Auto-Login (TOTP)"):
                with st.spinner("Logging in..."):
                    result = api_post("/api/kite/login", {"method": "auto"})
                    if result:
                        st.success("Connected!")
                        st.rerun()

            token = st.text_input("Or paste request_token:")
            if token and st.button("Login with Token"):
                result = api_post("/api/kite/login", {"method": "token", "token": token})
                if result:
                    st.success("Connected!")
                    st.rerun()

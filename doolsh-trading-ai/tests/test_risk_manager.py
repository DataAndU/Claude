"""Tests for the risk manager."""

from app.services.risk_manager import (
    check_risk, compute_position_size, get_risk_state,
    record_trade_close, record_trade_open,
    should_stop_loss, should_take_profit,
)


def test_check_risk_low_confidence():
    result = check_risk("RELIANCE", "BUY", 10, 2500.0, confidence=0.3)
    assert not result["allowed"]
    assert "Confidence" in result["reason"]


def test_check_risk_allowed():
    state = get_risk_state()
    state.reset()
    result = check_risk("RELIANCE", "BUY", 10, 2500.0, confidence=0.8)
    assert result["allowed"]


def test_position_size_cap():
    result = check_risk("TCS", "BUY", 9999, 100.0, confidence=0.9)
    assert result["allowed"]
    assert result["adjusted_qty"] <= 1000  # max_position_value / price


def test_daily_loss_limit():
    state = get_risk_state()
    state.reset()
    state.realized_pnl = -6000
    result = check_risk("INFY", "BUY", 1, 1500.0, confidence=0.9)
    assert not result["allowed"]
    state.reset()


def test_stop_loss_trigger():
    assert should_stop_loss(1000.0, 970.0)
    assert not should_stop_loss(1000.0, 985.0)


def test_take_profit_trigger():
    assert should_take_profit(1000.0, 1050.0)
    assert not should_take_profit(1000.0, 1020.0)


def test_trade_tracking():
    state = get_risk_state()
    state.reset()
    record_trade_open("SBIN", "BUY", 10, 500.0)
    assert "SBIN" in state.open_positions
    assert state.trade_count == 1
    record_trade_close("SBIN", 200.0)
    assert "SBIN" not in state.open_positions
    assert state.realized_pnl == 200.0
    state.reset()


def test_compute_position_size():
    qty = compute_position_size(2500.0)
    assert qty >= 1
    assert qty * 2500.0 <= 100_000

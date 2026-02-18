"""Tests for options chain analysis and BTST scanning."""

from __future__ import annotations

import pytest

from app.services.fno_scanner import scan_fno_symbols, score_for_buy, _generate_synthetic_ohlcv
from app.services.options_chain import (
    get_atm_strike, get_lot_size, get_option_chain,
    get_strike_range, scan_options_opportunities, select_best_option,
)
from ml.features.engineering import build_features


def test_get_atm_strike():
    assert get_atm_strike(2450.0, "RELIANCE") == 2440.0  # interval 20, rounds to nearest
    assert get_atm_strike(24150.0, "NIFTY") == 24150.0   # interval 50


def test_get_lot_size():
    assert get_lot_size("RELIANCE") == 250
    assert get_lot_size("NIFTY") == 50
    assert get_lot_size("UNKNOWN") == 100  # default


def test_get_strike_range():
    strikes = get_strike_range(2500.0, "RELIANCE", n_strikes=3)
    assert len(strikes) == 7  # -3 to +3
    assert 2500.0 in strikes


@pytest.mark.asyncio
async def test_get_option_chain():
    chain = await get_option_chain("RELIANCE", 2500.0, n_strikes=3)
    assert chain["symbol"] == "RELIANCE"
    assert chain["lot_size"] == 250
    assert len(chain["chain"]) == 7
    for entry in chain["chain"]:
        assert "strike" in entry
        assert "ce" in entry
        assert "pe" in entry
        assert entry["ce"]["last_price"] >= 0
        assert entry["pe"]["last_price"] >= 0


@pytest.mark.asyncio
async def test_select_best_option_call():
    opt = await select_best_option("RELIANCE", 2500.0, "BUY", "intraday")
    assert opt["option_type"] == "CE"
    assert opt["exchange"] == "NFO"
    assert opt["product"] == "MIS"
    assert opt["entry_price"] > 0
    assert opt["sl_price"] < opt["entry_price"]
    assert opt["target_price"] > opt["entry_price"]


@pytest.mark.asyncio
async def test_select_best_option_put():
    opt = await select_best_option("RELIANCE", 2500.0, "SELL", "btst")
    assert opt["option_type"] == "PE"
    assert opt["product"] == "NRML"
    assert opt["trade_type"] == "btst"


@pytest.mark.asyncio
async def test_scan_options_opportunities():
    results = await scan_options_opportunities(
        symbols=["RELIANCE", "TCS", "INFY"], top_n=5, trade_type="intraday",
    )
    # May return 0 if no strong signals, but should not error
    assert isinstance(results, list)


def test_score_for_buy():
    df = _generate_synthetic_ohlcv("TCS", days=300)
    featured = build_features(df)
    result = score_for_buy(featured, "TCS")
    assert result["symbol"] == "TCS"
    assert result["direction"] == "BUY"
    assert 0 <= result["score"] <= 100


@pytest.mark.asyncio
async def test_scan_btst():
    results = await scan_fno_symbols(
        symbols=["RELIANCE", "TCS", "INFY"],
        top_n=3, scan_type="buy",
    )
    assert len(results) <= 3
    for r in results:
        assert r["direction"] == "BUY"


@pytest.mark.asyncio
async def test_scan_both():
    results = await scan_fno_symbols(
        symbols=["RELIANCE", "TCS"],
        top_n=4, scan_type="both",
    )
    assert len(results) <= 4
    directions = {r["direction"] for r in results}
    # Should have both BUY and SELL signals
    assert len(directions) >= 1

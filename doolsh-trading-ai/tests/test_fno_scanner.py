"""Tests for the F&O symbol scanner."""

from __future__ import annotations

import pytest

from app.services.fno_scanner import FNO_SYMBOLS, _generate_synthetic_ohlcv, score_for_short, scan_fno_symbols
from ml.features.engineering import build_features


def test_fno_symbols_list():
    assert len(FNO_SYMBOLS) == 40
    assert "RELIANCE" in FNO_SYMBOLS
    assert "TCS" in FNO_SYMBOLS


def test_synthetic_ohlcv():
    df = _generate_synthetic_ohlcv("RELIANCE", days=300)
    assert len(df) == 300
    assert set(df.columns) == {"date", "open", "high", "low", "close", "volume"}
    assert (df["volume"] > 0).all()


def test_score_for_short():
    df = _generate_synthetic_ohlcv("RELIANCE", days=300)
    featured = build_features(df)
    result = score_for_short(featured, "RELIANCE")
    assert result["symbol"] == "RELIANCE"
    assert 0 <= result["score"] <= 100
    assert result["action"] in ("STRONG SELL", "SELL", "WEAK SELL", "HOLD")
    assert "price" in result
    assert "rsi" in result
    assert isinstance(result["reasons"], list)


@pytest.mark.asyncio
async def test_scan_fno_symbols():
    results = await scan_fno_symbols(symbols=["RELIANCE", "TCS", "INFY"], top_n=3)
    assert len(results) <= 3
    assert all(r["symbol"] in ("RELIANCE", "TCS", "INFY") for r in results)
    # Results should be sorted by score descending
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_scan_fno_full():
    results = await scan_fno_symbols(top_n=10)
    assert len(results) == 10
    # All have required fields
    for r in results:
        assert "symbol" in r
        assert "score" in r
        assert "action" in r

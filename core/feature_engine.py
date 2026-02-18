"""Feature engineering engine for Indian equity/F&O markets.

Computes 30+ technical indicators, regime features, and volume analysis.
All vectorized with pandas. Works with OHLCV data from Kite or yfinance.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

logger = logging.getLogger(__name__)


# ─── Core Indicators ─────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    if HAS_TA:
        df["rsi"] = ta.momentum.rsi(df["close"], window=period)
    else:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    if HAS_TA:
        macd_ind = ta.trend.MACD(df["close"], window_fast=fast, window_slow=slow, window_sign=signal)
        df["macd"] = macd_ind.macd()
        df["macd_signal"] = macd_ind.macd_signal()
        df["macd_hist"] = macd_ind.macd_diff()
    else:
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    if HAS_TA:
        bb = ta.volatility.BollingerBands(df["close"], window=period, window_dev=std_dev)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pct"] = bb.bollinger_pband()
    else:
        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        df["bb_upper"] = sma + std_dev * std
        df["bb_lower"] = sma - std_dev * std
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / df["bb_width"].replace(0, np.nan)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    if HAS_TA and {"high", "low", "close"}.issubset(df.columns):
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=period)
    elif {"high", "low", "close"}.issubset(df.columns):
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.ewm(com=period - 1, min_periods=period).mean()
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    if HAS_TA and {"high", "low", "close"}.issubset(df.columns):
        adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=period)
        df["adx"] = adx_ind.adx()
        df["plus_di"] = adx_ind.adx_pos()
        df["minus_di"] = adx_ind.adx_neg()
        df["di_crossover"] = df["plus_di"] - df["minus_di"]
    return df


def add_stochastic_rsi(df: pd.DataFrame, period: int = 14, smooth: int = 3) -> pd.DataFrame:
    if HAS_TA:
        stoch = ta.momentum.StochRSIIndicator(df["close"], window=period, smooth1=smooth, smooth2=smooth)
        df["stoch_rsi_k"] = stoch.stochrsi_k()
        df["stoch_rsi_d"] = stoch.stochrsi_d()
    else:
        if "rsi" not in df.columns:
            df = add_rsi(df, period)
        rsi = df["rsi"]
        rsi_min = rsi.rolling(period).min()
        rsi_max = rsi.rolling(period).max()
        df["stoch_rsi_k"] = ((rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)).rolling(smooth).mean()
        df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(smooth).mean()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    if "volume" not in df.columns:
        return df
    if HAS_TA:
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    else:
        sign = np.sign(df["close"].diff())
        df["obv"] = (sign * df["volume"]).fillna(0).cumsum()
    return df


# ─── Volume Analysis ────────────────────────────────────────────────────────

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    if "volume" not in df.columns:
        return df

    # Volume ratio (current vs 20-day average)
    vol_avg = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / vol_avg.replace(0, np.nan)

    # Volume-price trend
    df["vol_price_trend"] = (
        df["close"].pct_change() * df["volume_ratio"]
    ).rolling(5).mean()

    # VWAP approximation (daily)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
    df["vwap_deviation"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)

    return df


# ─── Moving Averages ────────────────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
    windows = windows or [5, 10, 20, 50]
    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
    return df


# ─── Returns & Volatility ──────────────────────────────────────────────────

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df["return_1"] = df["close"].pct_change()
    df["return_5"] = df["close"].pct_change(5)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    if "log_return" not in df.columns:
        df = add_returns(df)
    df["volatility"] = df["log_return"].rolling(window).std() * np.sqrt(252)  # annualized for equities
    return df


# ─── Ichimoku Cloud ─────────────────────────────────────────────────────────

def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    if not {"high", "low", "close"}.issubset(df.columns):
        return df

    # Tenkan-sen (9-period)
    high_9 = df["high"].rolling(9).max()
    low_9 = df["low"].rolling(9).min()
    df["ichimoku_tenkan"] = (high_9 + low_9) / 2

    # Kijun-sen (26-period)
    high_26 = df["high"].rolling(26).max()
    low_26 = df["low"].rolling(26).min()
    df["ichimoku_kijun"] = (high_26 + low_26) / 2

    # Senkou Span A
    df["ichimoku_senkou_a"] = ((df["ichimoku_tenkan"] + df["ichimoku_kijun"]) / 2).shift(26)

    # Senkou Span B
    high_52 = df["high"].rolling(52).max()
    low_52 = df["low"].rolling(52).min()
    df["ichimoku_senkou_b"] = ((high_52 + low_52) / 2).shift(26)

    # Cloud signals
    df["ichimoku_above_cloud"] = (
        (df["close"] > df["ichimoku_senkou_a"]) &
        (df["close"] > df["ichimoku_senkou_b"])
    ).astype(int)
    df["ichimoku_below_cloud"] = (
        (df["close"] < df["ichimoku_senkou_a"]) &
        (df["close"] < df["ichimoku_senkou_b"])
    ).astype(int)

    return df


# ─── Candlestick Patterns ──────────────────────────────────────────────────

def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        return df

    body = df["close"] - df["open"]
    body_abs = body.abs()
    high_low_range = df["high"] - df["low"]

    # Engulfing pattern
    prev_body = body.shift(1)
    bullish_engulf = (body > 0) & (prev_body < 0) & (body_abs > prev_body.abs())
    bearish_engulf = (body < 0) & (prev_body > 0) & (body_abs > prev_body.abs())
    df["engulfing_score"] = bullish_engulf.astype(int) - bearish_engulf.astype(int)

    # Doji
    df["is_doji"] = (body_abs / high_low_range.replace(0, np.nan) < 0.1).astype(int)

    # Hammer
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    df["is_hammer"] = (
        (lower_shadow > 2 * body_abs) & (upper_shadow < body_abs * 0.5)
    ).astype(int)

    return df


# ─── Regime Detection ──────────────────────────────────────────────────────

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    if "volatility" not in df.columns:
        df = add_volatility(df)
    if "adx" not in df.columns:
        df = add_adx(df)

    # Volatility regime percentile
    df["vol_regime"] = df["volatility"].rolling(60, min_periods=20).rank(pct=True)

    # Trend strength (ADX normalized)
    adx = df.get("adx", pd.Series(25, index=df.index))
    df["regime_trend_strength"] = adx.clip(0, 100) / 100.0

    # Price momentum
    df["momentum_20"] = df["close"].pct_change(20)

    # Risk score (higher = riskier)
    df["risk_score"] = (
        df["vol_regime"].fillna(0.5) * 0.4 +
        (1 - df["regime_trend_strength"].fillna(0.25)) * 0.3 +
        df["volatility"].fillna(0.2).clip(0, 1) * 0.3
    )

    return df


# ─── Feature Columns ────────────────────────────────────────────────────────

FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_pct", "bb_width", "atr",
    "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_5", "ema_10", "ema_20",
    "return_1", "return_5", "volatility",
    "adx", "plus_di", "minus_di", "di_crossover",
    "stoch_rsi_k", "stoch_rsi_d",
    "obv", "volume_ratio", "vol_price_trend", "vwap_deviation",
    "ichimoku_above_cloud", "ichimoku_below_cloud",
    "engulfing_score", "is_doji", "is_hammer",
    "vol_regime", "regime_trend_strength", "momentum_20", "risk_score",
]


# ─── Master Feature Builder ────────────────────────────────────────────────

def build_features(df: pd.DataFrame, advanced: bool = True) -> pd.DataFrame:
    """Build all features from raw OHLCV data."""
    df = df.copy()
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_returns(df)
    df = add_volatility(df)

    if {"high", "low", "close"}.issubset(df.columns):
        df = add_atr(df)
        df = add_adx(df)

    df = add_stochastic_rsi(df)

    if "volume" in df.columns:
        df = add_obv(df)
        df = add_volume_features(df)

    if advanced:
        df = add_ichimoku(df)
        df = add_candlestick_patterns(df)

    df = add_regime_features(df)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

"""Technical indicator feature engineering for market data.

Computes RSI, MACD, moving averages, Bollinger Bands, ATR, OBV,
and a composite risk score, all using vectorized pandas operations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_moving_averages(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    col: str = "close",
) -> pd.DataFrame:
    windows = windows or [5, 10, 20, 50, 200]
    for w in windows:
        df[f"sma_{w}"] = df[col].rolling(window=w).mean()
        df[f"ema_{w}"] = df[col].ewm(span=w, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.DataFrame:
    delta = df[col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col: str = "close",
) -> pd.DataFrame:
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, col: str = "close"
) -> pd.DataFrame:
    sma = df[col].rolling(window=period).mean()
    rolling_std = df[col].rolling(window=period).std()
    df["bb_upper"] = sma + std_dev * rolling_std
    df["bb_lower"] = sma - std_dev * rolling_std
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_pct"] = (df[col] - df["bb_lower"]) / df["bb_width"].replace(0, np.nan)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.ewm(com=period - 1, min_periods=period).mean()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    sign = np.sign(df["close"].diff())
    df["obv"] = (sign * df["volume"]).fillna(0).cumsum()
    return df


def add_returns(df: pd.DataFrame, col: str = "close") -> pd.DataFrame:
    df["return_1d"] = df[col].pct_change()
    df["return_5d"] = df[col].pct_change(5)
    df["log_return"] = np.log(df[col] / df[col].shift(1))
    return df


def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df["volatility"] = df["log_return"].rolling(window=window).std() * np.sqrt(252)
    return df


def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Composite risk score in [0, 1] from RSI, volatility, and BB position."""
    rsi_norm = df["rsi"].clip(0, 100) / 100.0
    vol_norm = df["volatility"].rank(pct=True)
    bb_risk = (1 - df["bb_pct"].clip(0, 1)).fillna(0.5)
    df["risk_score"] = (0.4 * rsi_norm + 0.35 * vol_norm + 0.25 * bb_risk).clip(0, 1)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps and drop NaN rows."""
    df = df.copy()
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_returns(df)
    df = add_volatility(df)
    if "high" in df.columns and "low" in df.columns:
        df = add_atr(df)
    if "volume" in df.columns:
        df = add_obv(df)
    df = compute_risk_score(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

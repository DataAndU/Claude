"""Technical indicator feature engineering for market data.

Computes RSI, MACD, moving averages, Bollinger Bands, ATR, OBV,
Stochastic RSI, ADX, VWAP, Ichimoku Cloud, volume profile,
market regime features, and a composite risk score,
all using vectorized pandas operations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_moving_averages(df: pd.DataFrame, windows: list[int] | None = None, col: str = "close") -> pd.DataFrame:
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


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, col: str = "close") -> pd.DataFrame:
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, col: str = "close") -> pd.DataFrame:
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


# ─────────────────────────────────────────────────────────────────────────────
# NEW AI-powered features
# ─────────────────────────────────────────────────────────────────────────────


def add_stochastic_rsi(df: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14,
                        smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    """Stochastic RSI — momentum oscillator that measures RSI relative to its range."""
    if "rsi" not in df.columns:
        df = add_rsi(df, period=rsi_period)
    rsi = df["rsi"]
    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()
    rsi_range = rsi_max - rsi_min
    df["stoch_rsi_k"] = ((rsi - rsi_min) / rsi_range.replace(0, np.nan)).rolling(window=smooth_k).mean()
    df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(window=smooth_d).mean()
    df["stoch_rsi_crossover"] = (df["stoch_rsi_k"] - df["stoch_rsi_d"]).fillna(0)
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index — measures trend strength regardless of direction."""
    if not {"high", "low", "close"}.issubset(df.columns):
        return df
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0
    minus_dm[~mask] = 0

    if "atr" not in df.columns:
        df = add_atr(df, period=period)
    atr = df["atr"]

    plus_di = 100 * (plus_dm.ewm(com=period - 1, min_periods=period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(com=period - 1, min_periods=period).mean() / atr.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df["adx"] = dx.ewm(com=period - 1, min_periods=period).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["di_crossover"] = plus_di - minus_di
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Volume Weighted Average Price — institutional benchmark price level."""
    if not {"high", "low", "close", "volume"}.issubset(df.columns):
        return df
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)
    df["vwap_deviation"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)
    return df


def add_ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> pd.DataFrame:
    """Ichimoku Cloud — multi-component trend/support/resistance system."""
    if not {"high", "low", "close"}.issubset(df.columns):
        return df
    high = df["high"]
    low = df["low"]

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_val = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)

    df["ichimoku_tenkan"] = tenkan_sen
    df["ichimoku_kijun"] = kijun_sen
    df["ichimoku_senkou_a"] = senkou_a
    df["ichimoku_senkou_b"] = senkou_b_val
    df["ichimoku_cloud_width"] = senkou_a - senkou_b_val
    # Price relative to cloud
    cloud_top = pd.concat([senkou_a, senkou_b_val], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b_val], axis=1).min(axis=1)
    df["ichimoku_above_cloud"] = (df["close"] > cloud_top).astype(float)
    df["ichimoku_below_cloud"] = (df["close"] < cloud_bottom).astype(float)
    df["ichimoku_tk_cross"] = tenkan_sen - kijun_sen
    return df


def add_volume_profile(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volume profile features — detect institutional accumulation/distribution."""
    if "volume" not in df.columns:
        return df
    vol = df["volume"]
    df["volume_sma"] = vol.rolling(window=window).mean()
    df["volume_ratio"] = vol / df["volume_sma"].replace(0, np.nan)
    # Accumulation/Distribution: high volume on up days, low volume on down days
    price_change = df["close"].diff()
    df["vol_price_trend"] = (np.sign(price_change) * df["volume_ratio"]).rolling(window=5).mean()
    # Volume momentum
    df["volume_momentum"] = vol.pct_change(5)
    # Climax detection: extremely high volume relative to average
    df["volume_climax"] = (df["volume_ratio"] > 2.5).astype(float)
    return df


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features that help ML models detect the current market regime.

    Regime types: trending-up, trending-down, mean-reverting, high-volatility.
    """
    close = df["close"]

    # Trend strength via slope of linear regression over rolling window
    def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        def slope(vals):
            if len(vals) < window:
                return np.nan
            y_mean = vals.mean()
            return ((x * (vals - y_mean)).sum()) / x_var if x_var != 0 else 0
        return series.rolling(window).apply(slope, raw=True)

    df["trend_slope_20"] = _rolling_slope(close, 20)
    df["trend_slope_50"] = _rolling_slope(close, 50)

    # Hurst exponent approximation (simplified R/S analysis)
    log_ret = df.get("log_return", close.pct_change())
    rolling_std = log_ret.rolling(20).std()
    rolling_range = log_ret.rolling(20).apply(lambda x: x.max() - x.min(), raw=True)
    rs = rolling_range / rolling_std.replace(0, np.nan)
    df["hurst_proxy"] = np.log(rs.clip(lower=1e-10)) / np.log(20)

    # Regime score: combines trend + mean-reversion indicators
    adx = df.get("adx", pd.Series(25, index=df.index))
    df["regime_trend_strength"] = (adx / 100.0).clip(0, 1)

    # Volatility regime (relative to historical)
    if "volatility" in df.columns:
        vol_percentile = df["volatility"].rolling(60).rank(pct=True)
        df["volatility_regime"] = vol_percentile

    # Mean-reversion score: high when price is far from SMA and RSI is extreme
    if "sma_20" in df.columns and "rsi" in df.columns:
        price_dev = ((close - df["sma_20"]) / df["sma_20"].replace(0, np.nan)).abs()
        rsi_extremity = ((df["rsi"] - 50).abs() / 50.0).clip(0, 1)
        df["mean_reversion_score"] = (price_dev * rsi_extremity).clip(0, 1)

    return df


def add_price_action_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect key price action patterns using statistical analysis."""
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        return df

    body = df["close"] - df["open"]
    candle_range = df["high"] - df["low"]
    body_pct = body.abs() / candle_range.replace(0, np.nan)

    # Doji detection (small body relative to range)
    df["is_doji"] = (body_pct < 0.1).astype(float)

    # Hammer/hanging man (long lower shadow, small body at top)
    lower_shadow = pd.concat([df["open"], df["close"]], axis=1).min(axis=1) - df["low"]
    upper_shadow = df["high"] - pd.concat([df["open"], df["close"]], axis=1).max(axis=1)
    df["hammer_score"] = (lower_shadow / candle_range.replace(0, np.nan)).fillna(0)

    # Engulfing pattern strength
    prev_body = body.shift(1)
    df["engulfing_score"] = np.where(
        (body > 0) & (prev_body < 0) & (body.abs() > prev_body.abs()),
        1.0,
        np.where(
            (body < 0) & (prev_body > 0) & (body.abs() > prev_body.abs()),
            -1.0,
            0.0,
        ),
    )

    # Consecutive direction count
    direction = np.sign(body)
    groups = (direction != direction.shift()).cumsum()
    df["consecutive_direction"] = direction.groupby(groups).cumcount() + 1
    df["consecutive_direction"] *= direction

    return df


def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    rsi_norm = df["rsi"].clip(0, 100) / 100.0
    vol_norm = df["volatility"].rank(pct=True)
    bb_risk = (1 - df["bb_pct"].clip(0, 1)).fillna(0.5)

    # Enhanced risk score with ADX and volume regime
    adx_factor = (df.get("adx", pd.Series(25, index=df.index)) / 100.0).clip(0, 1) * 0.15
    vol_regime = df.get("volatility_regime", pd.Series(0.5, index=df.index)) * 0.10

    df["risk_score"] = (
        0.30 * rsi_norm + 0.25 * vol_norm + 0.20 * bb_risk + adx_factor + vol_regime
    ).clip(0, 1)
    return df


def build_features(df: pd.DataFrame, advanced: bool = True) -> pd.DataFrame:
    """Build all feature columns from raw OHLCV data.

    Args:
        df: DataFrame with columns [open, high, low, close, volume].
        advanced: If True, include AI-powered advanced features
                  (Stochastic RSI, ADX, VWAP, Ichimoku, regime detection, etc.).
    """
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

    if advanced:
        df = add_stochastic_rsi(df)
        if "high" in df.columns and "low" in df.columns:
            df = add_adx(df)
            df = add_ichimoku(df)
        if "volume" in df.columns:
            df = add_vwap(df)
            df = add_volume_profile(df)
        df = add_market_regime_features(df)
        if {"open", "high", "low", "close"}.issubset(df.columns):
            df = add_price_action_patterns(df)

    df = compute_risk_score(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

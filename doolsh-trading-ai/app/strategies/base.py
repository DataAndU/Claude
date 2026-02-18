"""Base strategy interface and built-in NSE strategy implementations."""

from __future__ import annotations

import abc
from typing import Any, Dict, List

import pandas as pd

from ml.features.engineering import build_features


class BaseStrategy(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        ...

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "class": self.__class__.__name__}


class MACDCrossoverStrategy(BaseStrategy):
    name = "macd_crossover"

    def generate_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        featured = build_features(df)
        signals: List[Dict[str, Any]] = []
        prev_hist = 0.0
        for i, row in featured.iterrows():
            hist = row.get("macd_hist", 0.0)
            if prev_hist <= 0 < hist:
                sig = "BUY"
            elif prev_hist >= 0 > hist:
                sig = "SELL"
            else:
                sig = "HOLD"
            signals.append({
                "index": int(i), "signal": sig,
                "confidence": min(abs(hist) * 10, 1.0),
                "price": row["close"],
                "risk_score": row.get("risk_score", 0.5),
                "rsi": row.get("rsi", 50), "macd_hist": hist,
            })
            prev_hist = hist
        return signals


class RSIMeanReversionStrategy(BaseStrategy):
    name = "rsi_mean_reversion"

    def __init__(self, oversold: float = 30.0, overbought: float = 70.0):
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        featured = build_features(df)
        signals: List[Dict[str, Any]] = []
        for i, row in featured.iterrows():
            rsi = row.get("rsi", 50)
            if rsi < self.oversold:
                sig, conf = "BUY", (self.oversold - rsi) / self.oversold
            elif rsi > self.overbought:
                sig, conf = "SELL", (rsi - self.overbought) / (100 - self.overbought)
            else:
                sig, conf = "HOLD", 0.3
            signals.append({
                "index": int(i), "signal": sig,
                "confidence": round(min(conf, 1.0), 4),
                "price": row["close"],
                "risk_score": row.get("risk_score", 0.5),
                "rsi": rsi, "macd_hist": row.get("macd_hist", 0),
            })
        return signals


class DualMovingAverageCrossover(BaseStrategy):
    name = "dual_ma_crossover"

    def __init__(self, short_window: int = 10, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        featured = build_features(df)
        short_col = f"sma_{self.short_window}"
        long_col = f"sma_{self.long_window}"
        if short_col not in featured.columns or long_col not in featured.columns:
            featured[short_col] = featured["close"].rolling(self.short_window).mean()
            featured[long_col] = featured["close"].rolling(self.long_window).mean()
            featured.dropna(inplace=True)
            featured.reset_index(drop=True, inplace=True)
        signals: List[Dict[str, Any]] = []
        prev_diff = 0.0
        for i, row in featured.iterrows():
            diff = row[short_col] - row[long_col]
            if prev_diff <= 0 < diff:
                sig = "BUY"
            elif prev_diff >= 0 > diff:
                sig = "SELL"
            else:
                sig = "HOLD"
            signals.append({
                "index": int(i), "signal": sig,
                "confidence": min(abs(diff) / row["close"], 1.0),
                "price": row["close"],
                "risk_score": row.get("risk_score", 0.5),
                "rsi": row.get("rsi", 50), "macd_hist": row.get("macd_hist", 0),
            })
            prev_diff = diff
        return signals


STRATEGY_REGISTRY: Dict[str, type[BaseStrategy]] = {
    "macd_crossover": MACDCrossoverStrategy,
    "rsi_mean_reversion": RSIMeanReversionStrategy,
    "dual_ma_crossover": DualMovingAverageCrossover,
}

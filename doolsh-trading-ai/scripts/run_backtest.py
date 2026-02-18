"""Run a standalone backtest from the command line.

Usage:
    python -m scripts.run_backtest --symbol RELIANCE --model-type rf --version v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.market_data import fetch_daily_prices
from app.services.signal_generator import generate_signals
from backtesting.engine import BacktestConfig, BacktestEngine
from ml.features.engineering import build_features


async def main(symbol: str, model_type: str, version: str) -> None:
    save_dir = Path("ml/saved_models")
    ext = ".pkl" if model_type == "rf" else ".pt"
    model_path = str(save_dir / f"{model_type}_{symbol}_{version}{ext}")

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print(f"Fetching data for {symbol} via Kite...")
    df = await fetch_daily_prices(symbol, use_cache=False)
    print(f"Rows: {len(df)}")

    signals = generate_signals(df, model_path=model_path, model_type=model_type)
    print(f"Signals: {len(signals)}")

    engine = BacktestEngine(BacktestConfig())
    featured = build_features(df)
    result = engine.run(featured, signals)

    output = asdict(result)
    output.pop("equity_curve", None)
    output.pop("trades", None)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--model-type", default="rf", choices=["rf", "lstm"])
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    asyncio.run(main(args.symbol, args.model_type, args.version))

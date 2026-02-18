"""WebSocket live ticker via Kite Connect.

Provides real-time price updates for watchlist symbols.
Falls back to polling when WebSocket is unavailable.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from core.config import load_config
from core.kite_auth import get_access_token, get_kite, is_logged_in

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

_ticker = None
_subscribed_tokens: List[int] = []
_tick_data: Dict[int, Dict[str, Any]] = {}
_callbacks: List[Callable] = []
_running = False


def on_tick(callback: Callable[[Dict[int, Dict]], None]) -> None:
    """Register a callback for tick updates."""
    _callbacks.append(callback)


def get_latest_ticks() -> Dict[int, Dict[str, Any]]:
    """Return latest tick data."""
    return _tick_data.copy()


def start_ticker(instrument_tokens: Optional[List[int]] = None) -> bool:
    """Start the WebSocket ticker.

    Args:
        instrument_tokens: List of Kite instrument tokens to subscribe.
            If None, resolves tokens from watchlist config.
    """
    global _ticker, _running

    if not is_logged_in():
        logger.warning("Cannot start ticker: not logged in to Kite")
        return False

    cfg = load_config()
    try:
        from kiteconnect import KiteTicker

        _ticker = KiteTicker(cfg.kite_api_key, get_access_token())

        def _on_ticks(ws, ticks):
            for tick in ticks:
                token = tick.get("instrument_token")
                if token:
                    _tick_data[token] = {
                        "last_price": tick.get("last_price", 0),
                        "volume": tick.get("volume_traded", 0),
                        "buy_quantity": tick.get("total_buy_quantity", 0),
                        "sell_quantity": tick.get("total_sell_quantity", 0),
                        "change": tick.get("change", 0),
                        "ohlc": tick.get("ohlc", {}),
                        "timestamp": datetime.now(IST).isoformat(),
                    }
            for cb in _callbacks:
                try:
                    cb(_tick_data)
                except Exception as e:
                    logger.warning("Tick callback error: %s", e)

        def _on_connect(ws, response):
            tokens = instrument_tokens or _resolve_tokens()
            if tokens:
                ws.subscribe(tokens)
                ws.set_mode(ws.MODE_FULL, tokens)
                logger.info("Subscribed to %d instruments", len(tokens))

        def _on_close(ws, code, reason):
            global _running
            _running = False
            logger.info("Ticker closed: %s %s", code, reason)

        def _on_error(ws, code, reason):
            logger.error("Ticker error: %s %s", code, reason)

        _ticker.on_ticks = _on_ticks
        _ticker.on_connect = _on_connect
        _ticker.on_close = _on_close
        _ticker.on_error = _on_error

        _running = True
        thread = threading.Thread(target=_ticker.connect, daemon=True)
        thread.start()
        logger.info("Kite ticker started")
        return True

    except Exception as e:
        logger.error("Failed to start ticker: %s", e)
        return False


def stop_ticker() -> None:
    global _ticker, _running
    if _ticker:
        try:
            _ticker.close()
        except Exception:
            pass
        _ticker = None
    _running = False
    logger.info("Kite ticker stopped")


def is_ticker_running() -> bool:
    return _running


def _resolve_tokens() -> List[int]:
    """Resolve watchlist symbols to instrument tokens."""
    try:
        kite = get_kite()
        cfg = load_config()
        instruments = [f"{cfg.exchange}:{s}" for s in cfg.watchlist]
        data = kite.ltp(instruments)
        tokens = []
        for key, val in data.items():
            if "instrument_token" in val:
                tokens.append(val["instrument_token"])
        return tokens
    except Exception as e:
        logger.warning("Could not resolve instrument tokens: %s", e)
        return []

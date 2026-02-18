"""Kite WebSocket live ticker for real-time market data.

Connects to Kite Ticker, subscribes to instruments, and keeps an
in-memory dict of the latest ticks. Runs in a background thread.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional

from app.core.config import get_settings
from app.core.kite import get_access_token, get_kite

logger = logging.getLogger(__name__)
settings = get_settings()

# Latest tick per instrument token
_ticks: Dict[int, Dict[str, Any]] = {}
_ticker_thread: Optional[threading.Thread] = None
_running = False


def _on_ticks(ws, ticks: List[Dict]) -> None:
    for t in ticks:
        token = t.get("instrument_token")
        if token:
            _ticks[token] = t


def _on_connect(ws, response) -> None:
    logger.info("Kite Ticker connected")


def _on_close(ws, code, reason) -> None:
    logger.warning("Kite Ticker closed: code=%s reason=%s", code, reason)


def _on_error(ws, code, reason) -> None:
    logger.error("Kite Ticker error: code=%s reason=%s", code, reason)


def start_ticker(instrument_tokens: List[int]) -> None:
    """Start the WebSocket ticker in a daemon thread."""
    global _ticker_thread, _running

    if _running:
        logger.info("Ticker already running")
        return

    access_token = get_access_token()
    if not access_token:
        raise RuntimeError("Cannot start ticker — not logged in")

    from kiteconnect import KiteTicker

    kws = KiteTicker(settings.kite_api_key, access_token)
    kws.on_ticks = _on_ticks
    kws.on_connect = lambda ws, resp: (
        _on_connect(ws, resp),
        ws.subscribe(instrument_tokens),
        ws.set_mode(ws.MODE_FULL, instrument_tokens),
    )
    kws.on_close = _on_close
    kws.on_error = _on_error

    def _run():
        global _running
        _running = True
        kws.connect(threaded=False)
        _running = False

    _ticker_thread = threading.Thread(target=_run, daemon=True, name="kite-ticker")
    _ticker_thread.start()
    logger.info("Ticker started for %d instruments", len(instrument_tokens))


def stop_ticker() -> None:
    global _running
    _running = False
    logger.info("Ticker stop requested")


def get_latest_tick(instrument_token: int) -> Optional[Dict[str, Any]]:
    return _ticks.get(instrument_token)


def get_all_ticks() -> Dict[int, Dict[str, Any]]:
    return _ticks.copy()


def is_ticker_running() -> bool:
    return _running

"""Order management service — places, modifies, and cancels orders via Kite.

Supports both LIVE mode (real orders on Zerodha) and PAPER mode (simulated
fills logged to the database).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.kite import get_kite, is_logged_in

logger = logging.getLogger(__name__)
settings = get_settings()

# In-memory paper trade ledger (persisted to DB by the API layer)
_paper_orders: List[Dict[str, Any]] = []
_paper_order_id = 0


def _next_paper_id() -> str:
    global _paper_order_id
    _paper_order_id += 1
    return f"PAPER-{_paper_order_id:06d}"


async def place_order(
    symbol: str,
    side: str,  # "BUY" or "SELL"
    quantity: int,
    order_type: str = "MARKET",
    price: float = 0.0,
    trigger_price: float = 0.0,
    product: Optional[str] = None,
    tag: str = "doolsh",
) -> Dict[str, Any]:
    """Place an order on Kite (live) or simulate it (paper).

    Returns a dict with at minimum: order_id, status, symbol, side, quantity.
    """
    product = product or settings.trading_product
    exchange = settings.trading_exchange

    if settings.trading_mode == "paper":
        return _place_paper_order(symbol, side, quantity, order_type, price, product)

    # ---- LIVE MODE ----
    if not is_logged_in():
        raise RuntimeError("Kite not logged in — cannot place live order")

    kite = get_kite()
    kite_side = kite.TRANSACTION_TYPE_BUY if side.upper() == "BUY" else kite.TRANSACTION_TYPE_SELL

    order_type_map = {
        "MARKET": kite.ORDER_TYPE_MARKET,
        "LIMIT": kite.ORDER_TYPE_LIMIT,
        "SL": kite.ORDER_TYPE_SL,
        "SL-M": kite.ORDER_TYPE_SLM,
    }

    product_map = {
        "MIS": kite.PRODUCT_MIS,
        "CNC": kite.PRODUCT_CNC,
        "NRML": kite.PRODUCT_NRML,
    }

    params: Dict[str, Any] = {
        "tradingsymbol": symbol,
        "exchange": exchange,
        "transaction_type": kite_side,
        "quantity": quantity,
        "order_type": order_type_map.get(order_type.upper(), kite.ORDER_TYPE_MARKET),
        "product": product_map.get(product.upper(), kite.PRODUCT_MIS),
        "variety": kite.VARIETY_REGULAR,
        "tag": tag,
    }
    if order_type.upper() == "LIMIT" and price > 0:
        params["price"] = price
    if order_type.upper() in ("SL", "SL-M") and trigger_price > 0:
        params["trigger_price"] = trigger_price
    if order_type.upper() == "SL" and price > 0:
        params["price"] = price

    order_id = kite.place_order(**params)
    logger.info("LIVE order placed: %s %s %s qty=%d → id=%s", side, symbol, order_type, quantity, order_id)

    return {
        "order_id": str(order_id),
        "status": "PLACED",
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "order_type": order_type,
        "product": product,
        "mode": "live",
    }


def _place_paper_order(
    symbol: str,
    side: str,
    quantity: int,
    order_type: str,
    price: float,
    product: str,
) -> Dict[str, Any]:
    order_id = _next_paper_id()
    record = {
        "order_id": order_id,
        "status": "COMPLETE",
        "symbol": symbol,
        "side": side.upper(),
        "quantity": quantity,
        "order_type": order_type,
        "price": price,
        "product": product,
        "mode": "paper",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _paper_orders.append(record)
    logger.info("PAPER order: %s %s %s qty=%d price=%.2f", side, symbol, order_type, quantity, price)
    return record


async def cancel_order(order_id: str, variety: str = "regular") -> Dict[str, Any]:
    if settings.trading_mode == "paper":
        for o in _paper_orders:
            if o["order_id"] == order_id:
                o["status"] = "CANCELLED"
                return o
        return {"order_id": order_id, "status": "NOT_FOUND"}

    kite = get_kite()
    kite.cancel_order(variety=variety, order_id=order_id)
    return {"order_id": order_id, "status": "CANCELLED"}


async def modify_order(
    order_id: str,
    quantity: Optional[int] = None,
    price: Optional[float] = None,
    order_type: Optional[str] = None,
    trigger_price: Optional[float] = None,
    variety: str = "regular",
) -> Dict[str, Any]:
    if settings.trading_mode == "paper":
        for o in _paper_orders:
            if o["order_id"] == order_id:
                if quantity:
                    o["quantity"] = quantity
                if price:
                    o["price"] = price
                return o
        return {"order_id": order_id, "status": "NOT_FOUND"}

    kite = get_kite()
    params: Dict[str, Any] = {"variety": variety, "order_id": order_id}
    if quantity:
        params["quantity"] = quantity
    if price:
        params["price"] = price
    if trigger_price:
        params["trigger_price"] = trigger_price
    kite.modify_order(**params)
    return {"order_id": order_id, "status": "MODIFIED"}


async def get_orders() -> List[Dict[str, Any]]:
    if settings.trading_mode == "paper":
        return _paper_orders.copy()
    kite = get_kite()
    return kite.orders()


async def get_positions() -> Dict[str, Any]:
    if settings.trading_mode == "paper":
        return {"net": [], "day": []}
    kite = get_kite()
    return kite.positions()


async def get_holdings() -> List[Dict[str, Any]]:
    if settings.trading_mode == "paper":
        return []
    kite = get_kite()
    return kite.holdings()


def get_paper_orders() -> List[Dict[str, Any]]:
    return _paper_orders.copy()

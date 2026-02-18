"""Order placement and management via Kite Connect.

Supports both paper (simulated) and live order execution.
Paper mode maintains an in-memory order book for testing.
"""

from __future__ import annotations

import logging
import random
import string
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.config import load_config
from core.kite_auth import get_kite, is_logged_in

logger = logging.getLogger(__name__)

# Paper trading order book
_paper_orders: List[Dict[str, Any]] = []
_paper_positions: Dict[str, Dict[str, Any]] = {}
_paper_balance: float = 100000.0


def _gen_order_id() -> str:
    return "PAPER_" + "".join(random.choices(string.digits, k=12))


async def place_order(
    symbol: str,
    side: str,  # BUY or SELL
    quantity: int,
    order_type: str = "MARKET",
    price: float = 0.0,
    trigger_price: float = 0.0,
    product: str = "MIS",
    exchange: str = "NSE",
    trade_type: str = "intraday",
    tag: str = "kiteai",
) -> Dict[str, Any]:
    """Place an order — routes to Kite live or paper engine."""
    cfg = load_config()

    order = {
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "order_type": order_type,
        "price": round(price, 2),
        "product": product,
        "exchange": exchange,
        "trade_type": trade_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if cfg.is_live and is_logged_in():
        return await _place_live_order(order, trigger_price, tag)
    else:
        return _place_paper_order(order)


async def _place_live_order(
    order: dict,
    trigger_price: float,
    tag: str,
) -> Dict[str, Any]:
    """Place a real order via Kite Connect."""
    kite = get_kite()

    transaction = (
        kite.TRANSACTION_TYPE_BUY if order["side"] == "BUY"
        else kite.TRANSACTION_TYPE_SELL
    )

    order_type_map = {
        "MARKET": kite.ORDER_TYPE_MARKET,
        "LIMIT": kite.ORDER_TYPE_LIMIT,
        "SL": kite.ORDER_TYPE_SL,
        "SL-M": kite.ORDER_TYPE_SLM,
    }

    product_map = {
        "MIS": kite.PRODUCT_MIS,
        "NRML": kite.PRODUCT_NRML,
        "CNC": kite.PRODUCT_CNC,
    }

    params = {
        "tradingsymbol": order["symbol"],
        "exchange": order["exchange"],
        "transaction_type": transaction,
        "quantity": order["quantity"],
        "order_type": order_type_map.get(order["order_type"], kite.ORDER_TYPE_MARKET),
        "product": product_map.get(order["product"], kite.PRODUCT_MIS),
        "tag": tag,
    }

    if order["order_type"] == "LIMIT" and order["price"] > 0:
        params["price"] = order["price"]
    if trigger_price > 0:
        params["trigger_price"] = trigger_price

    try:
        order_id = kite.place_order(variety=kite.VARIETY_REGULAR, **params)
        logger.info(
            "LIVE ORDER placed: %s %s %s x%d @ %s (id=%s)",
            order["side"], order["symbol"], order["product"],
            order["quantity"], order["order_type"], order_id,
        )
        return {
            **order,
            "order_id": str(order_id),
            "status": "PLACED",
            "mode": "live",
        }
    except Exception as e:
        logger.error("Live order failed: %s", e)
        return {
            **order,
            "order_id": "",
            "status": "FAILED",
            "mode": "live",
            "error": str(e),
        }


def _place_paper_order(order: dict) -> Dict[str, Any]:
    """Simulate order execution in paper mode."""
    global _paper_balance
    order_id = _gen_order_id()
    cost = order["price"] * order["quantity"]

    # Simulate fill with slight slippage
    slippage = random.uniform(-0.001, 0.001)
    fill_price = order["price"] * (1 + slippage)
    fill_price = round(fill_price, 2)

    result = {
        **order,
        "order_id": order_id,
        "fill_price": fill_price,
        "status": "COMPLETE",
        "mode": "paper",
    }

    # Update paper positions
    sym = order["symbol"]
    if order["side"] == "BUY":
        if sym in _paper_positions:
            pos = _paper_positions[sym]
            total_qty = pos["quantity"] + order["quantity"]
            avg_price = (
                (pos["avg_price"] * pos["quantity"] + fill_price * order["quantity"])
                / total_qty
            )
            pos["quantity"] = total_qty
            pos["avg_price"] = round(avg_price, 2)
        else:
            _paper_positions[sym] = {
                "symbol": sym,
                "quantity": order["quantity"],
                "avg_price": fill_price,
                "side": "BUY",
                "product": order["product"],
                "pnl": 0.0,
            }
        _paper_balance -= cost
    else:
        if sym in _paper_positions:
            pos = _paper_positions[sym]
            pnl = (fill_price - pos["avg_price"]) * order["quantity"]
            pos["quantity"] -= order["quantity"]
            _paper_balance += cost + pnl
            result["pnl"] = round(pnl, 2)
            if pos["quantity"] <= 0:
                del _paper_positions[sym]
        else:
            # Short sell
            _paper_positions[sym] = {
                "symbol": sym,
                "quantity": order["quantity"],
                "avg_price": fill_price,
                "side": "SELL",
                "product": order["product"],
                "pnl": 0.0,
            }
            _paper_balance += cost

    _paper_orders.append(result)
    logger.info(
        "PAPER ORDER: %s %s x%d @ %.2f (id=%s)",
        order["side"], sym, order["quantity"], fill_price, order_id,
    )
    return result


async def cancel_order(order_id: str, variety: str = "regular") -> dict:
    """Cancel a pending order."""
    cfg = load_config()
    if cfg.is_live and is_logged_in():
        try:
            kite = get_kite()
            kite.cancel_order(variety=variety, order_id=order_id)
            return {"order_id": order_id, "status": "CANCELLED"}
        except Exception as e:
            return {"order_id": order_id, "status": "FAILED", "error": str(e)}

    # Paper mode
    for o in _paper_orders:
        if o["order_id"] == order_id and o["status"] == "PLACED":
            o["status"] = "CANCELLED"
            return {"order_id": order_id, "status": "CANCELLED"}
    return {"order_id": order_id, "status": "NOT_FOUND"}


async def get_orders() -> List[dict]:
    """Get today's orders."""
    cfg = load_config()
    if cfg.is_live and is_logged_in():
        try:
            kite = get_kite()
            return kite.orders()
        except Exception as e:
            logger.warning("Failed to fetch orders: %s", e)
    return _paper_orders.copy()


def get_paper_positions() -> Dict[str, dict]:
    return _paper_positions.copy()


def get_paper_balance() -> float:
    return round(_paper_balance, 2)


def reset_paper_trading(capital: float = 100000.0) -> None:
    global _paper_balance
    _paper_orders.clear()
    _paper_positions.clear()
    _paper_balance = capital

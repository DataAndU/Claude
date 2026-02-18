"""Order management and position endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.deps import get_current_user, rate_limit
from app.models.user import User
from app.schemas.trading import OrderOut, PlaceOrderRequest
from app.services.order_manager import (
    cancel_order,
    get_holdings,
    get_orders,
    get_positions,
    place_order,
)
from app.services.risk_manager import check_risk, compute_position_size

router = APIRouter(prefix="/orders", tags=["orders"], dependencies=[Depends(rate_limit)])


@router.post("/place", response_model=OrderOut)
async def api_place_order(body: PlaceOrderRequest, user: User = Depends(get_current_user)):
    """Place a manual order (paper or live depending on TRADING_MODE)."""
    risk = check_risk(body.symbol, body.side, body.quantity, body.price, confidence=1.0)
    if not risk["allowed"]:
        raise HTTPException(status_code=400, detail=risk["reason"])

    result = await place_order(
        symbol=body.symbol,
        side=body.side,
        quantity=risk["adjusted_qty"],
        order_type=body.order_type,
        price=body.price,
        trigger_price=body.trigger_price,
        product=body.product,
    )
    return OrderOut(**{k: result[k] for k in OrderOut.model_fields if k in result})


@router.delete("/{order_id}")
async def api_cancel_order(order_id: str, user: User = Depends(get_current_user)):
    return await cancel_order(order_id)


@router.get("/")
async def api_list_orders(user: User = Depends(get_current_user)):
    return await get_orders()


@router.get("/positions")
async def api_positions(user: User = Depends(get_current_user)):
    return await get_positions()


@router.get("/holdings")
async def api_holdings(user: User = Depends(get_current_user)):
    return await get_holdings()

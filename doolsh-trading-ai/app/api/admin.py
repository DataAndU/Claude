"""Admin-only API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import require_role
from app.core.security import generate_api_key, hash_password
from app.models.log import AuditLog
from app.models.ml_model import ModelVersion
from app.models.order import OrderLog
from app.models.strategy import Signal, Strategy, Trade
from app.models.user import ApiKey, User, UserRole
from app.schemas.auth import UserOut

router = APIRouter(
    prefix="/admin", tags=["admin"],
    dependencies=[Depends(require_role(UserRole.ADMIN))],
)


@router.get("/users", response_model=list[UserOut])
async def list_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).order_by(User.id))
    return result.scalars().all()


@router.patch("/users/{user_id}/role")
async def set_user_role(user_id: int, role: str, db: AsyncSession = Depends(get_db)):
    valid = {r.value for r in UserRole}
    if role not in valid:
        raise HTTPException(status_code=400, detail=f"Invalid role: {role}")
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.role = role
    await db.flush()
    return {"id": user.id, "username": user.username, "role": user.role}


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    await db.delete(user)


@router.post("/api-keys")
async def create_api_key(user_id: int, label: str = "default", db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="User not found")
    key = generate_api_key()
    db.add(ApiKey(user_id=user_id, key=key, label=label))
    await db.flush()
    return {"api_key": key, "label": label}


@router.get("/dashboard")
async def admin_dashboard(db: AsyncSession = Depends(get_db)):
    user_count = (await db.execute(select(func.count(User.id)))).scalar() or 0
    strategy_count = (await db.execute(select(func.count(Strategy.id)))).scalar() or 0
    signal_count = (await db.execute(select(func.count(Signal.id)))).scalar() or 0
    trade_count = (await db.execute(select(func.count(Trade.id)))).scalar() or 0
    order_count = (await db.execute(select(func.count(OrderLog.id)))).scalar() or 0
    model_count = (await db.execute(select(func.count(ModelVersion.id)))).scalar() or 0
    log_count = (await db.execute(select(func.count(AuditLog.id)))).scalar() or 0
    return {
        "users": user_count, "strategies": strategy_count,
        "signals": signal_count, "trades": trade_count,
        "orders": order_count, "model_versions": model_count,
        "audit_logs": log_count,
    }

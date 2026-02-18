"""Strategy CRUD API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.strategy import Strategy
from app.models.user import User
from app.schemas.trading import StrategyCreate, StrategyOut

router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.post("/", response_model=StrategyOut, status_code=status.HTTP_201_CREATED)
async def create_strategy(
    body: StrategyCreate, user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(select(Strategy).where(Strategy.name == body.name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Strategy name already exists")
    strategy = Strategy(name=body.name, description=body.description, parameters=body.parameters, user_id=user.id)
    db.add(strategy)
    await db.flush()
    await db.refresh(strategy)
    return strategy


@router.get("/", response_model=list[StrategyOut])
async def list_strategies(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Strategy).where(Strategy.user_id == user.id).order_by(Strategy.id))
    return result.scalars().all()


@router.get("/{strategy_id}", response_model=StrategyOut)
async def get_strategy(strategy_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Strategy).where(Strategy.id == strategy_id, Strategy.user_id == user.id))
    strategy = result.scalar_one_or_none()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return strategy


@router.delete("/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_strategy(strategy_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Strategy).where(Strategy.id == strategy_id, Strategy.user_id == user.id))
    strategy = result.scalar_one_or_none()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    await db.delete(strategy)

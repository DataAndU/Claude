"""Kite order log — tracks every order placed (paper or live)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class OrderLog(Base):
    __tablename__ = "order_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    exchange: Mapped[str] = mapped_column(String(10), default="NSE")
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[float] = mapped_column(Float, default=0.0)
    order_type: Mapped[str] = mapped_column(String(10), default="MARKET")
    product: Mapped[str] = mapped_column(String(10), default="MIS")
    status: Mapped[str] = mapped_column(String(20), default="PLACED")
    mode: Mapped[str] = mapped_column(String(10), default="paper")
    pnl: Mapped[float] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

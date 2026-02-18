"""SQLite database setup and ORM models.

Single-file database, zero external dependencies beyond SQLAlchemy.
All tables auto-created on first run.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


class Base(DeclarativeBase):
    pass


# ─── ORM Models ──────────────────────────────────────────────────────────────


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    symbol = Column(String(20), nullable=False)
    side = Column(String(4), nullable=False)  # BUY / SELL
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    cost = Column(Float, default=0.0)
    pnl = Column(Float, default=0.0)
    status = Column(String(20), default="open")  # open / closed / cancelled
    mode = Column(String(10), default="paper")  # paper / live
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    model_type = Column(String(30), nullable=True)
    confidence = Column(Float, nullable=True)
    close_reason = Column(String(50), nullable=True)
    closed_at = Column(DateTime, nullable=True)


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    signal = Column(String(10), nullable=False)  # BUY / SELL / HOLD
    confidence = Column(Float, default=0.0)
    model_type = Column(String(30), nullable=True)
    features_json = Column(Text, nullable=True)


class ModelRecord(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    model_type = Column(String(30), nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    accuracy = Column(Float, default=0.0)
    f1 = Column(Float, default=0.0)
    file_path = Column(String(200), nullable=True)
    feature_importance_json = Column(Text, nullable=True)


class DailyPnL(Base):
    __tablename__ = "daily_pnl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, unique=True)
    realized_pnl = Column(Float, default=0.0)
    trade_count = Column(Integer, default=0)
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)


# ─── Engine Setup ─────────────────────────────────────────────────────────────


def init_db(db_path: str = "database/trading.db") -> None:
    """Initialize the database engine and create all tables."""
    global _engine, _SessionLocal

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(f"sqlite:///{path}", echo=False)
    _SessionLocal = sessionmaker(bind=_engine)
    Base.metadata.create_all(_engine)
    logger.info("Database initialized at %s", path)


def get_session() -> Session:
    """Return a new database session."""
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


def get_engine():
    if _engine is None:
        init_db()
    return _engine

"""SQLite database for KiteAI.

Single-file database for trades, signals, models, and daily P&L.
Auto-creates tables on first run. Zero external DB dependencies.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from core.config import load_config

logger = logging.getLogger(__name__)

Base = declarative_base()


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    symbol = Column(String(50), nullable=False)
    exchange = Column(String(10), default="NSE")
    side = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    fill_price = Column(Float)
    product = Column(String(10), default="MIS")
    trade_type = Column(String(20), default="intraday")
    order_id = Column(String(50))
    status = Column(String(20), default="COMPLETE")
    mode = Column(String(10), default="paper")
    pnl = Column(Float, default=0.0)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    confidence = Column(Float)
    signal_quality = Column(String(5))
    regime = Column(String(30))
    close_reason = Column(String(30))
    closed_at = Column(DateTime)


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    symbol = Column(String(50), nullable=False)
    signal = Column(String(10), nullable=False)
    confidence = Column(Float)
    quality = Column(String(5))
    price = Column(Float)
    rsi = Column(Float)
    adx = Column(Float)
    model_type = Column(String(30))
    regime = Column(String(30))
    explanation = Column(Text)


class ModelRecord(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    model_type = Column(String(30), nullable=False)
    symbol = Column(String(50), nullable=False)
    accuracy = Column(Float)
    f1 = Column(Float)
    file_path = Column(String(200))
    feature_importance = Column(Text)


class DailyPnL(Base):
    __tablename__ = "daily_pnl"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, unique=True)
    realized_pnl = Column(Float, default=0.0)
    trade_count = Column(Integer, default=0)
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)


# ─── Engine & Session ────────────────────────────────────────────────────────

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        cfg = load_config()
        db_path = cfg.db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(_engine)
        logger.info("Database initialized: %s", db_path)
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


# ─── CRUD Operations ────────────────────────────────────────────────────────

def save_trade(data: Dict[str, Any]) -> int:
    session = get_session()
    trade = Trade(**{k: v for k, v in data.items() if hasattr(Trade, k)})
    session.add(trade)
    session.commit()
    tid = trade.id
    session.close()
    return tid


def get_trades(limit: int = 50, symbol: str = "") -> List[Dict]:
    session = get_session()
    q = session.query(Trade).order_by(Trade.timestamp.desc())
    if symbol:
        q = q.filter(Trade.symbol == symbol)
    trades = q.limit(limit).all()
    result = [
        {c.name: getattr(t, c.name) for c in Trade.__table__.columns}
        for t in trades
    ]
    session.close()
    return result


def save_signal(data: Dict[str, Any]) -> int:
    session = get_session()
    sig = Signal(**{k: v for k, v in data.items() if hasattr(Signal, k)})
    session.add(sig)
    session.commit()
    sid = sig.id
    session.close()
    return sid


def get_signals(limit: int = 50) -> List[Dict]:
    session = get_session()
    sigs = session.query(Signal).order_by(Signal.timestamp.desc()).limit(limit).all()
    result = [
        {c.name: getattr(s, c.name) for c in Signal.__table__.columns}
        for s in sigs
    ]
    session.close()
    return result


def save_model_record(data: Dict[str, Any]) -> int:
    session = get_session()
    rec = ModelRecord(**{k: v for k, v in data.items() if hasattr(ModelRecord, k)})
    session.add(rec)
    session.commit()
    mid = rec.id
    session.close()
    return mid


def save_daily_pnl(date_str: str, pnl: float, trades: int, wins: int, losses: int) -> None:
    session = get_session()
    existing = session.query(DailyPnL).filter(DailyPnL.date == date_str).first()
    if existing:
        existing.realized_pnl = pnl
        existing.trade_count = trades
        existing.win_count = wins
        existing.loss_count = losses
    else:
        session.add(DailyPnL(
            date=date_str, realized_pnl=pnl,
            trade_count=trades, win_count=wins, loss_count=losses,
        ))
    session.commit()
    session.close()


def get_daily_pnl(days: int = 30) -> List[Dict]:
    session = get_session()
    records = session.query(DailyPnL).order_by(DailyPnL.date.desc()).limit(days).all()
    result = [
        {c.name: getattr(r, c.name) for c in DailyPnL.__table__.columns}
        for r in records
    ]
    session.close()
    return result

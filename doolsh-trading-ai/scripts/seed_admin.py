"""Seed an admin user into the SQLite database.

Usage:
    python -m scripts.seed_admin
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.core.database import async_session_factory, init_db
from app.core.security import hash_password
from app.models.user import User, UserRole


async def main() -> None:
    await init_db()

    async with async_session_factory() as session:
        result = await session.execute(select(User).where(User.username == "admin"))
        if result.scalar_one_or_none():
            print("Admin user already exists. Skipping.")
            return

        admin = User(
            email="admin@doolsh.local",
            username="admin",
            hashed_password=hash_password("Admin@12345"),
            role=UserRole.ADMIN.value,
        )
        session.add(admin)
        await session.commit()
        print("Admin user created: admin / Admin@12345")


if __name__ == "__main__":
    asyncio.run(main())

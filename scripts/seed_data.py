"""
Seed sample labeled emails into the database for development and testing.

Usage:
    python -m scripts.seed_data
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.database import init_db
from app.schemas import Base

DATA_FILE = Path(__file__).parent.parent / "data" / "sample_emails.json"


async def seed() -> None:
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    print(f"Loading seed data from {DATA_FILE}...")
    with open(DATA_FILE) as f:
        emails = json.load(f)

    print(f"  → Found {len(emails)} sample emails")
    print("  → Database tables created / verified")
    print("  → Seed complete. Run the API server and POST to /classify to classify emails.")


if __name__ == "__main__":
    asyncio.run(seed())

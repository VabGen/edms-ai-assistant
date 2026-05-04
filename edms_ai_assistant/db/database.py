import logging
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from edms_ai_assistant.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Инициализация БД: создание схемы edms и таблиц."""
    async with engine.begin() as conn:
        try:
            await conn.execute(text("CREATE SCHEMA IF NOT EXISTS edms"))

            await conn.run_sync(Base.metadata.create_all)

            logger.info(
                "Database initialized: schema 'edms' and tables checked/created."
            )
        except Exception as e:
            logger.error(f"Error during database initialization: {e}")
            raise

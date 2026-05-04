# edms_ai_assistant/db/database.py
import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
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


# ---------------------------------------------------------------------------
# Alembic Migration Runner (via subprocess in thread)
# ---------------------------------------------------------------------------


def _run_sync_migrations() -> None:
    """Синхронный запуск миграций Alembic через subprocess."""
    project_root = Path(__file__).resolve().parent.parent.parent
    alembic_ini_path = project_root / "alembic.ini"

    if not alembic_ini_path.exists():
        logger.error(f"alembic.ini not found at {alembic_ini_path}")
        return

    logger.info(f"Attempting to run Alembic migrations from {project_root}...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=60
        )

        if result.returncode == 0:
            logger.info("Alembic migrations applied successfully.")
            if result.stdout:
                logger.debug(f"Alembic output:\n{result.stdout.strip()}")
        else:
            logger.error(f"Alembic migration failed with return code {result.returncode}")
            logger.error(f"Alembic stderr:\n{result.stderr.strip()}")
            logger.error(f"Alembic stdout:\n{result.stdout.strip()}")
            raise RuntimeError("Alembic migration failed")

    except Exception as e:
        logger.error(f"Failed to run Alembic subprocess: {repr(e)}")
        raise


async def _run_async_migrations() -> None:
    """
    Асинхронная обертка. Запускает миграции в отдельном потоке,
    чтобы не блокировать event loop FastAPI.
    """
    await asyncio.to_thread(_run_sync_migrations)


# ---------------------------------------------------------------------------
# Database Initialization
# ---------------------------------------------------------------------------


async def init_db():
    """Инициализация БД: применение миграций Alembic и проверка подключения."""

    # ── 1. Запуск миграций ────────────────────────────────────────────
    try:
        await _run_async_migrations()
    except Exception:
        pass

    # ── 2. Проверка подключения ───────────────────────────────────────
    async with engine.connect() as conn:
        try:
            await conn.execute(text("SELECT 1"))
            logger.info("Database connection established.")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
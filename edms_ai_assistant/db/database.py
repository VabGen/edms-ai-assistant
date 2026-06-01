# edms_ai_assistant/db/database.py
import asyncio
import contextlib
import logging
import subprocess
import sys
from collections.abc import AsyncGenerator
from pathlib import Path

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


async def get_db() -> AsyncGenerator[AsyncSession]:
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
    """Синхронный запуск миграций Alembic через subprocess.
    
    Raises:
        RuntimeError: If alembic.ini not found or migration fails.
        subprocess.TimeoutExpired: If migration takes longer than timeout.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    alembic_ini_path = project_root / "alembic.ini"

    if not alembic_ini_path.exists():
        error_msg = f"alembic.ini not found at {alembic_ini_path}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"Attempting to run Alembic migrations from {project_root}...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

        if result.returncode == 0:
            logger.info("Alembic migrations applied successfully.")
            if result.stdout:
                logger.debug(f"Alembic output:\n{result.stdout.strip()}")
        else:
            error_msg = (
                f"Alembic migration failed with return code {result.returncode}. "
                f"stderr: {result.stderr.strip() if result.stderr else 'None'}. "
                f"stdout: {result.stdout.strip() if result.stdout else 'None'}."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    except subprocess.TimeoutExpired as e:
        error_msg = f"Alembic migration timed out after {e.timeout}s"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"Failed to run Alembic subprocess: {e!r}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


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
    """Инициализация БД: применение миграций Alembic и проверка подключения.
    
    Raises:
        RuntimeError: If migration fails or database connection cannot be established.
    """

    # ── 1. Запуск миграций ────────────────────────────────────────────
    logger.info("Running database migrations...")
    try:
        await _run_async_migrations()
        logger.info("Database migrations completed successfully")
    except Exception as exc:
        logger.error("Database migration failed", exc_info=True)
        raise RuntimeError(
            f"Failed to apply database migrations: {exc}. "
            "Application cannot start without compatible database schema."
        ) from exc

    # ── 2. Проверка подключения ───────────────────────────────────────
    logger.info("Verifying database connection...")
    async with engine.connect() as conn:
        try:
            await conn.execute(text("SELECT 1"))
            logger.info("Database connection established and verified.")
        except Exception as exc:
            logger.error("Database connection verification failed", exc_info=True)
            raise RuntimeError(
                f"Database connection check failed: {exc}. "
                "Cannot start application without database connectivity."
            ) from exc

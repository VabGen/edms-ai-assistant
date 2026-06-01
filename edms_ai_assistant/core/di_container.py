"""Dependency Injection Container for EDMS AI Assistant.

Provides a structured, production-ready DI system using dependency-injector.
Replaces the ad-hoc app.state pattern with proper dependency management.

This is a pragmatic, transitional implementation that:
- Maintains backward compatibility with existing AppDeps
- Provides proper lifecycle management
- Enables testability through dependency override
- Serves as foundation for future full refactoring
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from edms_ai_assistant.config import settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import redis.asyncio as aioredis

    from edms_ai_assistant.clients.transport import IAsyncTransport
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factory Functions (must be defined before the container class)
# ---------------------------------------------------------------------------

def _create_redis_client(redis_url: str) -> aioredis.Redis:
    """Create Redis client with proper settings."""
    import redis.asyncio as aioredis
    
    return aioredis.from_url(
        redis_url,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD.get_secret_value() if settings.REDIS_PASSWORD else None,
        decode_responses=True,
    )


def _create_transport(base_url: str, timeout: int) -> IAsyncTransport:
    """Create HTTP transport."""
    from edms_ai_assistant.clients.transport import HttpxTransport
    return HttpxTransport(base_url=str(base_url), default_timeout=timeout)


def _create_chat_model() -> BaseChatModel:
    """Create chat model with current settings."""
    from edms_ai_assistant.llm import get_chat_model, reset_chat_model
    
    reset_chat_model()  # Ensure fresh instance
    return get_chat_model()


def _create_app_deps(
    transport: IAsyncTransport,
    redis: aioredis.Redis,
    chat_model: BaseChatModel,
):
    """Create AppDeps using the existing init_deps function (transitional)."""
    from edms_ai_assistant.core.deps import init_deps
    return init_deps(transport, redis, chat_model)


def _create_agent(deps, chat_model: BaseChatModel):
    """Create LangGraph agent."""
    from edms_ai_assistant.agent.agent import EdmsDocumentAgent
    return EdmsDocumentAgent(deps=deps, llm=chat_model)


class ApplicationContainer(containers.DeclarativeContainer):
    """Application DI container with core dependencies.
    
    This container provides the essential infrastructure and services
    needed by the application, using the existing AppDeps structure
    for backward compatibility during migration.
    """

    config = providers.Configuration()

    # ── Core Infrastructure ────────────────────────────────────────────────
    
    redis = providers.Singleton(
        _create_redis_client,
        redis_url=settings.REDIS_URL,
    )

    transport = providers.Singleton(
        _create_transport,
        base_url=settings.EDMS_BASE_URL,
        timeout=settings.EDMS_TIMEOUT,
    )

    chat_model = providers.Singleton(_create_chat_model)

    # ── Application Dependencies ──────────────────────────────────────────

    app_deps = providers.Singleton(
        _create_app_deps,
        transport=transport,
        redis=redis,
        chat_model=chat_model,
    )

    agent = providers.Singleton(
        _create_agent,
        deps=app_deps,
        chat_model=chat_model,
    )


# ---------------------------------------------------------------------------
# Container Lifecycle
# ---------------------------------------------------------------------------

_container: ApplicationContainer | None = None


def get_container() -> ApplicationContainer:
    """Get the global application container."""
    if _container is None:
        raise RuntimeError("DI container not initialized. Call init_container() first.")
    return _container


async def init_container() -> ApplicationContainer:
    """Initialize the DI container."""
    global _container
    
    logger.info("Initializing DI container...")
    
    _container = ApplicationContainer()
    
    # No init_resources() call - dependency-injector doesn't have this method
    # Async resources will be created when first accessed
    
    logger.info("DI container initialized successfully")
    return _container


async def shutdown_container() -> None:
    """Shutdown the DI container and clean up resources."""
    global _container
    
    if _container is not None:
        logger.info("Shutting down DI container...")
        
        # Clean up resources manually
        try:
            # Close Redis connection
            redis_instance = _container.redis()
            await redis_instance.close()
        except Exception as exc:
            logger.warning("Error closing Redis: %s", exc)
        
        try:
            # Close HTTP transport
            transport_instance = _container.transport()
            await transport_instance.close()
        except Exception as exc:
            logger.warning("Error closing transport: %s", exc)
        
        _container = None
        logger.info("DI container shut down successfully")
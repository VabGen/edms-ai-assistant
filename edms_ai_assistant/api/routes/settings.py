# edms_ai_assistant/api/routes/settings.py
"""
Settings API router.

Слой: Interface (Transport).

Endpoints:
    GET    /api/settings/meta  — feature flags (show_technical из .env)
    GET    /api/settings       — текущие эффективные технические настройки
    PATCH  /api/settings       — runtime-патч (in-memory, сброс при рестарте)
    DELETE /api/settings       — сброс к .env-дефолтам

Примечание по архитектуре:
    PATCH защищён флагом SETTINGS_PANEL_SHOW_TECHNICAL: бэкенд возвращает 403,
    если флаг выключен — клиент не может обойти UI-ограничение через прямой API.

    Пользовательские настройки (appearance/voice/documents) хранятся в
    chrome.storage.local на стороне клиента и через этот роутер НЕ проходят.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["Settings"])


# ─────────────────────────────────────────────────────────────────────────────
# /meta — feature flags
# ─────────────────────────────────────────────────────────────────────────────


class SettingsMetaResponse(BaseModel):
    """Feature flags controlling settings panel visibility in the Chrome plugin.

    Читается плагином при инициализации сайдбара.
    Позволяет администратору скрыть технический раздел без пересборки плагина.
    """

    show_technical: bool = Field(
        description=(
            "Показывать технический раздел настроек (LLM/Agent/RAG/EDMS). "
            "Управляется SETTINGS_PANEL_SHOW_TECHNICAL в .env."
        )
    )


@router.get(
    "/meta",
    response_model=SettingsMetaResponse,
    summary="Get settings panel feature flags",
    description=(
        "Returns feature flags read from server config. "
        "Plugin calls this on sidebar init to show/hide the technical section. "
        "Controlled by SETTINGS_PANEL_SHOW_TECHNICAL in .env."
    ),
)
async def get_settings_meta() -> SettingsMetaResponse:
    """Return settings panel feature flags.

    Returns:
        SettingsMetaResponse: Flags from .env via Pydantic Settings.
    """
    return SettingsMetaResponse(
        show_technical=settings.SETTINGS_PANEL_SHOW_TECHNICAL,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas (snake_case — зеркало useSettingsStore.ts mappers)
# ─────────────────────────────────────────────────────────────────────────────


class LLMSettingsSchema(BaseModel):
    """LLM runtime configuration patch. All fields optional."""

    generative_url: str | None = None
    generative_model: str | None = None
    embedding_url: str | None = None
    embedding_model: str | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=100, le=8192)
    timeout: int | None = Field(None, ge=10, le=600)
    max_retries: int | None = Field(None, ge=0, le=10)


class AgentSettingsSchema(BaseModel):
    """Agent runtime configuration patch. All fields optional."""

    max_iterations: int | None = Field(None, ge=1, le=50)
    max_context_messages: int | None = Field(None, ge=5, le=100)
    timeout: float | None = Field(None, ge=10.0, le=600.0)
    max_retries: int | None = Field(None, ge=0, le=10)
    enable_tracing: bool | None = None
    log_level: str | None = Field(None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")


class RAGSettingsSchema(BaseModel):
    """RAG pipeline configuration patch. All fields optional."""

    chunk_size: int | None = Field(None, ge=100, le=8000)
    chunk_overlap: int | None = Field(None, ge=0, le=2000)
    batch_size: int | None = Field(None, ge=1, le=100)
    embedding_batch_size: int | None = Field(None, ge=1, le=50)

    @field_validator("chunk_overlap", mode="after")
    @classmethod
    def overlap_less_than_chunk(cls, v: int | None, info: Any) -> int | None:
        """Validates chunk_overlap < chunk_size when both provided.

        Args:
            v: chunk_overlap value.
            info: Pydantic ValidationInfo.

        Returns:
            Validated value.

        Raises:
            ValueError: If chunk_overlap >= chunk_size.
        """
        if v is not None:
            chunk_size = info.data.get("chunk_size")
            if chunk_size is not None and v >= chunk_size:
                raise ValueError(
                    f"chunk_overlap ({v}) должен быть меньше chunk_size ({chunk_size})"
                )
        return v


class EDMSSettingsSchema(BaseModel):
    """EDMS client configuration patch. All fields optional."""

    base_url: str | None = None
    timeout: int | None = Field(None, ge=10, le=600)
    api_version: str | None = Field(None, max_length=10)


class UpdateSettingsRequest(BaseModel):
    """PATCH /api/settings body. All groups optional."""

    llm: LLMSettingsSchema | None = None
    agent: AgentSettingsSchema | None = None
    rag: RAGSettingsSchema | None = None
    edms: EDMSSettingsSchema | None = None


class SettingsResponse(BaseModel):
    """Current effective technical settings = .env base + runtime overrides."""

    llm: dict[str, Any]
    agent: dict[str, Any]
    rag: dict[str, Any]
    edms: dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Runtime settings store (singleton, in-memory)
# ─────────────────────────────────────────────────────────────────────────────


class _RuntimeSettingsStore:
    """In-memory store for technical settings runtime overrides.

    Singleton. Base values: config.py/.env.
    Overrides: applied via PATCH, lost on service restart.

    Для персистентности между рестартами — добавить Redis/PostgreSQL слой.
    """

    def __init__(self) -> None:
        self._overrides: dict[str, dict[str, Any]] = {}

    def apply_patch(self, patch: UpdateSettingsRequest) -> None:
        """Apply non-None fields from PATCH request body.

        Args:
            patch: Validated PATCH request body.
        """
        for group in ("llm", "agent", "rag", "edms"):
            schema = getattr(patch, group)
            if schema is None:
                continue
            data = schema.model_dump(exclude_none=True)
            if data:
                self._overrides.setdefault(group, {}).update(data)
                logger.info(
                    "Settings patched: group=%s fields=%s",
                    group,
                    list(data.keys()),
                )

    def get_current(self) -> SettingsResponse:
        """Merge .env base config + runtime overrides.

        Returns:
            SettingsResponse: Effective settings.
        """
        base: dict[str, dict[str, Any]] = {
            "llm": {
                "generative_url": str(settings.LLM_GENERATIVE_URL),
                "generative_model": settings.LLM_GENERATIVE_MODEL,
                "embedding_url": str(settings.LLM_EMBEDDING_URL),
                "embedding_model": settings.LLM_EMBEDDING_MODEL,
                "temperature": settings.LLM_TEMPERATURE,
                "max_tokens": settings.LLM_MAX_TOKENS,
                "timeout": settings.LLM_TIMEOUT,
                "max_retries": settings.LLM_MAX_RETRIES,
            },
            "agent": {
                "max_iterations": settings.AGENT_MAX_ITERATIONS,
                "max_context_messages": settings.AGENT_MAX_CONTEXT_MESSAGES,
                "timeout": settings.AGENT_TIMEOUT,
                "max_retries": settings.AGENT_MAX_RETRIES,
                "enable_tracing": settings.AGENT_ENABLE_TRACING,
                "log_level": settings.AGENT_LOG_LEVEL,
            },
            "rag": {
                "chunk_size": settings.RAG_CHUNK_SIZE,
                "chunk_overlap": settings.RAG_CHUNK_OVERLAP,
                "batch_size": settings.RAG_BATCH_SIZE,
                "embedding_batch_size": settings.RAG_EMBEDDING_BATCH_SIZE,
            },
            "edms": {
                "base_url": str(settings.EDMS_BASE_URL),
                "timeout": settings.EDMS_TIMEOUT,
                "api_version": settings.EDMS_API_VERSION,
            },
        }
        return SettingsResponse(
            llm={**base["llm"], **self._overrides.get("llm", {})},
            agent={**base["agent"], **self._overrides.get("agent", {})},
            rag={**base["rag"], **self._overrides.get("rag", {})},
            edms={**base["edms"], **self._overrides.get("edms", {})},
        )

    def reset(self) -> None:
        """Clear all overrides, restore .env defaults."""
        self._overrides.clear()
        logger.info("Runtime settings reset to .env defaults")


_store = _RuntimeSettingsStore()


def get_settings_store() -> _RuntimeSettingsStore:
    """FastAPI dependency: returns the singleton settings store.

    Returns:
        _RuntimeSettingsStore: Module-level singleton.
    """
    return _store


# ─────────────────────────────────────────────────────────────────────────────
# Technical settings routes
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "",
    response_model=SettingsResponse,
    summary="Get current effective technical settings",
)
async def get_settings(
    store: _RuntimeSettingsStore = Depends(get_settings_store),
) -> SettingsResponse:
    """Return current effective technical settings.

    Returns:
        SettingsResponse: Merged .env defaults + runtime overrides.
    """
    return store.get_current()


@router.patch(
    "",
    response_model=SettingsResponse,
    summary="Patch runtime technical settings",
    description=(
        "Applies partial update. In-memory only, resets on service restart. "
        "Returns resulting effective settings. "
        "Returns 403 if SETTINGS_PANEL_SHOW_TECHNICAL=false — "
        "client cannot bypass the UI flag via direct API call."
    ),
)
async def patch_settings(
    body: UpdateSettingsRequest,
    store: _RuntimeSettingsStore = Depends(get_settings_store),
) -> SettingsResponse:
    """Apply partial technical settings patch.

    Guards with SETTINGS_PANEL_SHOW_TECHNICAL flag — even if UI hides
    the technical section, a direct API call is also rejected when flag is false.

    Args:
        body: Partial settings update (all groups optional).
        store: Runtime settings store (injected by FastAPI).

    Returns:
        SettingsResponse: Effective settings after applying patch.

    Raises:
        HTTPException 403: SETTINGS_PANEL_SHOW_TECHNICAL=false.
        HTTPException 422: Pydantic validation error (e.g. chunk_overlap >= chunk_size).
        HTTPException 500: Unexpected store error.
    """
    if not settings.SETTINGS_PANEL_SHOW_TECHNICAL:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Технический раздел настроек отключён администратором "
                "(SETTINGS_PANEL_SHOW_TECHNICAL=false)."
            ),
        )
    try:
        store.apply_patch(body)
    except Exception as exc:
        logger.error("Failed to apply settings patch", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось применить настройки: {exc}",
        ) from exc

    logger.info(
        "Settings PATCH done, groups=%s",
        [g for g in ("llm", "agent", "rag", "edms") if getattr(body, g) is not None],
    )
    return store.get_current()


@router.delete(
    "",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Reset technical settings to .env defaults",
)
async def reset_settings(
    store: _RuntimeSettingsStore = Depends(get_settings_store),
) -> None:
    """Reset all runtime overrides, restore .env defaults.

    Args:
        store: Runtime settings store (injected by FastAPI).
    """
    store.reset()

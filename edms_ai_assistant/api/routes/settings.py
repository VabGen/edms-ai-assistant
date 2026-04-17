# edms_ai_assistant/api/routes/settings.py
"""
Settings API router.

Endpoints:
    GET    /api/settings/meta  — feature flags (show_technical из .env)
    GET    /api/settings       — текущие эффективные технические настройки
    PATCH  /api/settings       — runtime-патч (принимает camelCase от UI)
    DELETE /api/settings       — сброс к .env-дефолтам
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from edms_ai_assistant.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["Settings"])


# ─────────────────────────────────────────────────────────────────────────────
# /meta — feature flags
# ─────────────────────────────────────────────────────────────────────────────

class SettingsMetaResponse(BaseModel):
    show_technical: bool = Field(
        description="Показывать технический раздел настроек (LLM/Agent/RAG/EDMS)."
    )

@router.get("/meta", response_model=SettingsMetaResponse, summary="Get settings panel feature flags")
async def get_settings_meta() -> SettingsMetaResponse:
    return SettingsMetaResponse(show_technical=settings.SETTINGS_PANEL_SHOW_TECHNICAL)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas (Поддерживают и snake_case, и camelCase)
# ─────────────────────────────────────────────────────────────────────────────

class LLMSettingsSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    generative_url: str | None = Field(None, alias="generativeUrl")
    generative_model: str | None = Field(None, alias="generativeModel")
    embedding_url: str | None = Field(None, alias="embeddingUrl")
    embedding_model: str | None = Field(None, alias="embeddingModel")
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=100, le=8192)
    timeout: int | None = Field(None, ge=10, le=600)
    max_retries: int | None = Field(None, ge=0, le=10)

class AgentSettingsSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    max_iterations: int | None = Field(None, ge=1, le=50)
    max_context_messages: int | None = Field(None, ge=5, le=100)
    timeout: float | None = Field(None, ge=10.0, le=600.0)
    max_retries: int | None = Field(None, ge=0, le=10)
    enable_tracing: bool | None = Field(None, alias="enableTracing")
    log_level: str | None = Field(None, alias="logLevel", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

class RAGSettingsSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    chunk_size: int | None = Field(None, ge=100, le=8000, alias="chunkSize")
    chunk_overlap: int | None = Field(None, ge=0, le=2000, alias="chunkOverlap")
    batch_size: int | None = Field(None, ge=1, le=100, alias="batchSize")
    embedding_batch_size: int | None = Field(None, ge=1, le=50, alias="embeddingBatchSize")

    @field_validator("chunk_overlap", mode="after")
    @classmethod
    def overlap_less_than_chunk(cls, v: int | None, info: Any) -> int | None:
        if v is not None:
            chunk_size = info.data.get("chunk_size")
            if chunk_size is not None and v >= chunk_size:
                raise ValueError(f"chunk_overlap ({v}) должен быть меньше chunk_size ({chunk_size})")
        return v

class EDMSSettingsSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    base_url: str | None = Field(None, alias="baseUrl")
    timeout: int | None = Field(None, ge=10, le=600)

class UpdateSettingsRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    llm: LLMSettingsSchema | None = None
    agent: AgentSettingsSchema | None = None
    rag: RAGSettingsSchema | None = None
    edms: EDMSSettingsSchema | None = None

class SettingsResponse(BaseModel):
    llm: dict[str, Any]
    agent: dict[str, Any]
    rag: dict[str, Any]
    edms: dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Runtime settings store
# ─────────────────────────────────────────────────────────────────────────────

class _RuntimeSettingsStore:
    _SETTINGS_MAP: dict[str, dict[str, str]] = {
        "llm": {
            "generative_url": "LLM_GENERATIVE_URL",
            "generative_model": "LLM_GENERATIVE_MODEL",
            "embedding_url": "LLM_EMBEDDING_URL",
            "embedding_model": "LLM_EMBEDDING_MODEL",
            "temperature": "LLM_TEMPERATURE",
            "max_tokens": "LLM_MAX_TOKENS",
            "timeout": "LLM_TIMEOUT",
            "max_retries": "LLM_MAX_RETRIES",
        },
        "agent": {
            "max_iterations": "AGENT_MAX_ITERATIONS",
            "max_context_messages": "AGENT_MAX_CONTEXT_MESSAGES",
            "timeout": "AGENT_TIMEOUT",
            "max_retries": "AGENT_MAX_RETRIES",
            "enable_tracing": "AGENT_ENABLE_TRACING",
            "log_level": "AGENT_LOG_LEVEL",
        },
        "rag": {
            "chunk_size": "RAG_CHUNK_SIZE",
            "chunk_overlap": "RAG_CHUNK_OVERLAP",
            "batch_size": "RAG_BATCH_SIZE",
            "embedding_batch_size": "RAG_EMBEDDING_BATCH_SIZE",
        },
        "edms": {
            "base_url": "EDMS_BASE_URL",
            "timeout": "EDMS_TIMEOUT",
        },
    }

    def __init__(self) -> None:
        self._defaults: dict[str, Any] = {}
        self._overrides: dict[str, dict[str, Any]] = {}
        for group, field_map in self._SETTINGS_MAP.items():
            for _, attr in field_map.items():
                self._defaults[attr] = getattr(settings, attr, None)

    def apply_patch(self, patch: UpdateSettingsRequest) -> None:
        for group in ("llm", "agent", "rag", "edms"):
            schema = getattr(patch, group)
            if schema is None:
                continue

            # by_alias=False возвращает snake_case ключи, совпадающие с _SETTINGS_MAP
            data = schema.model_dump(exclude_none=True, by_alias=False)
            if not data:
                continue

            self._overrides.setdefault(group, {}).update(data)
            field_map = self._SETTINGS_MAP.get(group, {})

            for field_name, value in data.items():
                settings_attr = field_map.get(field_name)
                if not settings_attr:
                    continue

                # Принудительная мутация синглтона settings в runtime
                object.__setattr__(settings, settings_attr, value)
                logger.info("Settings patched: %s.%s = %r", group, field_name, value)

    def get_current(self) -> SettingsResponse:
        # Возвращаем в camelCase для удобства фронтенда
        return SettingsResponse(
            llm={
                "generativeUrl": str(settings.LLM_GENERATIVE_URL),
                "generativeModel": settings.LLM_GENERATIVE_MODEL,
                "embeddingUrl": str(settings.LLM_EMBEDDING_URL),
                "embeddingModel": settings.LLM_EMBEDDING_MODEL,
                "temperature": settings.LLM_TEMPERATURE,
                "maxTokens": settings.LLM_MAX_TOKENS,
                "timeout": settings.LLM_TIMEOUT,
                "maxRetries": settings.LLM_MAX_RETRIES,
            },
            agent={
                "maxIterations": settings.AGENT_MAX_ITERATIONS,
                "maxContextMessages": settings.AGENT_MAX_CONTEXT_MESSAGES,
                "timeout": settings.AGENT_TIMEOUT,
                "maxRetries": settings.AGENT_MAX_RETRIES,
                "enableTracing": settings.AGENT_ENABLE_TRACING,
                "logLevel": settings.AGENT_LOG_LEVEL,
            },
            rag={
                "chunkSize": settings.RAG_CHUNK_SIZE,
                "chunkOverlap": settings.RAG_CHUNK_OVERLAP,
                "batchSize": settings.RAG_BATCH_SIZE,
                "embeddingBatchSize": settings.RAG_EMBEDDING_BATCH_SIZE,
            },
            edms={
                "baseUrl": str(settings.EDMS_BASE_URL),
                "timeout": settings.EDMS_TIMEOUT,
            },
        )

    def reset(self) -> None:
        for attr, original_value in self._defaults.items():
            if original_value is not None:
                object.__setattr__(settings, attr, original_value)
        self._overrides.clear()
        logger.info("Runtime settings reset to .env defaults")

_store = _RuntimeSettingsStore()

def get_settings_store() -> _RuntimeSettingsStore:
    return _store

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=SettingsResponse, summary="Get current effective technical settings")
async def get_settings(store: _RuntimeSettingsStore = Depends(get_settings_store)) -> SettingsResponse:
    return store.get_current()

@router.patch("", response_model=SettingsResponse, summary="Patch runtime technical settings")
async def patch_settings(
    body: UpdateSettingsRequest,
    store: _RuntimeSettingsStore = Depends(get_settings_store),
) -> SettingsResponse:
    if not settings.SETTINGS_PANEL_SHOW_TECHNICAL:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Технический раздел отключен.")

    try:
        store.apply_patch(body)
    except Exception as exc:
        logger.error("Failed to apply settings patch", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return store.get_current()

@router.delete("", status_code=status.HTTP_204_NO_CONTENT, summary="Reset technical settings")
async def reset_settings(store: _RuntimeSettingsStore = Depends(get_settings_store)) -> None:
    store.reset()

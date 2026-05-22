"""
Типизированные исключения модуля суммаризации.

Иерархия:
    SummarizerError                          — базовое (все исключения модуля)
    ├── TextExtractionError                  — не удалось извлечь текст из файла
    ├── LLMError                             — ошибки LLM-провайдера
    │   ├── LLMTransportError                — сетевые ошибки (httpx)
    │   ├── LLMRateLimitedError              — 429
    │   ├── LLMServerError                   — 5xx
    │   ├── LLMClientError                   — 4xx (кроме 429), не ретраим
    │   └── LLMResponseError                 — невалидный/обрезанный ответ
    ├── ValidationError                      — структурированный вывод не прошёл pydantic
    ├── PipelineError                        — общий сбой пайплайна
    │   └── MapStageError                    — все Map-чанки упали
    └── CacheError                           — внутренний сбой кэша (логируется, но не падает)

Маппинг на HTTP-коды (см. api/router.py):
    TextExtractionError, ValidationError → 422
    LLMRateLimitedError                  → 503
    LLMClientError                       → 502
    LLMServerError, LLMTransportError    → 504
    PipelineError, LLMError, прочее      → 500
"""

from __future__ import annotations


class SummarizerError(Exception):
    """Базовое исключение модуля суммаризации."""


# ---------------------------------------------------------------------------
# Извлечение текста / валидация
# ---------------------------------------------------------------------------

class TextExtractionError(SummarizerError, ValueError):
    """Не удалось извлечь текст из файла (PDF/DOCX/etc)."""


class ValidationError(SummarizerError, ValueError):
    """Структурированный вывод LLM не прошёл pydantic-валидацию."""


# ---------------------------------------------------------------------------
# LLM-провайдер
# ---------------------------------------------------------------------------

class LLMError(SummarizerError):
    """Базовый класс для всех ошибок LLM-провайдера."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMTransportError(LLMError):
    """Сетевая ошибка / таймаут httpx."""


class LLMRateLimitedError(LLMError):
    """HTTP 429 от LLM-провайдера."""


class LLMServerError(LLMError):
    """HTTP 5xx от LLM-провайдера (после исчерпания ретраев)."""


class LLMClientError(LLMError):
    """HTTP 4xx (кроме 429) — постоянная ошибка, ретраить нельзя."""


class LLMResponseError(LLMError):
    """LLM вернул невалидный JSON или малформированный ответ."""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PipelineError(SummarizerError):
    """Общий сбой пайплайна суммаризации."""


class MapStageError(PipelineError):
    """Все чанки Map-стадии упали — продолжать невозможно."""


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class CacheError(SummarizerError):
    """Внутренний сбой кэша. По правилам модуля — best-effort, логируется и не падает."""


__all__ = [
    "CacheError",
    "LLMClientError",
    "LLMError",
    "LLMRateLimitedError",
    "LLMResponseError",
    "LLMServerError",
    "LLMTransportError",
    "MapStageError",
    "PipelineError",
    "SummarizerError",
    "TextExtractionError",
    "ValidationError",
]

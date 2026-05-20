# edms_ai_assistant/core/exceptions.py

from typing import Any
from fastapi import HTTPException


# ── API / HTTP Exceptions ────────────────────────────────────────────────────


class AppException(HTTPException):
    """Базовое исключение для бизнес-логики приложения."""

    def __init__(
            self, status_code: int, detail: str, error_code: str = "GENERIC_ERROR"
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code


class LLMException(AppException):
    """Ошибка при обращении к LLM."""

    def __init__(self, detail: str = "LLM Generation failed"):
        super().__init__(status_code=503, detail=detail, error_code="LLM_ERROR")


class DocumentTooLargeException(AppException):
    """Документ превышает лимиты обработки."""

    def __init__(self, detail: str = "Document exceeds processing limits"):
        super().__init__(status_code=413, detail=detail, error_code="DOC_TOO_LARGE")


class CacheException(AppException):
    """Ошибка работы с кэшем."""

    def __init__(self, detail: str = "Cache operation failed"):
        super().__init__(status_code=500, detail=detail, error_code="CACHE_ERROR")


# ── EDMS Integration Exceptions ──────────────────────────────────────────────


class EdmsError(Exception):
    """Базовое исключение для всех ошибок интеграции с EDMS."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        self.context = context or {}
        super().__init__(message)


class EdmsConnectionError(EdmsError):
    """Ошибки сети (DNS, Timeout, Connection refused)."""
    pass


class EdmsClientError(EdmsError):
    """Ошибки 4xx (клиентские)."""

    def __init__(self, message: str, status_code: int, context: dict[str, Any] | None = None):
        self.status_code = status_code
        super().__init__(message, context)


class EdmsNotFoundError(EdmsClientError):
    """Специфичный 404."""
    pass


class EdmsAuthenticationError(EdmsClientError):
    """Ошибки 401/403."""
    pass


class EdmsValidationError(EdmsClientError):
    """Ошибка валидации 422."""
    pass


class EdmsServerError(EdmsError):
    """Ошибки 5xx (серверные)."""

    def __init__(self, message: str, status_code: int, context: dict[str, Any] | None = None):
        self.status_code = status_code
        super().__init__(message, context)


class DocumentServiceError(EdmsError):
    """Базовая ошибка сервисного слоя документов."""

    def __init__(self, message: str, document_id: str | None = None):
        self.document_id = document_id
        super().__init__(message)


class DocumentNotFoundError(DocumentServiceError):
    """Документ не найден — API вернул пустой ответ или 404."""


class DocumentOperationError(DocumentServiceError):
    """Ошибка операции записи (start / cancel / execute / control)."""

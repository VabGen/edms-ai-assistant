from fastapi import HTTPException

class AppException(HTTPException):
    """Базовое исключение для бизнес-логики приложения."""
    def __init__(self, status_code: int, detail: str, error_code: str = "GENERIC_ERROR"):
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
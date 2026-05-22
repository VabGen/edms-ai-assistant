"""
Типизированная конфигурация модуля суммаризации.

Извлекает значения из глобального `Settings` через `from_app_settings`,
предоставляя строгую валидацию и явные дефолты вместо россыпи `getattr(...)`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, SecretStr


class SummarizerConfig(BaseModel):
    """Все runtime-параметры пайплайна суммаризации в одном месте."""

    model_config = {"frozen": True}

    # LLM
    llm_base_url: str = Field(default="http://localhost:11434/v1")
    llm_model: str
    llm_api_key: SecretStr | None = None
    llm_timeout_s: float = Field(default=120.0, gt=0)

    # Pipeline
    context_window_tokens: int = Field(default=4096, gt=0)
    max_output_tokens: int = Field(default=4096, gt=0)
    max_concurrent_map: int = Field(default=6, gt=0)

    # Telemetry
    otlp_endpoint: str | None = None

    # Optional quality scoring (LLM-as-judge); пока не реализовано.
    quality_model: str | None = None

    @classmethod
    def from_app_settings(cls, settings: object) -> SummarizerConfig:
        """Аккуратно достаёт нужные поля из глобальных settings."""
        api_key = getattr(settings, "LLM_API_KEY", None) or getattr(
            settings, "OPENAI_API_KEY", None
        )

        return cls(
            llm_base_url=str(
                getattr(settings, "LLM_GENERATIVE_URL", "http://localhost:11434/v1")
            ),
            llm_model=str(settings.LLM_GENERATIVE_MODEL),
            llm_api_key=api_key if api_key is not None else None,
            llm_timeout_s=float(getattr(settings, "LLM_TIMEOUT", 120.0)),
            context_window_tokens=int(
                getattr(settings, "SUMMARIZER_CONTEXT_WINDOW", 4096)
            ),
            max_output_tokens=int(
                getattr(settings, "SUMMARIZER_MAX_OUTPUT_TOKENS", 4096)
            ),
            max_concurrent_map=int(
                getattr(settings, "SUMMARIZER_MAX_CONCURRENT_MAP", 6)
            ),
            otlp_endpoint=getattr(settings, "TELEMETRY_ENDPOINT", None),
            quality_model=getattr(settings, "SUMMARIZER_QUALITY_MODEL", None),
        )

    def resolved_api_key(self) -> str:
        """Возвращает строковое значение API-ключа с фолбэком 'ollama'."""
        if self.llm_api_key is None:
            return "ollama"
        value = self.llm_api_key.get_secret_value().strip()
        return value or "ollama"

    def normalized_base_url(self) -> str:
        base = self.llm_base_url.rstrip("/")
        return base if base.endswith("/v1") else base + "/v1"

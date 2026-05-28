# edms_ai_assistant/config.py
"""Production-ready configuration with validation, security, and environment separation."""

from __future__ import annotations

import os

from pydantic import (
    Field,
    HttpUrl,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class EdmsSettings(BaseSettings):
    """Настройки для интеграции с EDMS (СЭД)."""

    model_config = SettingsConfigDict(
        env_prefix="EDMS_",
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: HttpUrl = Field(default="http://127.0.0.1:8098")
    timeout: int = Field(default=30, ge=10, le=600)
    long_timeout: int = Field(default=120, ge=10, le=600)
    api_version: str = "v1"

    # Реквизиты для авторизации (если токен получается по client_credentials)
    client_id: SecretStr | None = None
    client_secret: SecretStr | None = None

    # MCP & Vector DB
    mcp_url: HttpUrl | None = Field(
        default=None, description="FastMCP HTTP transport URL"
    )
    mcp_port: int = 9000


class Settings(BaseSettings):
    """
    Application configuration with strict validation.
    """

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application & K8s ────────────────────────────────────────────────────
    ENVIRONMENT: str = Field(
        default="development", pattern="^(development|staging|production)$"
    )
    APP_VERSION: str = Field(
        default="0.0.0-dev", description="Set by CI/CD from Git tags"
    )
    BUILD_COMMIT: str | None = Field(
        default=None, description="Set by CI/CD from Git SHA"
    )
    API_PORT: int = Field(default=8000, ge=1, le=65535)
    DEBUG: bool = Field(default=False)

    # ── Security ─────────────────────────────────────────────────────────────
    JWT_SECRET_KEY: SecretStr = Field(default="change-me-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 60

    # ── CORS ─────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: str = Field(default="*")

    @property
    def allowed_origins_list(self) -> list[str]:
        """Возвращает распарсенный список доменов для CORSMiddleware."""
        if self.ALLOWED_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

    @property
    def edms_base_url(self) -> str:
        return str(edms_settings.base_url)

    @property
    def edms_timeout(self) -> int:
        return edms_settings.timeout

    @property
    def edms_api_version(self) -> str:
        return edms_settings.api_version

    @property
    def chancellor_next_base_url(self) -> str:
        """Alias for backward compatibility with existing clients."""
        return str(edms_settings.base_url)

    @property
    def edms_mcp_url(self) -> HttpUrl | None:
        return edms_settings.mcp_url

    @property
    def edms_mcp_port(self) -> int:
        return edms_settings.mcp_port

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD.get_secret_value()}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def redis_url(self) -> str:
        if self.REDIS_PASSWORD:
            return (
                f"redis://:{self.REDIS_PASSWORD.get_secret_value()}@"
                f"{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
            )
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def allowed_extensions_list(self) -> set[str]:
        return {ext.strip().lower() for ext in self.ALLOWED_FILE_EXTENSIONS.split(",")}

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    # ── LLM Configuration ────────────────────────────────────────────────────
    LLM_GENERATIVE_URL: HttpUrl = Field(
        default="http://model-generative.shared.du.iba/v1"
    )
    LLM_GENERATIVE_MODEL: str = Field(default="generative-model")

    LLM_EMBEDDING_URL: HttpUrl = Field(
        default="http://model-embedding.shared.du.iba/v1"
    )
    LLM_EMBEDDING_MODEL: str = "embedding-model"

    LLM_API_KEY: SecretStr | None = None
    OPENAI_API_KEY: SecretStr | None = None

    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int | None = Field(default=2048, ge=100, le=8192)
    LLM_TIMEOUT: int = Field(default=120, ge=10, le=600)
    LLM_MAX_RETRIES: int = Field(default=3, ge=0, le=10)
    LLM_REQUEST_TIMEOUT: int = Field(default=120, ge=10, le=600)
    LLM_STREAM_USAGE: bool = False

    # Ollama local-backend specific
    LLM_OLLAMA_NUM_CTX: int = Field(default=4096, ge=512, le=131072)
    LLM_OLLAMA_NUM_PREDICT: int = Field(default=512, ge=64, le=8192)

    # ── Embedding Configuration ──────────────────────────────────────────────
    EMBEDDING_TIMEOUT: int = Field(default=120, ge=10, le=600)
    EMBEDDING_MAX_RETRIES: int = Field(default=3, ge=0, le=10)
    EMBEDDING_REQUEST_TIMEOUT: int = Field(default=120, ge=10, le=600)
    EMBEDDING_CTX_LENGTH: int = 8191
    EMBEDDING_CHUNK_SIZE: int = 1000
    EMBEDDING_MAX_RETRIES_PER_REQUEST: int = 6
    EMBEDDING_DIM: int = Field(default=1536, description="Vector dimensions for Qdrant")

    # ── Legacy properties (Aliases for backward compatibility) ────────────────

    @property
    def EDMS_BASE_URL(self) -> str:
        return self.edms_base_url

    @property
    def EDMS_TIMEOUT(self) -> int:
        return self.edms_timeout

    @property
    def EDMS_API_VERSION(self) -> str:
        return self.edms_api_version

    @property
    def CHANCELLOR_NEXT_BASE_URL(self) -> str:
        return self.chancellor_next_base_url

    @property
    def EDMS_MCP_URL(self) -> HttpUrl | None:
        return self.edms_mcp_url

    @property
    def EDMS_MCP_PORT(self) -> int:
        return self.edms_mcp_port

    @property
    def DATABASE_URL(self) -> str:
        return self.database_url

    @property
    def REDIS_URL(self) -> str:
        return self.redis_url

    @property
    def ALLOWED_EXTENSIONS_LIST(self) -> set[str]:
        return self.allowed_extensions_list

    @property
    def MAX_FILE_SIZE_BYTES(self) -> int:
        return self.max_file_size_bytes

    # ── Vector DB ────────────────────────────────────────────────────────────
    QDRANT_URL: HttpUrl = Field(default="http://qdrant:6333")

    # ── Database Configuration ───────────────────────────────────────────────
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: SecretStr = Field(default="password")
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "ai_assistant"

    CHECKPOINT_DB_URL: str | None = None
    SQL_DB_URL: str | None = None

    # ── Redis Configuration ──────────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: SecretStr | None = None
    CACHE_TTL_SECONDS: int = Field(default=300, ge=30, le=86400)

    # ── File Upload Configuration ────────────────────────────────────────────
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = Field(default=50, ge=1, le=500)
    ALLOWED_FILE_EXTENSIONS: str = ".docx,.doc,.pdf,.txt,.rtf,.xlsx,.xls,.pptx"

    # ── Agent Configuration ──────────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = Field(default=15, ge=1, le=50)
    AGENT_MAX_CONTEXT_MESSAGES: int = Field(default=20, ge=5, le=100)
    AGENT_TIMEOUT: float = Field(default=120.0, ge=10.0, le=600.0)
    AGENT_LLM_TIMEOUT: float = Field(default=120.0, ge=10.0, le=600.0)
    AGENT_ENABLE_TRACING: bool = False
    AGENT_LOG_LEVEL: str = "INFO"
    AGENT_MAX_RETRIES: int = 3
    AGENT_LEAN_PROMPT: bool = Field(default=False)

    SETTINGS_PANEL_SHOW_TECHNICAL: bool = Field(default=True)

    # ── RAG Configuration ────────────────────────────────────────────────────
    RAG_BATCH_SIZE: int = 20
    RAG_CHUNK_SIZE: int = 1200
    RAG_CHUNK_OVERLAP: int = 300
    RAG_EMBEDDING_BATCH_SIZE: int = 10
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # ── Rate Limiting ────────────────────────────────────────────────────────
    RATE_LIMIT_MAX_REQUESTS: int = Field(default=10, ge=1, le=1000)
    RATE_LIMIT_WINDOW_SECONDS: int = Field(default=60, ge=10, le=3600)

    # ── Logging Configuration ────────────────────────────────────────────────
    LOGGING_LEVEL: str = Field(
        default="ERROR", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    LOGGING_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    LOGGING_INCLUDE_TRACE_ID: bool = True

    # ── OpenTelemetry ─────────────────────────────────────────────────────────
    OTEL_ENABLED: bool = Field(
        default=False, description="Enable FastAPI & HTTPX auto-instrumentation"
    )

    # ── Summarization Configuration ──────────────────────────────────────────
    SUMMARIZER_CONTEXT_WINDOW: int = Field(default=4096)
    SUMMARIZER_QUALITY_MODEL: str | None = Field(default=None)
    SUMMARIZER_MAX_CONCURRENT_MAP: int = Field(default=6, ge=1, le=50)
    SUMMARIZER_L1_TTL_SECONDS: int = Field(default=3600)
    SUMMARIZER_L2_TTL_SECONDS: int = Field(default=2_592_000)

    # ── Validators ───────────────────────────────────────────────────────────

    @field_validator("DEBUG", mode="before")
    @classmethod
    def set_debug_default(cls, v: bool | None, info) -> bool:
        if v is not None:
            return v
        env = info.data.get("ENVIRONMENT", "development")
        return env == "development"

    @field_validator("LOGGING_LEVEL", mode="before")
    @classmethod
    def set_log_level_default(cls, v: str | None, info) -> str:
        if v is not None:
            return v
        env = info.data.get("ENVIRONMENT", "development")
        return "DEBUG" if env == "development" else "INFO"

    @model_validator(mode="after")
    def enforce_production_security(self) -> Settings:
        """Запрещает запуск в production с дефолтными секретами."""
        if self.ENVIRONMENT == "production":
            if self.JWT_SECRET_KEY.get_secret_value() == "change-me-in-production":
                raise ValueError(
                    "JWT_SECRET_KEY must be changed from default in production!"
                )
            if self.POSTGRES_PASSWORD.get_secret_value() == "password":
                raise ValueError(
                    "POSTGRES_PASSWORD must be changed from default in production!"
                )
        return self


# ── Global settings instances ────────────────────────────────────────────────
edms_settings = EdmsSettings()
settings = Settings()

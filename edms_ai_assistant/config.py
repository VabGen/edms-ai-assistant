# edms_ai_assistant/config.py
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm__generative: str = "http://model-generative.shared.du.iba/v1"
    llm__generative_model: str = "default-llm-model"
    llm__embedding: str = "http://model-embedding.shared.du.iba/v1"
    llm__embedding_model: str = "default-embedding-model"
    llm_api_key: str | None = None
    openai_api_key: str | None = None

    llm_temperature: float = 0.6
    llm_max_tokens: int | None = None
    llm_timeout: int = 120
    llm_max_retries: int = 3
    llm_request_timeout: int = 120
    llm_stream_usage: bool = True

    embedding_timeout: int = 120
    embedding_max_retries: int = 3
    embedding_request_timeout: int = 120
    embedding_ctx_length: int = 8191
    embedding_chunk_size: int = 1000
    embedding_max_retries_per_request: int = 6

    ENVIRONMENT: str = "development"

    chancellor_next_base_url: str = "http://127.0.0.1:8098"
    edms_timeout: int = 120
    chroma_persist_dir: str = "./chroma_db"
    checkpoint_db_url: str = "postgresql://postgres:1234@localhost:5432/postgres"
    sql_db_url: str = "postgresql://postgres:1234@localhost:5432/postgres"
    api_port: int = 8000
    debug: bool = True
    redis_url: str = "redis://127.0.0.1:6379/0"
    agent_enable_tracing: bool = True
    agent_log_level: str = "INFO"
    agent_max_retries: int = 3
    logging_level: str = "DEBUG"
    logging_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    telemetry_enabled: bool = True
    telemetry_endpoint: str = "http://127.0.0.1:8098"
    rag_batch_size: int = 20
    rag_chunk_size: int = 1200
    rag_chunk_overlap: int = 300
    rag_embedding_batch_size: int = 10
    react_app_api_url: str = "http://model-generative.shared.du.iba/v1"

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

    @property
    def LOGGING_LEVEL(self) -> str:
        return getattr(self, "logging_level", self.logging_level)

    @property
    def LLM_ENDPOINT(self) -> str:
        return getattr(self, "llm__generative", self.llm__generative)

    @property
    def EMBEDDING_ENDPOINT(self) -> str:
        return getattr(self, "llm__embedding", self.llm__embedding)

    @property
    def LLM_MODEL_NAME(self) -> str:
        return getattr(self, "llm__generative_model", self.llm__generative_model)

    @property
    def EMBEDDING_MODEL_NAME(self) -> str:
        return getattr(self, "llm__embedding_model", self.llm__embedding_model)

    @property
    def LLM_API_KEY(self) -> str | None:
        return getattr(self, "llm_api_key", self.llm_api_key)

    @property
    def OPENAI_API_KEY(self) -> str | None:
        return getattr(self, "openai_api_key", self.openai_api_key)

    @property
    def LLM_TEMPERATURE(self) -> float:
        return getattr(self, "llm_temperature", self.llm_temperature)

    @property
    def LLM_MAX_TOKENS(self) -> int | None:
        return getattr(self, "llm_max_tokens", self.llm_max_tokens)

    @property
    def LLM_TIMEOUT(self) -> int:
        return getattr(self, "llm_timeout", self.llm_timeout)

    @property
    def LLM_MAX_RETRIES(self) -> int:
        return getattr(self, "llm_max_retries", self.llm_max_retries)

    @property
    def LLM_REQUEST_TIMEOUT(self) -> int:
        return getattr(self, "llm_request_timeout", self.llm_request_timeout)

    @property
    def LLM_STREAM_USAGE(self) -> bool:
        return getattr(self, "llm_stream_usage", self.llm_stream_usage)

    @property
    def EMBEDDING_TIMEOUT(self) -> int:
        return getattr(self, "embedding_timeout", self.embedding_timeout)

    @property
    def EMBEDDING_MAX_RETRIES(self) -> int:
        return getattr(self, "embedding_max_retries", self.embedding_max_retries)

    @property
    def EMBEDDING_REQUEST_TIMEOUT(self) -> int:
        return getattr(
            self, "embedding_request_timeout", self.embedding_request_timeout
        )

    @property
    def EMBEDDING_CTX_LENGTH(self) -> int:
        return getattr(self, "embedding_ctx_length", self.embedding_ctx_length)

    @property
    def EMBEDDING_CHUNK_SIZE(self) -> int:
        return getattr(self, "embedding_chunk_size", self.embedding_chunk_size)

    @property
    def EMBEDDING_MAX_RETRIES_PER_REQUEST(self) -> int:
        return getattr(
            self,
            "embedding_max_retries_per_request",
            self.embedding_max_retries_per_request,
        )

    @property
    def CHANCELLOR_NEXT_BASE_URL(self) -> str:
        return getattr(self, "chancellor_next_base_url", self.chancellor_next_base_url)

    @property
    def EDMS_TIMEOUT(self):
        return getattr(self, "edms_timeout", self.edms_timeout)

    @property
    def CHROMA_PERSIST_DIR(self) -> str:
        return getattr(self, "chroma_persist_dir", self.chroma_persist_dir)

    @property
    def CHECKPOINT_DB_URL(self) -> str:
        return getattr(self, "checkpoint_db_url", self.checkpoint_db_url)

    @property
    def SQL_DB_URL(self) -> str:
        return getattr(self, "sql_db_url", self.sql_db_url)

    @property
    def API_PORT(self) -> int:
        return getattr(self, "api_port", self.api_port)

    @property
    def DEBUG(self) -> bool:
        return getattr(self, "debug", self.debug)

    @property
    def REDIS_URL(self) -> str:
        return getattr(self, "redis_url", self.redis_url)

    @property
    def AGENT_ENABLE_TRACING(self) -> bool:
        return getattr(self, "agent_enable_tracing", self.agent_enable_tracing)

    @property
    def AGENT_LOG_LEVEL(self) -> str:
        return getattr(self, "agent_log_level", self.agent_log_level)

    @property
    def AGENT_MAX_RETRIES(self) -> int:
        return getattr(self, "agent_max_retries", self.agent_max_retries)

    @property
    def LOGGING_LEVEL(self) -> str:
        return getattr(self, "logging_level", self.logging_level)

    @property
    def LOGGING_FORMAT(self) -> str:
        return getattr(self, "logging_format", self.logging_format)

    @property
    def TELEMETRY_ENABLED(self) -> bool:
        return getattr(self, "telemetry_enabled", self.telemetry_enabled)

    @property
    def TELEMETRY_ENDPOINT(self) -> str:
        return getattr(self, "telemetry_endpoint", self.telemetry_endpoint)

    @property
    def RAG_BATCH_SIZE(self) -> int:
        return getattr(self, "rag_batch_size", self.rag_batch_size)

    @property
    def RAG_CHUNK_SIZE(self) -> int:
        return getattr(self, "rag_chunk_size", self.rag_chunk_size)

    @property
    def RAG_CHUNK_OVERLAP(self) -> int:
        return getattr(self, "rag_chunk_overlap", self.rag_chunk_overlap)

    @property
    def RAG_EMBEDDING_BATCH_SIZE(self) -> int:
        return getattr(self, "rag_embedding_batch_size", self.rag_embedding_batch_size)

    @property
    def REACT_APP_API_URL(self) -> str:
        return getattr(self, "react_app_api_url", self.react_app_api_url)


settings = Settings()

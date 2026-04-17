"""
LLM and Embedding model initialization.

⚠️ ARCHITECTURE NOTICE (Runtime Settings):
Эти функции читают параметры из глобального объекта `settings`, который может быть 
изменен в runtime через PATCH /api/settings или Tool update_runtime_settings.
ПОЭТОМУ ЗАПРЕЩАЕТСЯ кешировать результат этих функций на уровне модуля или в __init__ 
агента (например: self.llm = get_chat_model()). 
Вызывайте get_chat_model() ВНУТРИ метода execute/run агента при каждом новом запросе!
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from edms_ai_assistant.config import settings

logger = logging.getLogger(__name__)


def _normalize_url(url: object) -> str:
    """Strip trailing slash from URL."""
    return str(url).rstrip("/")


def _detect_backend(base_url: str, model_name: str) -> str:
    """Detect the appropriate LLM backend from URL and model name."""
    is_ollama_endpoint = "11434" in base_url or "ollama" in base_url.lower()
    is_cloud_model = "cloud" in model_name.lower()

    if is_ollama_endpoint and is_cloud_model:
        return "openai_ollama"
    if is_ollama_endpoint:
        return "ollama_local"
    return "openai"


def get_chat_model() -> BaseLanguageModel:
    """Create a chat model instance from CURRENT runtime settings.

    Returns:
        Configured LangChain chat model instance.
    Raises:
        RuntimeError: If the model cannot be initialized.
    """
    # Читаем АКТУАЛЬНЫЕ значения из настроек (могут быть изменены через чат/UI)
    base_url = _normalize_url(settings.LLM_GENERATIVE_URL)
    model_name = settings.LLM_GENERATIVE_MODEL
    temperature = settings.LLM_TEMPERATURE
    max_tokens = settings.LLM_MAX_TOKENS
    timeout = settings.LLM_TIMEOUT
    max_retries = settings.LLM_MAX_RETRIES

    backend = _detect_backend(base_url, model_name)

    logger.info(
        "Initializing chat model",
        extra={
            "backend": backend,
            "base_url": base_url,
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        },
    )

    if backend == "ollama_local":
        try:
            from langchain_ollama import ChatOllama
            model = ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=temperature,
                num_predict=512,
                num_ctx=4096,
                timeout=timeout,
                streaming=True,
                seed=42,
                top_p=0.9,
            )
            return model
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize ChatOllama local: {exc}") from exc

    if backend == "openai_ollama":
        try:
            from langchain_openai import ChatOpenAI
            openai_base = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
            llm_params: dict = {
                "base_url": openai_base,
                "api_key": "ollama",
                "model": model_name,
                "temperature": temperature,
                "timeout": timeout,
                "max_retries": max_retries,
                "streaming": True,
            }
            if max_tokens:
                llm_params["max_tokens"] = max_tokens
            return ChatOpenAI(**llm_params)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize ChatOpenAI→Ollama: {exc}") from exc

    # OpenAI-compatible
    try:
        from langchain_openai import ChatOpenAI
        api_key: str = "placeholder-key"
        if settings.OPENAI_API_KEY:
            api_key = settings.OPENAI_API_KEY.get_secret_value()
        elif settings.LLM_API_KEY:
            api_key = settings.LLM_API_KEY.get_secret_value()

        llm_params = {
            "base_url": base_url,
            "api_key": api_key,
            "model": model_name,
            "temperature": temperature,
            "timeout": timeout,
            "max_retries": max_retries,
            "streaming": True,
        }
        if max_tokens:
            llm_params["max_tokens"] = max_tokens
        return ChatOpenAI(**llm_params)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize ChatOpenAI: {exc}") from exc


def get_embedding_model() -> Embeddings:
    """Create an embedding model instance from CURRENT runtime settings."""
    from langchain_openai import OpenAIEmbeddings

    base_url = _normalize_url(settings.LLM_EMBEDDING_URL)
    model_name = settings.LLM_EMBEDDING_MODEL

    api_key: str = "placeholder-key"
    if settings.LLM_API_KEY:
        api_key = settings.LLM_API_KEY.get_secret_value()
    elif settings.OPENAI_API_KEY:
        api_key = settings.OPENAI_API_KEY.get_secret_value()

    logger.info("Initializing embedding model", extra={"base_url": base_url, "model": model_name})

    try:
        embedding_model = OpenAIEmbeddings(
            openai_api_base=base_url,
            openai_api_key=api_key,
            model=model_name,
            request_timeout=getattr(settings, "EMBEDDING_REQUEST_TIMEOUT", 120),
            max_retries=getattr(settings, "EMBEDDING_MAX_RETRIES", 3),
            chunk_size=getattr(settings, "EMBEDDING_CHUNK_SIZE", 1000),
        )
        return embedding_model
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize embedding model: {exc}") from exc
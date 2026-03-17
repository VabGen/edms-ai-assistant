# edms_ai_assistant/llm.py
"""
LLM and Embedding model initialization with caching and error handling.
"""

import logging
import functools

from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from edms_ai_assistant.config import settings

logger = logging.getLogger(__name__)


def _normalize_url(url) -> str:
    """
    Normalize URL to string and strip trailing slash.
    """
    url_str = str(url).rstrip("/")
    return url_str


@functools.lru_cache(maxsize=1)
def get_chat_model():
    settings_kwargs = {
        "model": "gpt-4o-mini",
        "temperature": 0.6,
        "openai_api_base": "https://api.proxyapi.ru/openai/v1",
        "openai_api_key": settings.OPENAI_API_KEY,
        "max_retries": 5,
        "timeout": 90,
        "streaming": True,
        "max_tokens": 4096,
        "seed": 42,
        "top_p": 0.0000001,
    }

    try:
        model = ChatOpenAI(**settings_kwargs)
        logger.info(f"LLM Model '{settings_kwargs['model']}' успешно инициализирована.")
        return model
    except Exception as e:
        logger.error(f"Ошибка при инициализации LLM: {e}")
        raise


@functools.lru_cache(maxsize=1)
def get_chat_modell() -> BaseLanguageModel:
    """
    Initialize and cache the chat model instance.
    """
    logger.info(
        f"Инициализация ChatModel: endpoint={settings.LLM_GENERATIVE_URL}, "
        f"model={settings.LLM_GENERATIVE_MODEL}, temperature={settings.LLM_TEMPERATURE}"
    )

    llm_params = {
        "base_url": _normalize_url(settings.LLM_GENERATIVE_URL),
        "api_key": (
            settings.LLM_API_KEY.get_secret_value()
            if settings.LLM_API_KEY
            else "placeholder-key"
        ),
        "model": settings.LLM_GENERATIVE_MODEL,
        "temperature": settings.LLM_TEMPERATURE,
        "timeout": settings.LLM_TIMEOUT,
        "max_retries": settings.LLM_MAX_RETRIES,
        "streaming": False,
        "model_kwargs": {"extra_body": {"skip_special_tokens": False}},
    }

    optional = {
        "max_tokens": settings.LLM_MAX_TOKENS,
        "request_timeout": getattr(settings, "LLM_REQUEST_TIMEOUT", None),
        "default_headers": getattr(settings, "LLM_DEFAULT_HEADERS", None),
        "default_query": getattr(settings, "LLM_DEFAULT_QUERY", None),
    }
    for key, value in optional.items():
        if value is not None:
            llm_params[key] = value

    llm_params = {k: v for k, v in llm_params.items() if v is not None}

    logger.debug(f"ChatOpenAI parameters: {llm_params}")

    try:
        llm = ChatOpenAI(**llm_params)
        logger.info(f"ChatModel инициализирован: {type(llm).__name__}")
        return llm
    except Exception as e:
        logger.error(f"Ошибка инициализации ChatModel: {e}", exc_info=True)
        raise


@functools.lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """
    Initialize and cache the embedding model instance.
    """
    logger.info(
        f"Инициализация EmbeddingModel: {settings.LLM_EMBEDDING_URL}, "
        f"model: {settings.LLM_EMBEDDING_MODEL}"
    )

    embedding_params = {
        "openai_api_base": _normalize_url(settings.LLM_EMBEDDING_URL),
        "openai_api_key": (
            settings.LLM_API_KEY.get_secret_value()
            if settings.LLM_API_KEY
            else "placeholder-key"
        ),
        "model": settings.LLM_EMBEDDING_MODEL,
        "request_timeout": getattr(settings, "EMBEDDING_REQUEST_TIMEOUT", 120),
        "max_retries": getattr(settings, "EMBEDDING_MAX_RETRIES", 3),
        "default_headers": getattr(settings, "EMBEDDING_DEFAULT_HEADERS", None),
        "default_query": getattr(settings, "EMBEDDING_DEFAULT_QUERY", None),
        "chunk_size": getattr(settings, "EMBEDDING_CHUNK_SIZE", 1000),
    }

    embedding_params = {k: v for k, v in embedding_params.items() if v is not None}

    try:
        embedding_model = OpenAIEmbeddings(**embedding_params)
        logger.info(f"EmbeddingModel инициализирован: {type(embedding_model).__name__}")
        return embedding_model
    except Exception as e:
        logger.error(f"Ошибка инициализации EmbeddingModel: {e}", exc_info=True)
        raise

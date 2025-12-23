# edms_ai_assistant/llm.py
import logging
import functools
from langchain_core.language_models import BaseLanguageModel, BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from edms_ai_assistant.config import settings
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


# @functools.lru_cache(maxsize=1)
# def get_chat_model() -> BaseLanguageModel:
#     """
#     Инициализирует и кэширует ChatModel.
#     Использует lru_cache для обеспечения инициализации только один раз.
#     """
#     logger.info(
#         f"Инициализация ChatModel: {settings.LLM_ENDPOINT}, модель: {settings.LLM_MODEL_NAME}"
#     )
#
#     llm_params = {
#         "openai_api_base": settings.LLM_ENDPOINT,
#         "openai_api_key": settings.LLM_API_KEY or "placeholder-key",
#         "model_name": settings.LLM_MODEL_NAME,
#         "temperature": settings.LLM_TEMPERATURE,
#         "max_tokens": getattr(settings, "LLM_MAX_TOKENS", None),
#         "timeout": getattr(settings, "LLM_TIMEOUT", 120),
#         "max_retries": getattr(settings, "LLM_MAX_RETRIES", 3),
#         "request_timeout": getattr(settings, "LLM_REQUEST_TIMEOUT", 120),
#         "default_headers": getattr(settings, "LLM_DEFAULT_HEADERS", None),
#         "default_query": getattr(settings, "LLM_DEFAULT_QUERY", None),
#     }
#     llm_params = {k: v for k, v in llm_params.items() if v is not None}
#
#     llm = ChatOpenAI(**llm_params)
#
#     return llm


@functools.lru_cache(maxsize=1)
def get_chat_model():
    settings_kwargs = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "openai_api_base": "https://api.proxyapi.ru/openai/v1",
        "openai_api_key": settings.OPENAI_API_KEY,
        "max_retries": 5,  # Увеличиваем для стабильности через прокси
        "timeout": 90,  # Увеличиваем для тяжелых промптов аналитики
        "streaming": True,  # Включаем потоковую передачу
        "max_tokens": 4096,  # Ограничиваем выход, чтобы не "улетал" в бесконечность
        "model_kwargs": {
            "seed": 42,
            "top_p": 0.0000001,  # Максимальное сужение выбора токенов
            "parallel_tool_calls": False,  # Выключаем параллельность, если важна строгая последовательность шагов
        }
    }

    try:
        model = ChatOpenAI(**settings_kwargs)
        logger.info(f"LLM Model '{settings_kwargs['model']}' initialized successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise


@functools.lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """
    Инициализирует и кэширует EmbeddingModel.
    Использует lru_cache для обеспечения инициализации только один раз.
    """
    logger.info(
        f"Инициализация EmbeddingModel: {settings.EMBEDDING_ENDPOINT}, модель: {settings.EMBEDDING_MODEL_NAME}"
    )

    # Сбор параметров
    embedding_params = {
        "openai_api_base": settings.EMBEDDING_ENDPOINT,
        "openai_api_key": settings.LLM_API_KEY or "placeholder-key",
        "model": settings.EMBEDDING_MODEL_NAME,
        "request_timeout": getattr(settings, "EMBEDDING_REQUEST_TIMEOUT", 120),
        "max_retries": getattr(settings, "EMBEDDING_MAX_RETRIES", 3),
        "default_headers": getattr(settings, "EMBEDDING_DEFAULT_HEADERS", None),
        "default_query": getattr(settings, "EMBEDDING_DEFAULT_QUERY", None),
        "chunk_size": getattr(settings, "EMBEDDING_CHUNK_SIZE", 1000),
    }
    embedding_params = {k: v for k, v in embedding_params.items() if v is not None}

    embedding_model = OpenAIEmbeddings(**embedding_params)

    return embedding_model

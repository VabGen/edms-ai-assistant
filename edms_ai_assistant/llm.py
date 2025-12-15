# edms_ai_assistant/llm.py

import logging
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from edms_ai_assistant.config import settings

logger = logging.getLogger(__name__)


def get_chat_model() -> BaseLanguageModel:
    """Инициализирует ChatModel."""
    logger.info(
        f"Инициализация ChatModel: {settings.LLM_ENDPOINT}, модель: {settings.LLM_MODEL_NAME}"
    )

    llm = ChatOpenAI(
        openai_api_base=settings.LLM_ENDPOINT,
        openai_api_key=settings.LLM_API_KEY or "placeholder-key",
        model_name=settings.LLM_MODEL_NAME,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=getattr(settings, "LLM_MAX_TOKENS", None),
        timeout=getattr(settings, "LLM_TIMEOUT", 120),
        max_retries=getattr(settings, "LLM_MAX_RETRIES", 3),
        request_timeout=getattr(settings, "LLM_REQUEST_TIMEOUT", 120),
        default_headers=getattr(settings, "LLM_DEFAULT_HEADERS", None),
        default_query=getattr(settings, "LLM_DEFAULT_QUERY", None),
    )
    return llm


def get_embedding_model() -> Embeddings:
    """Инициализирует EmbeddingModel."""
    logger.info(
        f"Инициализация EmbeddingModel: {settings.EMBEDDING_ENDPOINT}, модель: {settings.EMBEDDING_MODEL_NAME}"
    )

    embedding_model = OpenAIEmbeddings(
        openai_api_base=settings.EMBEDDING_ENDPOINT,
        openai_api_key=settings.LLM_API_KEY or "placeholder-key",
        model=settings.EMBEDDING_MODEL_NAME,
        request_timeout=getattr(settings, "EMBEDDING_REQUEST_TIMEOUT", 120),
        max_retries=getattr(settings, "EMBEDDING_MAX_RETRIES", 3),
        default_headers=getattr(settings, "EMBEDDING_DEFAULT_HEADERS", None),
        default_query=getattr(settings, "EMBEDDING_DEFAULT_QUERY", None),
        chunk_size=getattr(settings, "EMBEDDING_CHUNK_SIZE", 1000),
    )
    return embedding_model


# class CustomHTTPEmbeddings(Embeddings):
#     """Кастомная модель эмбеддингов."""
#
#     def __init__(self, endpoint_url: str, model_name: str):
#         self.endpoint_url = endpoint_url
#         self.model_name = model_name
#
#     def embed_documents(self, texts: list[str]) -> list[list[float]]:
#         import requests
#
#         try:
#             payload = {"input": texts, "model": self.model_name}
#             response = requests.post(self.endpoint_url + "/embeddings", json=payload)
#             response.raise_for_status()
#             data = response.json()
#             embeddings = [item["embedding"] for item in data.get("data", [])]
#             return embeddings
#         except Exception as e:
#             logger.error(f"Ошибка вызова эндпоинта эмбеддингов: {e}")
#             embedding_size = 384
#             return [[0.0] * embedding_size for _ in texts]
#
#     def embed_query(self, text: str) -> list[float]:
#         return self.embed_documents([text])[0]
#
#
# def get_embedding_model_custom() -> Embeddings:
#     """Инициализирует кастомную EmbeddingModel."""
#     logger.info(
#         f"Инициализация кастомной EmbeddingModel: {settings.EMBEDDING_ENDPOINT}"
#     )
#     return CustomHTTPEmbeddings(
#         endpoint_url=settings.EMBEDDING_ENDPOINT,
#         model_name=settings.EMBEDDING_MODEL_NAME,
#     )


get_embedding_model_to_use = get_embedding_model

# edms_ai_assistant/llm.py

import logging
from typing import List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from edms_ai_assistant.config import settings
import httpx

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
        request_timeout=getattr(settings, "LLM_REQUEST_TIMEOUT", 120),
        max_retries=getattr(settings, "LLM_MAX_RETRIES", 3),
    )
    return llm


def get_embedding_model_openai() -> Embeddings:
    """Инициализирует стандартную OpenAI/LangChain EmbeddingModel."""
    logger.info(
        f"Инициализация EmbeddingModel (OpenAI): {settings.EMBEDDING_ENDPOINT}, модель: {settings.EMBEDDING_MODEL_NAME}"
    )

    embedding_model = OpenAIEmbeddings(
        openai_api_base=settings.EMBEDDING_ENDPOINT,
        openai_api_key=settings.LLM_API_KEY or "placeholder-key",
        model=settings.EMBEDDING_MODEL_NAME,
        request_timeout=getattr(settings, "EMBEDDING_REQUEST_TIMEOUT", 120),
        max_retries=getattr(settings, "EMBEDDING_MAX_RETRIES", 3),
        chunk_size=getattr(settings, "EMBEDDING_CHUNK_SIZE", 1000),
    )
    return embedding_model


class CustomHTTPEmbeddings(Embeddings):
    """
    Кастомная АСИНХРОННАЯ модель эмбеддингов, использующая httpx.
    LangChain V2 требует реализации asinc-методов для асинхронных фреймворков.
    """

    def __init__(self, endpoint_url: str, model_name: str, embedding_size: int = 384):
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.embedding_size = embedding_size  # Установим размер по умолчанию

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Асинхронный вызов для списка документов."""
        async with httpx.AsyncClient() as client:
            try:
                payload = {"input": texts, "model": self.model_name}
                response = await client.post(
                    self.endpoint_url + "/embeddings",
                    json=payload,
                    timeout=getattr(settings, "EMBEDDING_REQUEST_TIMEOUT", 120)
                )
                response.raise_for_status()
                data = response.json()
                embeddings = [item["embedding"] for item in data.get("data", [])]
                return embeddings
            except Exception as e:
                logger.error(f"Ошибка асинхронного вызова эндпоинта эмбеддингов: {e}")
                return [[0.0] * self.embedding_size for _ in texts]

    async def aembed_query(self, text: str) -> List[float]:
        """Асинхронный вызов для одного запроса."""
        return (await self.aembed_documents([text]))[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("Этот метод синхронный и не должен использоваться в асинхронном приложении.")

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError("Этот метод синхронный и не должен использоваться в асинхронном приложении.")


def get_embedding_model_custom() -> Embeddings:
    """Инициализирует кастомную EmbeddingModel."""
    logger.info(
        f"Инициализация кастомной EmbeddingModel: {settings.EMBEDDING_ENDPOINT}"
    )
    embedding_size = getattr(settings, "EMBEDDING_SIZE", 384)
    return CustomHTTPEmbeddings(
        endpoint_url=settings.EMBEDDING_ENDPOINT,
        model_name=settings.EMBEDDING_MODEL_NAME,
        embedding_size=embedding_size
    )


def get_embedding_model_to_use() -> Embeddings:
    """
    Выбирает и инициализирует модель эмбеддингов на основе конфигурации.
    Использует кастомную, если явно указано (например, через EMBEDDING_CUSTOM=True в settings).
    В вашем случае: если не указана опция выбора, оставляем вашу логику.
    """
    return get_embedding_model_openai()

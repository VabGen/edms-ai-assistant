# edms_ai_assistant/tools/document_metadata.py

import json
import logging
from pydantic import BaseModel, Field
from .base import EdmsApiClient

logger = logging.getLogger(__name__)


class GetDocumentByIdArgs(BaseModel):
    document_id: str = Field(description="UUID документа.")


class DocumentMetadataTools:
    """Инструменты для работы с метаданными документов."""

    def __init__(self, api_client: EdmsApiClient):
        self.api_client = api_client

    async def get_by_id(self, document_id: str) -> str:
        """
        Получить все детальные метаданные (автор, статус, суммы, даты и т.д.) о документе по его ID.
        """
        logger.info(f"Вызов API: doc_metadata.get_by_id для {document_id}")
        endpoint = f"api/document/{document_id}"

        data = await self.api_client.get(endpoint)

        if data.get("error"):
            return json.dumps({"error": "Документ не найден или ошибка API.", "details": data}, ensure_ascii=False)

        # Возвращаем полный JSON, чтобы Planner мог извлечь все поля через JSONPath.
        return json.dumps(data, ensure_ascii=False)
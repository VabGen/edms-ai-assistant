# edms_ai_assistant\tools\document.py
import logging
from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.clients.document_client import DocumentClient

logger = logging.getLogger(__name__)


class DocDetailsInput(BaseModel):
    document_id: str = Field(..., description="UUID документа (context_ui_id)")
    token: str = Field(..., description="Токен авторизации пользователя")


@tool("doc_get_details", args_schema=DocDetailsInput)
async def doc_get_details(document_id: str, token: str) -> Dict[str, Any]:
    """
        Получает подробную информацию о документе из EDMS.
        Используй этот инструмент, когда нужно узнать:
        - Название, дату или номер документа.
        - Текущий статус документа.
        - Список вложений или связанных объектов.
        - Кто является автором или исполнителем.
        Возвращает словарь с полными метаданными документа.
        """
    try:
        async with DocumentClient() as client:
            raw_data = await client.get_document_metadata(token, document_id)
            doc = DocumentDto.model_validate(raw_data)

            full_data = doc.model_dump(
                exclude_none=True,
                exclude_unset=True,
                exclude_defaults=True,
                exclude={
                    "organizationId",
                    "journalId",
                    "profileId",
                    "documentVersionId",
                    "processId"
                }
            )

            return {
                "status": "success",
                "document_full_info": full_data
            }

    except Exception as e:
        logger.error(f"Ошибка при обработке DocumentDto: {e}", exc_info=True)
        return {"error": f"Ошибка обработки данных документа: {str(e)}"}



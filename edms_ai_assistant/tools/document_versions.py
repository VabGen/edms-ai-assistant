# edms_ai_assistant/tools/document_versions.py
import logging
from typing import Dict, Any, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.clients.document_client import DocumentClient

logger = logging.getLogger(__name__)


class DocumentVersionsInput(BaseModel):
    """Схема входных данных для получения версий документа."""

    document_id: str = Field(..., description="UUID документа")
    token: str = Field(..., description="Токен авторизации пользователя (JWT)")


@tool("doc_get_versions", args_schema=DocumentVersionsInput)
async def doc_get_versions(document_id: str, token: str) -> Dict[str, Any]:
    """
    Получает список всех версий документа с их метаданными.

    Используй этот инструмент когда:
    - Пользователь спрашивает о версиях документа
    - Нужно сравнить разные версии
    - Нужна информация об истории изменений

    Возвращает:
    - Список версий с номерами, датами создания и авторами
    - Для каждой версии: version_number, created_date, author, document_id
    """
    try:
        async with DocumentClient() as client:
            versions = await client.get_document_versions(token, document_id)

            if not versions:
                return {
                    "status": "success",
                    "message": f"Документ {document_id} имеет только одну версию или версии не найдены.",
                    "versions": [],
                }

            sorted_versions = sorted(versions, key=lambda v: v.get("version", 0))

            versions_info = []
            for v in sorted_versions:
                versions_info.append(
                    {
                        "version_number": v.get("version"),
                        "document_id": v.get("documentId"),
                        "created_date": v.get("createDate"),
                        "parent_document_id": v.get("parentDocumentId"),
                    }
                )

            return {
                "status": "success",
                "total_versions": len(versions_info),
                "versions": versions_info,
                "message": f"Найдено {len(versions_info)} версий документа",
            }

    except Exception as e:
        logger.error(f"[DOC-VERSIONS-TOOL] Error: {e}", exc_info=True)
        return {"status": "error", "message": f"Ошибка получения версий: {str(e)}"}

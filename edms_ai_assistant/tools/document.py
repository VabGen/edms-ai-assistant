# edms_ai_assistant/tools/document.py

from typing import Dict, Any
import json
import logging
from uuid import UUID
from edms_ai_assistant.infrastructure.api_clients.document_client import DocumentClient

logger = logging.getLogger(__name__)


async def _get_document_client() -> DocumentClient:
    """Вспомогательная функция для инициализации stateless клиента."""
    return DocumentClient()


# @tool
async def get_document_tool(document_id: str, service_token: str) -> str:
    """
    Инструмент для получения информации о документе по ID.
    Требует валидный UUID документа.
    """
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        return json.dumps(
            {
                "error": f"Неверный формат ID документа: '{document_id}'. Ожидается UUID."
            },
            ensure_ascii=False,
        )

    logger.info(f"Вызов get_document_tool с document_id: {document_id}")

    client = await _get_document_client()
    async with client:
        try:
            result = await client.get_document(service_token, doc_uuid)
            logger.info(f"Результат get_document_tool: {result}")

            if result is None:
                return json.dumps(
                    {"document": "Документ не найден или ошибка API."},
                    ensure_ascii=False,
                )

            json_result = result.model_dump_json(
                indent=None, by_alias=True, exclude_none=True
            )
            return json_result
        except Exception as e:
            logger.error(f"Ошибка в get_document_tool: {e}")
            return json.dumps(
                {"error": f"Ошибка API при получении документа: {str(e)}"},
                ensure_ascii=False,
            )


# @tool
async def search_documents_tool(filters: Dict[str, Any], service_token: str) -> str:
    """
    Инструмент для поиска документов по фильтрам (например, номеру, дате, статусу).
    Возвращает список JSON-объектов документов.
    """
    logger.info(f"Вызов search_documents_tool с filters: {filters}")

    client = await _get_document_client()
    async with client:
        try:
            result = await client.search_documents(service_token, filters)
            dict_results = [
                doc.model_dump_json(by_alias=True, exclude_none=True) for doc in result
            ]
            return json.dumps(dict_results, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Ошибка в search_documents_tool: {e}")
            return json.dumps(
                {"error": f"Ошибка API при поиске документов: {str(e)}"},
                ensure_ascii=False,
            )

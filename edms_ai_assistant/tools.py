# edms_ai_assistant/tools.py

import json
import random
import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
import logging

from .clients.employee_client import EmployeeClient
from .clients.document_client import DocumentClient
from .clients.attachment_client import AttachmentClient
from .constants import SUMMARY_TYPES

from .models import (
    GetDocumentMetadataArgs,
    GetAttachmentContentArgs,
    SummarizeContentArgs,
    SearchEmployeeArgs,
    GetEmployeeByIdArgs
)

logger = logging.getLogger(__name__)

DOCUMENT_CLIENT = DocumentClient()
ATTACHMENT_CLIENT = AttachmentClient()
EMPLOYEE_CLIENT = EmployeeClient()


# ----------------------------------------------------------------
# ИНСТРУМЕНТЫ (TOOLS) - РЕАЛИЗАЦИЯ С EDMS API
# ----------------------------------------------------------------

@tool(args_schema=GetDocumentMetadataArgs)
async def doc_metadata_get_by_id(document_id: str, **kwargs) -> Dict[str, Any]:
    """
    Получить метаданные документа по его уникальному ID (UUID).
    """
    token = kwargs.get("token")
    if not token:
        return {"error": "Missing token"}

    metadata = await DOCUMENT_CLIENT.get_document_metadata(token=token, document_id=document_id)
    return metadata or {"error": f"Document ID {document_id} not found or no access."}


@tool(args_schema=GetAttachmentContentArgs)
async def doc_attachment_get_content(document_id: str, attachment_id: str, **kwargs) -> Dict[str, Any]:
    """
    Скачивает контент файла вложения по ID документа и ID вложения.
    Возвращает 'content_key' (путь к временному файлу или ID BLOB).
    """
    token = kwargs.get("token")
    if not token:
        return {"error": "Missing token"}

    logger.info(f"TOOL: Скачивание вложения {attachment_id} для документа {document_id}")

    try:
        content_bytes = await ATTACHMENT_CLIENT.download_attachment(
            token=token,
            document_id=document_id,
            attachment_id=attachment_id
        )

        if content_bytes is None:
            return {"error": f"Не удалось скачать контент вложения {attachment_id}."}

        # --- Имитация сохранения файла на диск ---
        # В реальной системе нужно сохранить bytes во временный файл и вернуть путь.
        # Поскольку у нас нет реального кода сохранения, мы имитируем это.

        file_size = len(content_bytes)
        logger.info(f"Скачивание успешно. Размер: {file_size} байт.")

        return {
            "content_key": f"/temp/content-{attachment_id}.bin",
            "size": file_size,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Ошибка в doc_attachment_get_content: {e}")
        return {"error": f"Ошибка связи с EDMS API при загрузке: {type(e).__name__}"}


@tool(args_schema=SummarizeContentArgs)
async def doc_content_summarize(document_id: str, content_key: str, file_name: str,
                                summary_type_id: str = "5", query: Optional[str] = None,
                                metadata_context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Создает структурированную или текстовую сводку по контенту вложенного файла,
    используя указанный тип сводки (Abstract, TL;DR, Structured и т.д.).
    """
    logger.info(f"TOOL: Запрос сводки для ключа: {content_key}, Тип: {summary_type_id}")

    # Получаем описание выбранного типа
    summary_info = SUMMARY_TYPES.get(summary_type_id, SUMMARY_TYPES["5"])
    type_name = summary_info["name"]
    type_description = summary_info["description"]

    # Имитация генерации сводки в зависимости от типа
    if summary_type_id == "6" and query:
        mock_summary = f"**Сводка по запросу ('{query}'):** Контент файла '{file_name}' подтверждает, что..."
    elif summary_type_id == "7":
        mock_summary = (
            "**Структурированная сводка**\n"
            "1. Цель: Регулирование отношений между сторонами.\n"
            "2. Метод/Стороны: Сторона А (Поставщик) и Сторона Б (Заказчик).\n"
            "3. Результаты/Условия: Общая сумма - 10 000 руб. Условия оплаты - 50% предоплата.\n"
            "4. Выводы/Сроки: Срок исполнения - 30 рабочих дней."
        )
    elif summary_type_id == "3":
        mock_summary = f"**TL;DR:** Документ является исходящим письмом о важных условиях договора. ({type_description})"
    else:
        mock_summary = (
            f"**Автоматическая сводка ({type_name}):** Анализ контента '{file_name}' завершен. "
            f"Это {type_description}. Документ создан {metadata_context.get('createDate', 'N/A')}. "
            "Основное содержание касается регулирования условий поставки."
        )

    return {"summary": mock_summary.strip()}


@tool(args_schema=SearchEmployeeArgs)
async def employee_tools_search(**kwargs) -> List[Dict[str, Any]]:
    """
    Используется для поиска списка сотрудников по частичному имени через EDMS API.
    """
    token = kwargs.get("token")
    if not token:
        return {"error": "Missing token"}

    search_query = (
            kwargs.get("search_query")
            or kwargs.get("query")
            or kwargs.get("search_term")
    )

    if not search_query:
        return [{"error": "Необходимо предоставить аргумент 'search_query' для поиска сотрудника."}]

    logger.info(f"TOOL: Поиск сотрудника по запросу: {search_query}")

    try:
        results = await EMPLOYEE_CLIENT.search_employees(
            token=token,
            query=search_query
        )

        return results if isinstance(results, list) else []

    except Exception as e:
        logger.error(f"Ошибка в employee_tools_search: {e}")
        return [{"error": f"Ошибка связи с EDMS API: {type(e).__name__}"}]


@tool(args_schema=GetEmployeeByIdArgs)
async def employee_tools_get_by_id(employee_id: str, **kwargs) -> Dict[str, Any]:
    """
    Используется для получения полных метаданных сотрудника по его ID через EDMS API.
    """
    token = kwargs.get("token")
    if not token:
        return {"error": "Missing token"}

    logger.info(f"TOOL: Запрос данных сотрудника по ID: {employee_id}")

    try:
        data = await EMPLOYEE_CLIENT.get_employee(
            token=token,
            employee_id=employee_id
        )

        if data is None or (isinstance(data, dict) and "error" in data):
            return {"error": f"Сотрудник с ID {employee_id} не найден или произошла ошибка."}

        return data

    except Exception as e:
        logger.error(f"Ошибка в employee_tools_get_by_id: {e}")
        return {"error": f"Ошибка связи с EDMS API: {type(e).__name__}"}


ALL_TOOLS = [
    doc_metadata_get_by_id,
    doc_attachment_get_content,
    doc_content_summarize,
    employee_tools_search,
    employee_tools_get_by_id
]
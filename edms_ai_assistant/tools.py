# edms_ai_assistant/tools.py

import logging
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from langchain.tools import tool
import json

from .clients.employee_client import BaseEmployeeClient, EmployeeClient
from .clients.document_client import BaseDocumentClient, DocumentClient
from .clients.attachment_client import BaseAttachmentClient, AttachmentClient
from .constants import SUMMARY_TYPES
from .models import (
    GetDocumentMetadataArgs,
    GetAttachmentContentArgs,
    SummarizeContentArgs,
    SearchEmployeeArgs,
    GetEmployeeByIdArgs
)
from .utils.file_utils import extract_text_from_bytes

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# ФАБРИКА КЛИЕНТОВ (Для DI)
# ----------------------------------------------------------------

def default_document_client() -> BaseDocumentClient:
    """Возвращает клиента по умолчанию для работы с документами."""
    return DocumentClient()


def default_attachment_client() -> BaseAttachmentClient:
    """Возвращает клиента по умолчанию для работы с вложениями."""
    return AttachmentClient()


def default_employee_client() -> BaseEmployeeClient:
    """Возвращает клиента по умолчанию для работы с сотрудниками."""
    return EmployeeClient()


# ----------------------------------------------------------------
# РЕАЛИЗАЦИЯ ИНСТРУМЕНТОВ (С DI)
# ----------------------------------------------------------------
def build_doc_tools(document_client: BaseDocumentClient, attachment_client: BaseAttachmentClient) -> List[BaseTool]:
    """Строит инструменты, зависящие от документов/вложений."""

    @tool(args_schema=GetDocumentMetadataArgs)
    async def doc_metadata_get_by_id_tool(document_id: str, token: str, **kwargs) -> Dict[str, Any]:
        """
        Получить полные метаданные документа по его уникальному ID (UUID).
        """
        if not token:
            return {"error": "Missing required 'token' argument for authorization."}

        logger.info(f"TOOL: Запрос метаданных для ID: {document_id}")

        try:
            metadata = await document_client.get_document_metadata(token=token, document_id=document_id)

            if not metadata:
                return {"error": f"Document ID {document_id} not found or access denied."}

            return {"metadata": json.dumps(metadata)}

        except Exception as e:
            logger.error(f"Ошибка в doc_metadata_get_by_id: {e}", exc_info=True)
            return {"error": f"EDMS API Error: {type(e).__name__}"}

    @tool(args_schema=GetAttachmentContentArgs)
    async def doc_attachment_get_content_tool(document_id: str, attachment_id: str, token: str, **kwargs) -> Dict[
        str, Any]:
        """
        Скачивает контент файла вложения по ID документа и ID вложения.
        Извлекает текст из байтов файла и возвращает его под ключом 'content'.
        """
        if not token:
            return {"error": "Missing required 'token' argument."}

        logger.info(f"TOOL: Скачивание и извлечение контента вложения {attachment_id} для документа {document_id}")

        try:
            # Вызываем модифицированный метод клиента, который возвращает байты и имя
            result = await attachment_client.download_attachment(
                token=token,
                document_id=document_id,
                attachment_id=attachment_id
            )

            if result is None:
                return {"error": f"Не удалось скачать контент вложения {attachment_id}. Результат пуст."}

            file_bytes, file_name = result

            if not file_bytes:
                return {"error": f"Не удалось скачать контент вложения {attachment_id}. Байты файла не получены."}

            # --- Использование утилиты для извлечения текста ---
            extracted_text = extract_text_from_bytes(file_bytes, file_name)

            if extracted_text is None:
                return {
                    "error": f"Не удалось извлечь текст из файла '{file_name}' (формат не поддерживается или ошибка чтения)."}

            return {
                "content": extracted_text,
                "file_name": file_name,
                "file_size_bytes": len(file_bytes),
                "status": "success",
                "message": f"Текст ({len(extracted_text)} символов) из '{file_name}' успешно извлечен."
            }

        except Exception as e:
            logger.error(f"Ошибка в doc_attachment_get_content: {e}", exc_info=True)
            return {"error": f"Ошибка связи с EDMS API при загрузке: {type(e).__name__}"}

    @tool(args_schema=SummarizeContentArgs)
    async def doc_content_summarize_tool(
            document_id: str,
            content: str,  # Принимаем контент (как мы решили ранее)
            file_name: str,
            summary_type_id: str = "5",  # Используем дефолтный ID "5"
            query: Optional[str] = None,
            metadata_context: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Создает структурированную или текстовую сводку по контенту файла.
        Использует указанный тип сводки (Abstract, TL;DR, Structured и т.д.).
        """
        logger.info(f"TOOL: Запрос сводки для контента ({len(content)} символов), Тип ID: {summary_type_id}")

        if not content:
            return {"error": "Отсутствует контент для создания сводки."}

        # 1. ИСПОЛЬЗУЕМ СЛОВАРЬ И УСТАНАВЛИВАЕМ ДЕФОЛТ
        summary_info = SUMMARY_TYPES.get(summary_type_id, SUMMARY_TYPES["5"])
        type_name = summary_info["name"]
        type_description = summary_info["description"]

        # 2. Мокирование (Используем контент для имитации сводки)

        context_text = f"Дата создания: {metadata_context.get('createDate', 'N/A')}" if metadata_context else ""
        first_100_chars = content[:100].replace('\n', ' ').strip()

        if summary_type_id == "6" and query:
            mock_summary = (
                f"**Сводка по запросу ('{query}'):** Анализ контента '{file_name}' завершен. "
                f"Текст начинается с: '{first_100_chars}...'. "
                f"Ответ на запрос: [На основании извлеченного текста, вот ответ на ваш вопрос]."
            )
        elif summary_type_id == "7":
            mock_summary = (
                f"**Структурированная сводка** (из {file_name}):\n"
                f"1. **Цель:** Регулирование отношений по оказанию услуг.\n"
                f"2. **Стороны:** Исполнитель и Заказчик.\n"
                f"3. **Сумма:** Указана в разделе 3 договора.\n"
            )
        else:
            mock_summary = (
                f"**Автоматическая сводка ({type_name}):** Анализ контента '{file_name}' завершен. "
                f"Это {type_description}. {context_text}. "
                f"Основное содержание: {first_100_chars}..."
            )
        # ----------------------------------------------------------------

        return {
            "summary": mock_summary.strip(),
            "summary_type": type_name
        }

    return [
        doc_metadata_get_by_id_tool,
        doc_attachment_get_content_tool,
        doc_content_summarize_tool,
    ]


def build_employee_tools(employee_client: BaseEmployeeClient) -> List[BaseTool]:
    @tool(args_schema=SearchEmployeeArgs)
    async def employee_tools_search_tool(search_query: str, token: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Используется для поиска списка сотрудников по частичному совпадению (фамилия, имя, должность) через EDMS API.
        """
        if not token:
            return [{"error": "Missing required 'token' argument."}]

        if not search_query:
            return [{"error": "Необходимо предоставить аргумент 'search_query' для поиска сотрудника."}]

        logger.info(f"TOOL: Поиск сотрудника по запросу: {search_query}")

        try:
            results = await employee_client.search_employees(token=token, query=search_query)

            if not results:
                return [{"info": f"Сотрудники по запросу '{search_query}' не найдены."}]

            return {"results": json.dumps(results)}

        except Exception as e:
            logger.error(f"Ошибка в employee_tools_search: {e}", exc_info=True)
            return [{"error": f"Ошибка связи с EDMS API: {type(e).__name__}"}]

    @tool(args_schema=GetEmployeeByIdArgs)
    async def employee_tools_get_by_id_tool(employee_id: str, token: str, **kwargs) -> Dict[str, Any]:
        """
        Используется для получения полных метаданных сотрудника по его ID (UUID) через EDMS API.
        """
        if not token:
            return {"error": "Missing required 'token' argument."}

        logger.info(f"TOOL: Запрос данных сотрудника по ID: {employee_id}")

        try:
            data = await employee_client.get_employee(token=token, employee_id=employee_id)

            if not data or (isinstance(data, dict) and "error" in data):
                return {"error": f"Сотрудник с ID {employee_id} не найден или произошла ошибка."}

            return {"employee_data": json.dumps(data)}

        except Exception as e:
            logger.error(f"Ошибка в employee_tools_get_by_id: {e}", exc_info=True)
            return {"error": f"Ошибка связи с EDMS API: {type(e).__name__}"}

    return [
        employee_tools_search_tool,
        employee_tools_get_by_id_tool,
    ]


# ----------------------------------------------------------------
# ФУНКЦИЯ ДЛЯ СБОРКИ ИНСТРУМЕНТОВ
# ----------------------------------------------------------------

def get_all_tools(
        doc_client: BaseDocumentClient = default_document_client(),
        att_client: BaseAttachmentClient = default_attachment_client(),
        emp_client: BaseEmployeeClient = default_employee_client(),
) -> List[BaseTool]:
    """
    Основная функция для сборки всех инструментов.
    Принимает клиентов EDMS для Dependency Injection.
    """
    doc_tools = build_doc_tools(doc_client, att_client)
    emp_tools = build_employee_tools(emp_client)

    return doc_tools + emp_tools

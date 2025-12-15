# Файл: edms_ai_assistant/tools.py
import json
import random
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
# Относительный импорт моделей аргументов
from .models import (
    GetDocumentMetadataArgs,
    GetAttachmentContentArgs,
    SummarizeContentArgs,
    SearchEmployeeArgs,
    GetEmployeeByIdArgs
)

# --- ИМИТАЦИЯ ДАННЫХ ---
DOC_METADATA_MOCK = {
    "title": "Контракт на поставку ПО №2025/СМ-123",
    "reg_date": "2025-10-25",
    "author_name": "Иванов И.И.",
    "contract_sum": 1250000.50,
    "duration_end": "2026-10-25",
    "attachmentDocument": [
        {"id": "att-b6c8-47d3", "fileName": "ContractBody.pdf", "size": 5242880}
    ]
}

EMPLOYEE_MOCK_LIST = [
    {"id": "emp-101", "name": "Петров А.С.", "position": "Менеджер"},
    {"id": "emp-102", "name": "Петровна Е.М.", "position": "Секретарь"},
    {"id": "emp-103", "name": "Сидоров В.К.", "position": "Разработчик"},
]


# ----------------------------------------------------------------
# ИНСТРУМЕНТЫ (TOOLS) - ЗАГЛУШКИ
# ----------------------------------------------------------------

@tool(args_schema=GetDocumentMetadataArgs)
def doc_metadata_get_by_id(document_id: str) -> Dict[str, Any]:
    """
    Получает все детальные метаданные о документе по его ID (UUID).
    """
    # Имитация: возвращаем метаданные
    return DOC_METADATA_MOCK


@tool(args_schema=GetAttachmentContentArgs)
def doc_attachment_get_content(document_id: str, attachment_id: str) -> Dict[str, Any]:
    """
    Скачивает контент файла вложения по ID документа и ID вложения.
    Возвращает 'content_key' (путь к временному файлу или ID BLOB).
    """
    # Имитация: создаем временный ключ, который будет использован в следующем шаге
    return {"content_key": f"/temp/content-{attachment_id}.pdf", "status": "success"}


@tool(args_schema=SummarizeContentArgs)
def doc_content_summarize(document_id: str, content_key: str, file_name: str,
                          metadata_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Создает структурированную или текстовую сводку по контенту.
    """
    # Имитация: генерируем сводку на основе контекста
    context_str = json.dumps(metadata_context, ensure_ascii=False) if metadata_context else "Контекст не предоставлен."

    summary = f"""
    Сводка документа '{file_name}':
    1. **Общая информация:** Документ успешно проанализирован из {content_key}.
    2. **Ключевой контекст:** Полученные метаданные: {context_str}
    3. **Вывод LLM:** Краткое содержание, с учетом суммы {metadata_context.get('contract_sum', 'N/A')} и даты регистрации {metadata_context.get('reg_date', 'N/A')}.
    """
    return {"summary": summary.strip()}


@tool(args_schema=SearchEmployeeArgs)
def employee_tools_search(**kwargs) -> List[Dict[str, Any]]:  # <-- ИСПОЛЬЗУЕМ **kwargs для гибкости
    """
    Используется для поиска списка сотрудников.
    """
    # КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Проверяем все возможные имена аргументов, которые генерирует LLM
    search_query = (
            kwargs.get("search_query")
            or kwargs.get("query")
            or kwargs.get("search_term")
    )

    if not search_query:
        # Если LLM не предоставил ни одного ожидаемого имени аргумента
        raise ValueError(
            "Необходимо предоставить аргумент 'search_query', 'query', или 'search_term' для поиска сотрудника.")

    # Имитация: если ищем 'Петров', возвращаем несколько результатов (для HiTL).
    if "Петров" in search_query or "петр" in search_query:
        return [EMPLOYEE_MOCK_LIST[0], EMPLOYEE_MOCK_LIST[1]]
    # Имитация: иначе возвращаем один результат (нормальный поток).
    if "Сидоров" in search_query:
        return [EMPLOYEE_MOCK_LIST[2]]

    return []


@tool(args_schema=GetEmployeeByIdArgs)
def employee_tools_get_by_id(employee_id: str) -> Dict[str, Any]:
    """
    Используется для получения полных метаданных сотрудника по его UUID.
    """
    # Имитация: полный профиль
    return {"id": employee_id, "full_name": "Петров Алексей Сергеевич", "email": "a.petrov@org.com", "department": "IT"}


ALL_TOOLS = [
    doc_metadata_get_by_id,
    doc_attachment_get_content,
    doc_content_summarize,
    employee_tools_search,
    employee_tools_get_by_id
]
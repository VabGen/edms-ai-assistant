from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


# ----------------------------------------------------------------------
# 1. МОДЕЛИ ДЛЯ ОРКЕСТРАЦИИ (Orchestration Models)
# ----------------------------------------------------------------------

class ToolCallRequest(BaseModel):
    """
    Модель запроса на вызов конкретного инструмента.
    """
    tool_name: str = Field(..., description="Имя инструмента (например, 'doc_metadata_get_by_id').")
    arguments: Dict[str, Any] = Field(...,
                                      description="Аргументы для инструмента. Для передачи данных используй синтаксис JSONPath: '$STEPS[0].result.attachmentDocument[0].id'.")


class Plan(BaseModel):
    """
    Общий план действий, который генерирует LLM-Планировщик.
    """
    reasoning: str = Field(..., description="Объяснение плана, почему выбраны именно эти шаги.")
    steps: List[ToolCallRequest] = Field(...,
                                         description="Список инструментов, которые нужно вызвать в заданной последовательности.")


# ----------------------------------------------------------------------
# 2. МОДЕЛЬ СОСТОЯНИЯ (State Model) - Ядро LangGraph
# ----------------------------------------------------------------------

class OrchestratorState(TypedDict):
    """
    Словарь состояния, используемый для передачи данных между узлами LangGraph.

    messages: История диалога, включая пользовательские запросы и ответы LLM.
    tools_to_call: Список запланированных вызовов Tools.
    tool_results_history: Результаты выполненных вызовов Tools.
    context_ui_id: ID активной сущности (UUID) из интерфейса.
    user_context: Дополнительный контекст, инжектированный пользователем или системой.
    """
    messages: List[BaseMessage]
    tools_to_call: List[Dict[str, Any]]
    tool_results_history: List[Dict[str, Any]]
    context_ui_id: Optional[str]
    user_context: Optional[Dict[str, Any]] # НОВОЕ ПОЛЕ


# ----------------------------------------------------------------------
# 3. МОДЕЛИ ДЛЯ АРГУМЕНТОВ ИНСТРУМЕНТОВ (ARGS_SCHEMA)
# ----------------------------------------------------------------------

class GetDocumentMetadataArgs(BaseModel):
    document_id: str = Field(..., description="UUID документа, метаданные которого нужно получить.")


class GetAttachmentContentArgs(BaseModel):
    """Аргументы для doc_attachment_get_content: получение контента файла"""
    document_id: str = Field(..., description="UUID родительского документа.")
    attachment_id: str = Field(..., description="UUID вложения, контент которого нужно скачать.")


class SummarizeContentArgs(BaseModel):
    """
    Аргументы для doc_content_summarize: создание сводки по контенту.
    Максимально универсальная модель для любого типа документа.
    """
    document_id: str = Field(..., description="UUID родительского документа (для контекста).")
    content_key: str = Field(...,
                             description="Ключ для доступа к контенту файла в хранилище (например, путь к временному файлу или ID BLOB).")
    file_name: str = Field(..., description="Имя файла для сводки.")

    metadata_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Словарь с ключевыми метаданными документа (например, сумма, даты, автор), которые помогают LLM в сводке."
    )


class SearchEmployeeArgs(BaseModel):
    search_query: str = Field(
        ...,
        description="КЛЮЧЕВОЕ ПОЛЕ: 'search_query'. Это частичное совпадение (фамилия, имя или должность) для поиска сотрудника."
    )


class GetEmployeeByIdArgs(BaseModel):
    employee_id: str = Field(..., description="UUID сотрудника для получения полных метаданных.")
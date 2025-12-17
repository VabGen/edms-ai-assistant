# edms_ai_assistant/models.py

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


# ----------------------------------------------------------------------
# 1. МОДЕЛИ API (FastAPI Request/Response Models)
# ----------------------------------------------------------------------

class UserContext(BaseModel):
    """Модель для дополнительного контекста пользователя."""
    role: Optional[str] = Field(None, description="Роль пользователя в системе (например, 'Менеджер' или 'Бухгалтер').")
    permissions: List[str] = Field(default_factory=list, description="Список прав доступа пользователя в EDMS.")


class UserInput(BaseModel):
    """Входные данные от пользователя через API."""
    message: str = Field(..., description="Основное текстовое сообщение от пользователя.")
    user_token: str = Field(..., description="JWT токен пользователя для аутентификации и контекста LangGraph.")
    context_ui_id: Optional[str] = Field(
        None, description="UUID документа или объекта в EDMS, с которым пользователь взаимодействует."
    )
    context: Optional[UserContext] = Field(
        None, description="Дополнительный структурированный контекст о пользователе."
    )
    file_path: Optional[str] = Field(
        None, description="Временный путь к загруженному файлу, полученный из /upload-file."
    )


class FileUploadResponse(BaseModel):
    """Ответ после успешной загрузки файла."""
    file_path: str = Field(
        ..., description="Временный путь на сервере. Клиент должен передать его в поле `file_path` в /chat."
    )
    file_name: str = Field(..., description="Имя загруженного файла.")


class AssistantResponse(BaseModel):
    """Ответ, отправляемый пользователю."""
    response: str = Field(..., description="Финальный текстовый ответ ассистента.")


# ----------------------------------------------------------------------
# 2. МОДЕЛИ ДЛЯ ОРКЕСТРАЦИИ (Orchestration Models)
# ----------------------------------------------------------------------

class ToolCallRequest(BaseModel):
    """
    Модель запроса на вызов конкретного инструмента, генерируемая планировщиком (LLM).
    """
    tool_name: str = Field(..., description="Имя инструмента (например, 'doc_metadata_get_by_id').")
    arguments: Dict[str, Any] = Field(
        ...,
        description=(
            "Аргументы для инструмента. Для передачи данных используй синтаксис JSONPath. "
            "Пример: '$STEPS[0].result.attachmentDocument[0].id'."
        ),
    )


class Plan(BaseModel):
    """
    Общий план действий, который генерирует LLM-Планировщик.
    """
    reasoning: str = Field(..., description="Объяснение плана, почему выбраны именно эти шаги.")
    steps: List[ToolCallRequest] = Field(
        ..., description="Список инструментов, которые нужно вызвать в заданной последовательности."
    )


# ----------------------------------------------------------------------
# 3. МОДЕЛЬ СОСТОЯНИЯ (LangGraph State Model)
# ----------------------------------------------------------------------
class OrchestratorState(TypedDict):
    """
    Словарь состояния, используемый для передачи данных между узлами LangGraph.
    Все ключи должны быть здесь.
    """
    messages: List[BaseMessage]
    tools_to_call: List[Dict[str, Any]]
    tool_results_history: List[Dict[str, Any]]
    context_ui_id: Optional[str]
    user_context: Optional[Dict[str, Any]]
    user_token: Optional[str]
    file_path: Optional[str]
    required_file_name: Optional[str]

# ----------------------------------------------------------------------
# 4. МОДЕЛИ ДЛЯ АРГУМЕНТОВ ИНСТРУМЕНТОВ (ARGS_SCHEMA)
# ----------------------------------------------------------------------
class GetDocumentMetadataArgs(BaseModel):
    document_id: str = Field(..., description="UUID документа, метаданные которого нужно получить.")
    token: str = Field(..., description="JWT токен пользователя для аутентификации в EDMS.")

class GetEmployeeByIdArgs(BaseModel):
    employee_id: str = Field(..., description="UUID сотрудника для получения полных метаданных.")
    token: str = Field(..., description="JWT токен пользователя для аутентификации в EDMS.")


class GetAttachmentContentArgs(BaseModel):
    """Аргументы для doc_attachment_get_content: получение контента файла"""
    document_id: str = Field(..., description="UUID родительского документа.")
    attachment_id: str = Field(..., description="UUID вложения, контент которого нужно скачать.")
    token: str = Field(..., description="JWT токен пользователя для аутентификации в EDMS.")


class SummarizeContentArgs(BaseModel):
    document_id: str = Field(..., description="ID документа (UUID) из контекста.")
    content: str = Field(..., description="Текстовый контент файла, извлеченный с предыдущего шага.")
    file_name: str = Field(..., description="Имя файла, извлеченное с предыдущего шага.")
    summary_type_id: str = Field(
        "5",
        description=(
            "ID типа требуемой сводки. "
            "Доступные ID: 1 (Abstract), 2 (Extractive), 3 (TL;DR), 4 (Single-sentence), "
            "5 (Multi-sentence - по умолчанию), 6 (Query-focused), 7 (Structured)."
        ),
    )
    query: Optional[str] = Field(
        None,
        description="Обязателен, если summary_type_id='6' (Query-focused). Содержит вопрос пользователя.",
    )


class SearchEmployeeArgs(BaseModel):
    search_query: str = Field(
        ...,
        description="КЛЮЧЕВОЕ ПОЛЕ: 'search_query'. Частичное совпадение (фамилия, имя или должность) для поиска сотрудника.",
    )
    token: str = Field(..., description="JWT токен пользователя для аутентификации в EDMS.")


class GetEmployeeByIdArgs(BaseModel):
    employee_id: str = Field(..., description="UUID сотрудника для получения полных метаданных.")

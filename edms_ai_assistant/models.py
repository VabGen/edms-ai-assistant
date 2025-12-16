from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class UserContext(BaseModel):
    """Модель для дополнительного контекста пользователя."""
    role: Optional[str] = Field(None, description="Роль пользователя в системе.")
    permissions: List[str] = Field(default_factory=list, description="Права доступа пользователя.")


class UserInput(BaseModel):
    """Входные данные от пользователя через API."""
    message: str = Field(..., description="Основное текстовое сообщение от пользователя.")
    user_token: str = Field(..., description="JWT токен пользователя для аутентификации и доступа к EDMS.")
    context_ui_id: Optional[str] = Field(None, description="ID документа или объекта, если запрос контекстный.")
    context: Optional[UserContext] = Field(None, description="Дополнительный контекст о пользователе.")
    file_path: Optional[str] = Field(None,
                                     description="Временный путь к загруженному файлу, полученный из /upload-file.")


class FileUploadResponse(BaseModel):
    """Ответ после успешной загрузки файла."""
    file_path: str = Field(..., description="Временный путь, который нужно передать в /chat.")
    file_name: str = Field(..., description="Имя загруженного файла.")


class AssistantResponse(BaseModel):
    """Ответ, отправляемый пользователю."""
    response: str = Field(..., description="Финальный ответ ассистента.")


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
# 2. МОДЕЛЬ СОСТОЯНИЯ (State Model) - Ядро LangGraph (ИСПРАВЛЕНО)
# ----------------------------------------------------------------------
class OrchestratorState(TypedDict):
    """
    Словарь состояния, используемый для передачи данных между узлами LangGraph.
    ...
    """
    messages: List[BaseMessage]
    tools_to_call: List[Dict[str, Any]]
    tool_results_history: List[Dict[str, Any]]
    context_ui_id: Optional[str]
    user_context: Optional[Dict[str, Any]]
    user_token: Optional[str]
    file_path: Optional[str]


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
    Аргументы для создания сводки содержимого документа.
    """
    document_id: str = Field(..., description="ID документа СЭД (например, 'df0b5549-a82a-11f0-b3ec-820d629f4f0e').")
    content_key: str = Field(..., description="Внутренний ключ к файлу, полученный после скачивания вложения ($.STEPS[1].result.content_key).")
    file_name: str = Field(..., description="Имя файла вложения для контекста ($.STEPS[0].result.attachmentDocument[0].name).")
    summary_type_id: str = Field(
        "5",
        description=(
            "ID типа требуемой сводки. "
            "Доступные ID: 1 (Abstract), 2 (Extractive), 3 (TL;DR), 4 (Single-sentence), "
            "5 (Multi-sentence - по умолчанию), 6 (Query-focused), 7 (Structured)."
        )
    )
    query: Optional[str] = Field(
        None,
        description="Обязателен, если summary_type_id='6' (Query-focused). Содержит вопрос пользователя, на который нужно ответить на основе контента."
    )


class SearchEmployeeArgs(BaseModel):
    search_query: str = Field(
        ...,
        description="КЛЮЧЕВОЕ ПОЛЕ: 'search_query'. Это частичное совпадение (фамилия, имя или должность) для поиска сотрудника."
    )


class GetEmployeeByIdArgs(BaseModel):
    employee_id: str = Field(..., description="UUID сотрудника для получения полных метаданных.")

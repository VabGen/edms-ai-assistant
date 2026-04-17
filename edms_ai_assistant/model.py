# edms_ai_assistant/model.py
"""
EDMS AI Assistant — Public data contracts (Pydantic v2).
"""

from __future__ import annotations

import re
from datetime import datetime
from dataclasses import field, dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import TypedDict

from edms_ai_assistant.utils.regex_utils import UUID_RE


# ─────────────────────────────────────────────────────────────
# LangGraph state
# ─────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """
    Состояние LangGraph графа.

    Единственное каноническое определение — используется
    как в _build_graph, так и везде, где нужен тип состояния.
    Reducer add_messages обеспечивает корректное слияние.
    """

    messages: Annotated[list[BaseMessage], add_messages]


# ─────────────────────────────────────────────────────────────
# Input models
# ─────────────────────────────────────────────────────────────


class UserContext(BaseModel):
    """Профиль пользователя из EDMS."""

    firstName: str | None = Field(None, max_length=100)
    lastName: str | None = Field(None, max_length=100)
    middleName: str | None = Field(None, max_length=100)
    role: str | None = Field(None, max_length=100)
    post: str | None = Field(None, max_length=200)


class UserInput(BaseModel):
    """
    Входное сообщение от клиента к /chat эндпоинту.

    Валидация здесь минимальна — детальная валидация
    делегируется AgentRequest в agent.py (Service Layer).
    """

    message: str = Field(..., min_length=1, max_length=8000)
    user_token: str = Field(..., min_length=10)
    context_ui_id: str | None = Field(
        None,
        description="UUID активного документа в UI EDMS",
    )
    context: UserContext | None = None
    file_path: str | None = Field(
        None,
        max_length=500,
        description="UUID вложения EDMS или путь к локальному файлу",
    )
    file_name: str | None = None
    human_choice: str | None = Field(
        None,
        max_length=200,
        description=(
            "Выбор пользователя: тип суммаризации (extractive/abstractive/thesis) "
            "или UUID сотрудников через запятую для disambiguation"
        ),
    )
    thread_id: str | None = Field(None, max_length=255)
    preferred_summary_format: str | None = None

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        """Removes surrounding whitespace from the message."""
        return v.strip()


# ─────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────

ResponseStatus = Literal["success", "error", "requires_action", "processing"]


class AssistantResponse(BaseModel):
    """
    Стандартизированный ответ агента клиенту.

    Fields:
        status: Машиночитаемый статус выполнения.
        response: Текстовый ответ для отображения пользователю.
        action_type: Тип требуемого действия (если requires_action).
        message: Системное или пользовательское сообщение.
        thread_id: ID треда для продолжения диалога.
        requires_reload: True если EDMS нужно перезагрузить страницу
                         (после мутирующих операций: ознакомление, поручение и т.д.)
    """

    status: ResponseStatus = "success"
    response: str | None = None
    action_type: str | None = None
    message: str | None = None
    thread_id: str | None = None
    requires_reload: bool = Field(
        default=False,
        description=(
            "Сигнал фронтенду перезагрузить страницу EDMS "
            "после успешного выполнения мутирующих операций"
        ),
    )
    navigate_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Auxiliary request/response models
# ─────────────────────────────────────────────────────────────


class FileUploadResponse(BaseModel):
    """Ответ на загрузку файла."""

    file_path: str
    file_name: str


class NewChatRequest(BaseModel):
    """Запрос на создание нового треда диалога."""

    user_token: str = Field(..., min_length=10)


class AgentStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    REQUIRES_ACTION = "requires_action"
    PROCESSING = "processing"
    REQUIRES_CONFIRMATION = "requires_confirmation"  # <--- ДОБАВЛЕНО


class AgentResponse(BaseModel):
    status: AgentStatus
    content: str | None = None
    message: str | None = None
    action_type: ActionType | None = None
    requires_reload: bool = False
    navigate_url: str | None = None
    requires_confirmation: bool = False
    confirmation_prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionType(str, Enum):
    SUMMARIZE_SELECTION = "summarize_selection"
    DISAMBIGUATION = "requires_disambiguation"
    CONFIRMATION = "requires_confirmation"


class AgentRequest(BaseModel):
    message: str = Field(default="", max_length=8000)
    user_token: str = Field(..., min_length=10)
    context_ui_id: str | None = Field(
        None,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$|^$",
    )
    thread_id: str | None = Field(None, max_length=255)
    user_context: dict[str, Any] = Field(default_factory=dict)
    file_path: str | None = Field(None, max_length=500)
    file_name: str | None = Field(None, max_length=260)
    human_choice: str | None = Field(None, max_length=200)

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def validate_message_or_choice(self) -> AgentRequest:
        has_message = bool(self.message and self.message.strip())
        has_choice = bool(self.human_choice and self.human_choice.strip())
        if not has_message and not has_choice:
            raise ValueError("Either message or human_choice must be provided")
        if not has_message and has_choice:
            self.message = self.human_choice
        return self

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str | None) -> str | None:
        if not v:
            return None
        stripped = v.strip()
        if UUID_RE.match(stripped):
            return stripped
        if len(stripped) < 500:
            if stripped.startswith("/") or re.match(r"^[A-Za-z]:\\", stripped) or re.match(r"^[^/\\]+[\\/]", stripped):
                return stripped
        raise ValueError(f"Invalid file_path format: {v!r}")


@dataclass
class ContextParams:
    """
    Immutable execution context passed through the entire agent lifecycle.
    """
    user_token: str
    document_id: str | None = None
    file_path: str | None = None
    thread_id: str = "default"
    user_name: str = "пользователь"
    user_first_name: str | None = None
    user_last_name: str | None = None
    user_full_name: str | None = None
    user_id: str | None = None
    current_date: str = field(default_factory=lambda: datetime.now().strftime("%d.%m.%Y"))
    current_year: str = field(default_factory=lambda: str(datetime.now().year))
    uploaded_file_name: str | None = None
    user_context: dict = field(default_factory=dict)
    intent: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.user_token or not isinstance(self.user_token, str):
            raise ValueError("user_token must be a non-empty string")
        if self.file_path and not self.uploaded_file_name:
            fp = str(self.file_path).strip()
            if not UUID_RE.match(fp):
                self.uploaded_file_name = Path(fp).name
        if not self.user_full_name:
            parts = [p for p in (self.user_last_name, self.user_first_name) if p]
            if parts:
                self.user_full_name = " ".join(parts)


# ─────────────────────────────────────────────────────────────
# Shared Utility Functions (to avoid circular imports)
# ─────────────────────────────────────────────────────────────

_MUTATION_SUCCESS_PHRASES: tuple[str, ...] = (
    "успешно добавлен", "успешно создан", "список ознакомления",
    "поручение создано", "поручение успешно", "обращение заполнено",
    "карточка заполнена", "добавлено в список", "ознакомление создано",
    "задача создана", "заголовок обновлен", "заголовок изменен",
    "адрес заявителя обновлен", "телефон в карточке обновлен",
    "изменение выполнено успешно", "операция выполнена успешно",
)

def _is_mutation_response(content: str | None) -> bool:
    """Returns True if the response describes a successful mutating EDMS operation."""
    if not content:
        return False
    lower = content.lower()
    return any(phrase in lower for phrase in _MUTATION_SUCCESS_PHRASES)

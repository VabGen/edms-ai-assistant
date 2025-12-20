# edms_ai_assistant/models.py
import operator
from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class UserContext(BaseModel):
    """Информация о пользователе для кастомизации ответов."""
    role: Optional[str] = Field(None, description="Роль (Менеджер, Админ и т.д.)")
    permissions: List[str] = Field(default_factory=list, description="Список прав доступа")


class UserInput(BaseModel):
    """То, что прилетает с фронта."""
    message: str = Field(..., description="Текст запроса")
    user_token: str = Field(..., description="JWT токен из фронта")
    context_ui_id: Optional[str] = Field(
        None,
        description="ID объекта в EDMS (документ, сотрудник, задача, и т.д.)"
    )
    context: Optional[UserContext] = None
    file_path: Optional[str] = None


class FileUploadResponse(BaseModel):
    """Ответ после успешной загрузки файла."""
    file_path: str = Field(..., description="Путь к сохраненному файлу")
    file_name: str = Field(..., description="Оригинальное имя файла")


class AssistantResponse(BaseModel):
    """То, что фронт ожидает получить обратно."""
    response: str
    thread_id: str


class AgentState(TypedDict):
    """Как данные хранятся внутри LangGraph."""
    messages: Annotated[List[BaseMessage], operator.add]
    user_token: str
    context_ui_id: Optional[str]
    user_context: Optional[Dict[str, Any]]
    file_path: Optional[str]

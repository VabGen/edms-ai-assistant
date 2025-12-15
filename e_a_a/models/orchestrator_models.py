# edms_ai_assistant/models/orchestrator_models.py
from pydantic import BaseModel
from typing import Optional


class UserContext(BaseModel):
    """
    Контекст пользователя, связанный с его текущим положением в системе.
    """
    current_page: str = "general"
    document_id: Optional[str] = None


class UserInput(BaseModel):
    """
    Основная модель данных, получаемая от клиента через поле 'user_request'.
    """
    message: str
    user_token: str
    context: Optional[UserContext] = None


class AssistantResponse(BaseModel):
    """
    Модель ответа, отправляемого клиенту.
    """
    response: str
    documents_created: list = []
    tasks_created: list = []
    attachments_processed: list = []

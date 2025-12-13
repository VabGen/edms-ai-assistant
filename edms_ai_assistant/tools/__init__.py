# edms_ai_assistant/tools/__init__.py

from .document_metadata import DocumentMetadataTools
from .document_attachment import DocumentAttachmentTools
from .employee_tools import EmployeeTools
from edms_ai_assistant.tools.base import EdmsApiClient
from edms_ai_assistant.llm import get_chat_model

def get_available_tools(auth_token: str, llm_summarizer) -> dict:
    """
    Фабрика для инициализации и регистрации всех доступных инструментов.
    Возвращает словарь экземпляров классов инструментов.
    """
    # 1. Инициализация общего API-клиента
    api_client = EdmsApiClient(token=auth_token)

    # 2. Регистрация инструментов
    return {
        "doc_metadata": DocumentMetadataTools(api_client=api_client),
        "doc_attachment": DocumentAttachmentTools(api_client=api_client, llm_summarizer=llm_summarizer),
        "employee_tools": EmployeeTools(api_client=api_client),
    }
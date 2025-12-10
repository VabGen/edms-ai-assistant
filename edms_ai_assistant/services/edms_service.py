# edms_ai_assistant/services/edms_service.py

import logging
from typing import List, Optional
from uuid import UUID
from edms_ai_assistant.infrastructure.api_clients.employee_client import EmployeeClient
from edms_ai_assistant.infrastructure.api_clients.document_client import DocumentClient
from edms_ai_assistant.infrastructure.api_clients.attachment_client import AttachmentClient
from edms_ai_assistant.generated.resources_openapi import EmployeeDto, DocumentDto, UserInfoDto

logger = logging.getLogger(__name__)


class EdmsService:
    """
    Сервисный слой для взаимодействия с EDMS API.
    Использует stateless (token-free) клиенты.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url

    def _get_employee_client(self) -> EmployeeClient:
        return EmployeeClient(base_url=self.base_url)

    def _get_document_client(self) -> DocumentClient:
        # Создаем новый stateless клиент
        return DocumentClient(base_url=self.base_url)

    def _get_attachment_client(self) -> AttachmentClient:
        # Создаем новый stateless клиент
        return AttachmentClient(base_url=self.base_url)

    # --- Методы для работы с сотрудниками ---
    async def find_employees_by_last_names(self, token: str, last_names: List[str]) -> List[EmployeeDto]:
        """
        Находит сотрудников по списку фамилий.
        API-клиент должен быть обновлен для приема 'token'.
        """
        found_employees = []
        client = self._get_employee_client()
        for name in last_names:
            # Предполагаем, что client.search_employees обновлен
            results = await client.search_employees(token=token, query=name)
            for emp_data in results:
                if emp_data.get('lastName', '').lower() == name.lower():
                    found_employees.append(EmployeeDto.model_validate(emp_data))
        return found_employees

    async def get_current_user(self, token: str) -> Optional[UserInfoDto]:
        """Получает информацию о текущем пользователе."""
        # Предполагаем, что get_current_user в клиенте также принимает token
        client = self._get_employee_client()
        logger.warning("Метод get_current_user требует уточнения API и клиента.")
        return None

    # --- Методы для работы с документами ---
    async def get_document_by_id(self, token: str, document_id: str) -> Optional[DocumentDto]:
        """
        Получает документ по ID.
        """
        client = self._get_document_client()
        try:
            doc_uuid = UUID(document_id)
        except ValueError:
            logger.error(f"Неверный формат ID документа: {document_id}")
            return None

        result = await client.get_document(token, doc_uuid)
        return result

    # --- Методы для работы с вложениями ---
    async def download_attachment(self, token: str, document_id: str, attachment_id: str) -> Optional[bytes]:
        """
        Скачивает вложение как байты.
        """
        client = self._get_attachment_client()
        try:
            doc_uuid = UUID(document_id)
            att_uuid = UUID(attachment_id)
        except ValueError:
            logger.error(f"Неверный формат ID документа или вложения: {document_id}, {attachment_id}")
            return None

        return await client.download_attachment(token, doc_uuid, att_uuid)

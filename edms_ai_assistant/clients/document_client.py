# Файл: edms_ai_assistant/clients/document_client.py

from typing import Optional, Dict, Any, List
from .base_client import EdmsBaseClient

class DocumentClient(EdmsBaseClient):
    """Асинхронный клиент для работы с EDMS Document API."""

    async def get_document_metadata(self, token: str, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Получить метаданные документа по ID (GET api/document/{id}).
        """
        data = await self._make_request(
            "GET",
            f"api/document/{document_id}",
            token=token
        )
        return data

    async def get_contract_responsible(self, token: str, document_id: str) -> List[Dict[str, Any]]:
        """
        Получить ответственных по договору (GET api/document/{documentId}/responsible).
        """
        result = await self._make_request(
            "GET",
            f"api/document/{document_id}/responsible",
            token=token
        )
        # Ожидаем List<ContractResponsibleDto>
        return result if isinstance(result, list) else []

    async def search_documents(
        self, token: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Упрощенный поиск документов.
        NOTE: Оставляем как общий GET-метод, так как точный "search" POST-эндпоинт не предоставлен.
        """
        params = filters or {}
        result = await self._make_request(
            "GET",
            "api/document",
            token=token,
            params=params
        )
        return result if isinstance(result, list) else []
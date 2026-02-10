# edms_ai_assistant/clients/document_client.py
from typing import Optional, Dict, Any, List
from abc import abstractmethod
from .base_client import EdmsHttpClient, EdmsBaseClient


class EdmsDocumentClient(EdmsBaseClient):
    """Абстрактный класс для работы с документами."""

    @abstractmethod
    async def get_document_metadata(
        self, token: str, document_id: str
    ) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def get_contract_responsible(
        self, token: str, document_id: str
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def search_documents(
        self, token: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class DocumentClient(EdmsDocumentClient, EdmsHttpClient):
    """Асинхронный клиент для работы с EDMS Document API."""

    async def get_document_metadata(
        self, token: str, document_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Получить метаданные документа по ID (GET api/document/{id}).
        """
        data = await self._make_request(
            "GET", f"api/document/{document_id}", token=token
        )
        return data

    async def get_contract_responsible(
        self, token: str, document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Получить ответственных по договору (GET api/document/{documentId}/responsible).
        """
        result = await self._make_request(
            "GET", f"api/document/{document_id}/responsible", token=token
        )
        return result if isinstance(result, list) else []

    async def search_documents(
        self, token: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Упрощенный поиск документов.
        """
        params = filters or {}
        result = await self._make_request(
            "GET", "api/document", token=token, params=params
        )
        return result if isinstance(result, list) else []

    async def get_document_versions(
        self, token: str, document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Получить все версии документа (GET api/document/{id}/version).
        """
        result = await self._make_request(
            "GET", f"api/document/{document_id}/version", token=token
        )
        return result if isinstance(result, list) else []

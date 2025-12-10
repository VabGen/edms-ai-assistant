# edms_ai_assistant\infrastructure\api_clients\document_client.py
"""
EDMS Document Client — асинхронный клиент для взаимодействия с EDMS API (документы).
КЛИЕНТ ЯВЛЯЕТСЯ STATELESS: не хранит service_token, принимает его в каждом запросе.
"""
import httpx
import logging
from typing import Optional, Dict, Any, List, Union
from uuid import UUID
from edms_ai_assistant.config import settings
from edms_ai_assistant.utils.retry_utils import async_retry

try:
    from edms_ai_assistant.generated.resources_openapi import (DocumentDto, DocCategory, RequiredFieldEnum,
                                                               UserInfoDto, Status2, DocumentUserColorDto,
                                                               TaskDto, DeliveryMethodDto, DocumentVersionDto,
                                                               DocumentProcessDto, AttachmentDocumentDto,
                                                               DocumentRecipientDtoModel, ControlDto,
                                                               MiniUserInfoDto, IntroductionDto, DocumentAppealDto,
                                                               DocumentResponsibleExecutorDto, DocumentUserPropsDto,
                                                               AccessGriefDto, CurrencyDto, DocumentInventoryDataDto,
                                                               DocumentFormDefinitionDto, AdditionalDocumentDto
                                                               )
except ImportError:
    class DocumentDto:
        @classmethod
        def model_validate(cls, data): return cls(**data)

        def model_dump(self): return self.__dict__

        def __init__(self, **kwargs): self.__dict__.update(kwargs)

from edms_ai_assistant.utils.api_utils import (
    handle_api_error,
    prepare_auth_headers,
)

logger = logging.getLogger(__name__)


class DocumentClient:
    """
    Асинхронный клиент для работы с EDMS Document API.
    Не хранит service_token, что исключает ошибки авторизации в асинхронной среде.
    """

    def __init__(
            self,
            base_url: Optional[str] = None,
            timeout: Optional[int] = None,
    ):
        resolved_base_url = base_url or str(settings.chancellor_next_base_url)
        self.base_url = resolved_base_url.rstrip("/")
        self.timeout = timeout or settings.edms_timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Закрывает HTTP-клиент."""
        await self.client.aclose()

    def _get_headers(self, token: str) -> Dict[str, str]:
        """Возвращает заголовки с авторизацией, используя переданный токен."""
        return prepare_auth_headers(token)

    @async_retry(
        max_attempts=3,
        delay=1.0,
        backoff=2.0,
        exceptions=(httpx.RequestError, httpx.HTTPStatusError),
    )
    async def _make_request(
            self,
            method: str,
            endpoint: str,
            token: str,
            **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Выполняет HTTP-запрос, обрабатывает ошибки и возвращает декодированный ответ.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers(token)
        kwargs['headers'] = headers

        try:
            response = await self.client.request(method, url, **kwargs)
            await handle_api_error(response, f"{method} {url}")

            if response.status_code == 204 or not response.content:
                return {}

            return response.json()

        except httpx.HTTPStatusError:
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error for {method} {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for {method} {url}: {e}")
            raise

    # === Документы (возвращают DocumentDto или None) ===
    async def get_document(self, token: str, document_id: UUID) -> Optional[DocumentDto]:
        """Получить документ по ID. Возвращает типизированную модель."""
        data = await self._make_request("GET", f"api/document/{document_id}", token=token)

        if not data:
            return None

        try:
            return DocumentDto.model_validate(data)
        except Exception as e:
            logger.error(f"Ошибка валидации DocumentDto для ID {document_id}: {e}")
            return None

    async def create_document(self, token: str, profile_id: UUID) -> Dict[str, Any]:
        """Создать новый документ. Возвращает JSON."""
        data = {"id": str(profile_id)}
        return await self._make_request("POST", "api/document", token=token, json=data)

    async def search_documents(
            self, token: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Поиск документов с фильтрацией. Возвращает список JSON."""
        params = filters or {}
        result = await self._make_request("GET", "api/document", token=token, params=params)
        return result if isinstance(result, list) else []

    # === Версии, История, Статусы и т.д. (возвращают Dict или List) ===
    async def create_document_version(
            self, token: str, document_id: UUID, body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Создать новую версию документа. Возвращает JSON."""
        return await self._make_request(
            "POST", f"api/document/{document_id}/version", token=token, json=body
        )

    async def get_all_versions(
            self, token: str, document_id: UUID
    ) -> List[Dict[str, Any]]:
        """Получить все версии документа. Возвращает список JSON."""
        result = await self._make_request("GET", f"api/document/{document_id}/version", token=token)
        return result if isinstance(result, list) else []

    async def get_document_history(self, token: str, document_id: UUID) -> Dict[str, Any]:
        """Получить историю документа. Возвращает JSON."""
        return await self._make_request("GET", f"api/document/{document_id}/history/v2", token=token)

    async def get_document_recipients(
            self, token: str, document_id: UUID
    ) -> List[Dict[str, Any]]:
        """Получить список адресатов документа. Возвращает список JSON."""
        result = await self._make_request("GET", f"api/document/{document_id}/recipient", token=token)
        return result if isinstance(result, list) else []

    async def get_correspondents(
            self, token: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Получить список контрагентов. Возвращает список JSON."""
        params = filters or {}
        result = await self._make_request("GET", "api/document/recipient", token=token, params=params)
        return result if isinstance(result, list) else []

    async def get_document_statuses(
            self, token: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Получить статусы документов. Возвращает список JSON."""
        params = filters or {}
        result = await self._make_request("GET", "api/document/status", token=token, params=params)
        return result if isinstance(result, list) else []

    async def get_status_groups(
            self, token: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Получить группировку по статусам. Возвращает список JSON."""
        params = filters or {}
        result = await self._make_request("GET", "api/document/status-group", token=token, params=params)
        return result if isinstance(result, list) else []

    # === Операции ===
    async def execute_document_operations(
            self, token: str, document_id: UUID, operations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Выполнить операции над документом (согласование, подписание и т.д.).
        Возвращает JSON-объект.
        """
        result = await self._make_request(
            "POST", f"api/document/{document_id}/execute", token=token, json=operations
        )
        if not result:
            return {"status": "success", "message": "Operations executed successfully"}
        return result

    # === Автор, Свойства, Ответственные ===
    async def change_document_author(
            self, token: str, document_id: UUID, new_author_id: UUID
    ) -> Dict[str, Any]:
        """Изменить автора документа. Возвращает JSON."""
        data = {"id": str(new_author_id)}
        return await self._make_request(
            "PUT", f"api/document/{document_id}/change-document-author", token=token, json=data
        )

    async def get_document_properties(
            self, token: str, document_id: UUID
    ) -> Dict[str, Any]:
        """Получить свойства документа. Возвращает JSON."""
        return await self._make_request("GET", f"api/document/{document_id}/properties", token=token)

    async def get_contract_responsibles(
            self, token: str, document_id: UUID
    ) -> List[Dict[str, Any]]:
        """Получить ответственных по договору. Возвращает список JSON."""
        result = await self._make_request(
            "GET", f"api/document/{document_id}/responsible", token=token
        )
        return result if isinstance(result, list) else []

    async def get_contract_version_info(
            self, token: str, document_id: UUID
    ) -> List[Dict[str, Any]]:
        """Получить информацию о версиях договора. Возвращает список JSON."""
        result = await self._make_request(
            "GET", f"api/document/{document_id}/contract-version-info", token=token
        )
        return result if isinstance(result, list) else []

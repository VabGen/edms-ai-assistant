# edms_ai_assistant/core/deps.py
from __future__ import annotations

import redis.asyncio as aioredis
from pydantic import BaseModel, ConfigDict
from typing import Any, Optional

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.clients.transport import IAsyncTransport
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.clients.document_creator_client import DocumentCreatorClient
from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.clients.department_client import DepartmentClient
from edms_ai_assistant.clients.group_client import GroupClient
from edms_ai_assistant.clients.reference_client import ReferenceClient
from edms_ai_assistant.clients.attachment_client import AttachmentClient
from edms_ai_assistant.clients.access_grief_client import AccessGriefClient
from edms_ai_assistant.clients.control_client import ControlClient
from edms_ai_assistant.clients.task_client import TaskClient

from edms_ai_assistant.services.document_service import DocumentService
from edms_ai_assistant.services.document_enricher import DocumentEnricher
from edms_ai_assistant.services.appeal_extraction_service import AppealExtractionService
from edms_ai_assistant.services.appeal_autofill_service import AppealAutofillService
from edms_ai_assistant.services.resolution_service import ResolutionService
from edms_ai_assistant.services.task_service import TaskService
from edms_ai_assistant.services.introduction_service import IntroductionService
from edms_ai_assistant.services.file_processor import FileProcessorService
from edms_ai_assistant.services.nlp_service import EDMSNaturalLanguageService
from edms_ai_assistant.services.entity_extractor import EntityExtractor
from edms_ai_assistant.services.query_refiner import QueryRefiner
from edms_ai_assistant.config import edms_settings, settings


class AppDeps(BaseModel):
    """Контейнер зависимостей приложения (Application Dependencies)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    transport: IAsyncTransport
    redis: aioredis.Redis
    base_client: EdmsBaseClient

    # Клиенты
    document_client: DocumentClient
    employee_client: EmployeeClient
    department_client: DepartmentClient
    group_client: GroupClient
    reference_client: ReferenceClient
    attachment_client: AttachmentClient
    access_grief_client: AccessGriefClient
    control_client: ControlClient
    document_creator_client: DocumentCreatorClient
    task_client: TaskClient

    # Сервисы
    document_service: DocumentService
    appeal_extraction_service: AppealExtractionService
    appeal_autofill_service: AppealAutofillService
    resolution_service: ResolutionService
    task_service: TaskService
    introduction_service: IntroductionService
    file_processor_service: FileProcessorService
    nlp_service: EDMSNaturalLanguageService

    # Опциональные сервисы (инициализируемые позже в lifespan)
    summarization_service: Optional[Any] = None


def init_deps(transport: IAsyncTransport, redis: aioredis.Redis, llm: Any) -> AppDeps:
    """Фабрика для создания и связывания всех зависимостей приложения."""

    base_client = EdmsBaseClient(transport, edms_settings)

    # ── Инициализация клиентов ────────────────────────────────────────────────
    document_client = DocumentClient(transport, edms_settings)
    employee_client = EmployeeClient(transport, edms_settings)
    department_client = DepartmentClient(transport, edms_settings)
    group_client = GroupClient(transport, edms_settings)
    reference_client = ReferenceClient(transport, edms_settings)
    attachment_client = AttachmentClient(transport, edms_settings)
    access_grief_client = AccessGriefClient(transport, edms_settings)
    control_client = ControlClient(transport, edms_settings)
    document_creator_client = DocumentCreatorClient(transport, edms_settings)
    task_client = TaskClient(transport, edms_settings)

    # ── Инициализация сервисов ───────────────────────────────────────────────
    entity_extractor = EntityExtractor()
    query_refiner = QueryRefiner()
    nlp_service = EDMSNaturalLanguageService(entity_extractor=entity_extractor, query_refiner=query_refiner)
    enricher = DocumentEnricher(document_client)

    document_service = DocumentService(
        document_client=document_client,
        document_enricher=enricher,
        nlp_service=nlp_service,
        redis=redis,
        cache_ttl=settings.CACHE_TTL_SECONDS
    )

    resolution_service = ResolutionService(
        employee_client=employee_client,
        department_client=department_client,
        group_client=group_client,
    )

    appeal_extraction_service = AppealExtractionService(llm=llm)
    appeal_autofill_service = AppealAutofillService(
        doc_client=document_client,
        attach_client=attachment_client,
        ref_client=reference_client,
        extraction_service=appeal_extraction_service,
        chat_model=llm
    )
    task_service = TaskService(resolution_service=resolution_service, task_client=task_client)
    introduction_service = IntroductionService(resolution_service=resolution_service, transport=transport)
    file_processor_service = FileProcessorService()

    return AppDeps(
        transport=transport,
        redis=redis,
        base_client=base_client,
        document_client=document_client,
        employee_client=employee_client,
        department_client=department_client,
        group_client=group_client,
        reference_client=reference_client,
        attachment_client=attachment_client,
        access_grief_client=access_grief_client,
        control_client=control_client,
        document_creator_client=document_creator_client,
        task_client=task_client,
        document_service=document_service,
        appeal_extraction_service=appeal_extraction_service,
        appeal_autofill_service=appeal_autofill_service,
        resolution_service=resolution_service,
        task_service=task_service,
        introduction_service=introduction_service,
        file_processor_service=file_processor_service,
        nlp_service=nlp_service,
    )

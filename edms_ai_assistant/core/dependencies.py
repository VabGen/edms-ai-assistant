# edms_ai_assistant/core/dependencies.py
from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

import redis.asyncio as aioredis
from fastapi import Depends

from edms_ai_assistant.clients.access_grief_client import AccessGriefClient
from edms_ai_assistant.clients.attachment_client import AttachmentClient
from edms_ai_assistant.clients.control_client import ControlClient
from edms_ai_assistant.clients.department_client import DepartmentClient
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.clients.document_creator_client import DocumentCreatorClient
from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.clients.group_client import GroupClient
from edms_ai_assistant.clients.reference_client import ReferenceClient
from edms_ai_assistant.clients.task_client import TaskClient
from edms_ai_assistant.clients.transport import HttpxTransport, IAsyncTransport
from edms_ai_assistant.config import EdmsSettings, settings
from edms_ai_assistant.services.appeal_autofill_service import AppealAutofillService
from edms_ai_assistant.services.appeal_extraction_service import AppealExtractionService
from edms_ai_assistant.services.document_enricher import DocumentEnricher
from edms_ai_assistant.services.document_service import DocumentService
from edms_ai_assistant.services.entity_extractor import EntityExtractor
from edms_ai_assistant.services.file_processor import FileProcessorService
from edms_ai_assistant.services.introduction_service import IntroductionService
from edms_ai_assistant.services.nlp_service import EDMSNaturalLanguageService
from edms_ai_assistant.services.query_refiner import QueryRefiner
from edms_ai_assistant.services.resolution_service import ResolutionService
from edms_ai_assistant.services.subject_service import SubjectService
from edms_ai_assistant.services.task_service import TaskService

# ── Настройки ────────────────────────────────────────────────────────────


@lru_cache
def get_edms_settings() -> EdmsSettings:
    from edms_ai_assistant.config import edms_settings

    return edms_settings


# ── Транспорт ────────────────────────────────────────────────────────────


@lru_cache
def get_transport() -> HttpxTransport:
    s = get_edms_settings()
    return HttpxTransport(base_url=str(s.base_url), default_timeout=s.timeout)


@lru_cache
def get_redis() -> aioredis.Redis:
    return aioredis.from_url(settings.REDIS_URL, decode_responses=True)


# ── Клиенты ──────────────────────────────────────────────────────────────


def get_document_client(
    transport: Annotated[IAsyncTransport, Depends(get_transport)],
    edms_settings: Annotated[EdmsSettings, Depends(get_edms_settings)],
) -> DocumentClient:
    return DocumentClient(transport, edms_settings)


def get_employee_client(
    transport: Annotated[IAsyncTransport, Depends(get_transport)],
    edms_settings: Annotated[EdmsSettings, Depends(get_edms_settings)],
) -> EmployeeClient:
    return EmployeeClient(transport, edms_settings)


def get_reference_client(
    transport: Annotated[IAsyncTransport, Depends(get_transport)],
    edms_settings: Annotated[EdmsSettings, Depends(get_edms_settings)],
) -> ReferenceClient:
    return ReferenceClient(transport, edms_settings)


def get_creator_client(
    transport: Annotated[IAsyncTransport, Depends(get_transport)],
    edms_settings: Annotated[EdmsSettings, Depends(get_edms_settings)],
) -> DocumentCreatorClient:
    return DocumentCreatorClient(transport, edms_settings)


def get_task_client(
    transport: Annotated[IAsyncTransport, Depends(get_transport)],
    edms_settings: Annotated[EdmsSettings, Depends(get_edms_settings)],
) -> TaskClient:
    return TaskClient(transport, edms_settings)


def get_department_client(
    transport: Annotated[IAsyncTransport, Depends(get_transport)],
    edms_settings: Annotated[EdmsSettings, Depends(get_edms_settings)],
) -> DepartmentClient:
    return DepartmentClient(transport, edms_settings)


def get_group_client(
    transport: Annotated[IAsyncTransport, Depends(get_transport)],
    edms_settings: Annotated[EdmsSettings, Depends(get_edms_settings)],
) -> GroupClient:
    return GroupClient(transport, edms_settings)


def get_attachment_client(
    transport: Annotated[IAsyncTransport, Depends(get_transport)],
    edms_settings: Annotated[EdmsSettings, Depends(get_edms_settings)],
) -> AttachmentClient:
    return AttachmentClient(transport, edms_settings)


def get_access_grief_client(
    transport: Annotated[IAsyncTransport, Depends(get_transport)],
    edms_settings: Annotated[EdmsSettings, Depends(get_edms_settings)],
) -> AccessGriefClient:
    return AccessGriefClient(transport, edms_settings)


def get_control_client(
    transport: Annotated[IAsyncTransport, Depends(get_transport)],
    edms_settings: Annotated[EdmsSettings, Depends(get_edms_settings)],
) -> ControlClient:
    return ControlClient(transport, edms_settings)


# ── LLM ──────────────────────────────────────────────────────────────────


def get_chat_model() -> Any:
    from edms_ai_assistant.llm import get_chat_model as gcm

    return gcm()


# ── Сервисы ──────────────────────────────────────────────────────────────


def get_entity_extractor() -> EntityExtractor:
    return EntityExtractor()


def get_query_refiner() -> QueryRefiner:
    return QueryRefiner()


def get_nlp_service(
    entity_extractor: Annotated[EntityExtractor, Depends(get_entity_extractor)],
    query_refiner: Annotated[QueryRefiner, Depends(get_query_refiner)],
) -> EDMSNaturalLanguageService:
    return EDMSNaturalLanguageService(
        entity_extractor=entity_extractor, query_refiner=query_refiner
    )


def get_enricher(
    document_client: Annotated[DocumentClient, Depends(get_document_client)],
) -> DocumentEnricher:
    return DocumentEnricher(document_client)


def get_document_service(
    document_client: Annotated[DocumentClient, Depends(get_document_client)],
    enricher: Annotated[DocumentEnricher, Depends(get_enricher)],
    nlp_service: Annotated[EDMSNaturalLanguageService, Depends(get_nlp_service)],
    redis: Annotated[aioredis.Redis, Depends(get_redis)],
) -> DocumentService:
    return DocumentService(
        document_client=document_client,
        document_enricher=enricher,
        nlp_service=nlp_service,
        redis=redis,
        cache_ttl=settings.CACHE_TTL_SECONDS,
    )


def get_resolution_service(
    employee_client: Annotated[EmployeeClient, Depends(get_employee_client)],
    department_client: Annotated[DepartmentClient, Depends(get_department_client)],
    group_client: Annotated[GroupClient, Depends(get_group_client)],
) -> ResolutionService:
    return ResolutionService(
        employee_client=employee_client,
        department_client=department_client,
        group_client=group_client,
    )


def get_task_service(
    resolution_service: Annotated[ResolutionService, Depends(get_resolution_service)],
    task_client: Annotated[TaskClient, Depends(get_task_client)],
) -> TaskService:
    return TaskService(resolution_service=resolution_service, task_client=task_client)


def get_introduction_service(
    resolution_service: Annotated[ResolutionService, Depends(get_resolution_service)],
    transport: Annotated[IAsyncTransport, Depends(get_transport)],
) -> IntroductionService:
    return IntroductionService(
        resolution_service=resolution_service, transport=transport
    )


def get_appeal_extraction_service(
    llm: Annotated[Any, Depends(get_chat_model)],
) -> AppealExtractionService:
    return AppealExtractionService(llm=llm)


def get_appeal_autofill_service(
    doc_client: Annotated[DocumentClient, Depends(get_document_client)],
    attach_client: Annotated[AttachmentClient, Depends(get_attachment_client)],
    ref_client: Annotated[ReferenceClient, Depends(get_reference_client)],
    extraction_service: Annotated[
        AppealExtractionService, Depends(get_appeal_extraction_service)
    ],
    llm: Annotated[Any, Depends(get_chat_model)],
) -> AppealAutofillService:
    return AppealAutofillService(
        doc_client=doc_client,
        attach_client=attach_client,
        ref_client=ref_client,
        extraction_service=extraction_service,
        chat_model=llm,
    )


def get_file_processor_service() -> FileProcessorService:
    return FileProcessorService()


def get_subject_service(
    ref_client: Annotated[ReferenceClient, Depends(get_reference_client)],
) -> SubjectService:
    return SubjectService(ref_client)


# ── Псевдонимы ───────────────────────────────────────────────────────────

DocService = Annotated[DocumentService, Depends(get_document_service)]
ResolutionSer = Annotated[ResolutionService, Depends(get_resolution_service)]
TaskSer = Annotated[TaskService, Depends(get_task_service)]
IntroSer = Annotated[IntroductionService, Depends(get_introduction_service)]
AppealExtSer = Annotated[
    AppealExtractionService, Depends(get_appeal_extraction_service)
]
AppealAutoSer = Annotated[AppealAutofillService, Depends(get_appeal_autofill_service)]
FileProcSer = Annotated[FileProcessorService, Depends(get_file_processor_service)]
SubjService = Annotated[SubjectService, Depends(get_subject_service)]

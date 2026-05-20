# edms_ai_assistant/core/deps.py
from __future__ import annotations

from pydantic import BaseModel

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.clients.document_creator_client import DocumentCreatorClient
from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.clients.department_client import DepartmentClient
from edms_ai_assistant.clients.group_client import GroupClient
from edms_ai_assistant.clients.reference_client import ReferenceClient
from edms_ai_assistant.clients.attachment_client import AttachmentClient
from edms_ai_assistant.clients.access_grief_client import AccessGriefClient
from edms_ai_assistant.clients.control_client import ControlClient

from edms_ai_assistant.services.document_service import DocumentService
from edms_ai_assistant.services.appeal_extraction_service import AppealExtractionService
from edms_ai_assistant.services.appeal_autofill_service import AppealAutofillService
from edms_ai_assistant.services.resolution_service import ResolutionService
from edms_ai_assistant.services.task_service import TaskService
from edms_ai_assistant.services.introduction_service import IntroductionService
from edms_ai_assistant.services.file_processor import FileProcessorService
from edms_ai_assistant.services.nlp_service import EDMSNaturalLanguageService


class AppDeps(BaseModel):
    """Контейнер зависимостей приложения (Application Dependencies).

    Используется для внедрения зависимостей (DI) в инструменты (Tool Factories)
    и обработчики API. Содержит заранее сконфигурированные инстансы клиентов и сервисов.
    """
    model_config = {"arbitrary_types_allowed": True}

    # Базовый клиент (нужен для Tool Factories, делающих прямые запросы)
    base_client: EdmsBaseClient

    # Клиенты (Транспортный слой)
    document_client: DocumentClient
    employee_client: EmployeeClient
    department_client: DepartmentClient
    group_client: GroupClient
    reference_client: ReferenceClient
    attachment_client: AttachmentClient
    access_grief_client: AccessGriefClient
    control_client: ControlClient
    document_creator_client: DocumentCreatorClient

    # Сервисы (Бизнес-логика)
    document_service: DocumentService
    appeal_extraction_service: AppealExtractionService
    appeal_autofill_service: AppealAutofillService
    resolution_service: ResolutionService
    task_service: TaskService
    introduction_service: IntroductionService
    file_processor_service: FileProcessorService
    nlp_service: EDMSNaturalLanguageService


def init_deps(base_client: EdmsBaseClient) -> AppDeps:
    """Фабрика для создания и связывания всех зависимостей приложения.

    Вызывается один раз при старте приложения (в lifespan).

    Args:
        base_client: Сконфигурированный базовый HTTP-клиент для СЭД.

    Returns:
        Полностью сконфигурированный контейнер AppDeps.
    """
    # ── Инициализация клиентов ────────────────────────────────────────────────
    document_client = DocumentClient(base_client)
    employee_client = EmployeeClient(base_client)
    department_client = DepartmentClient(base_client)
    group_client = GroupClient(base_client)
    reference_client = ReferenceClient(base_client)
    attachment_client = AttachmentClient(base_client)
    access_grief_client = AccessGriefClient(base_client)
    control_client = ControlClient(base_client)
    document_creator_client = DocumentCreatorClient(base_client)

    # ── Инициализация сервисов ───────────────────────────────────────────────
    # Внедряем клиенты в сервисы согласно их конструкторам
    resolution_service = ResolutionService(
        employee_client=employee_client,
        department_client=department_client,
        group_client=group_client,
    )

    file_processor_service = FileProcessorService()
    nlp_service = EDMSNaturalLanguageService()

    # Предполагаем, что остальные сервисы также принимают нужные клиенты
    document_service = DocumentService(
        document_client=document_client,
        # document_enricher=DocumentEnricher(base_client), # Если внедряем обогатитель
    )
    appeal_extraction_service = AppealExtractionService()
    appeal_autofill_service = AppealAutofillService(
        reference_client=reference_client,
        # employee_client=employee_client,
    )
    task_service = TaskService()
    introduction_service = IntroductionService()

    return AppDeps(
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
        document_service=document_service,
        appeal_extraction_service=appeal_extraction_service,
        appeal_autofill_service=appeal_autofill_service,
        resolution_service=resolution_service,
        task_service=task_service,
        introduction_service=introduction_service,
        file_processor_service=file_processor_service,
        nlp_service=nlp_service,
    )
# edms_ai_assistant/services/nlp_service.py
from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any, TYPE_CHECKING

from edms_ai_assistant.utils.edms_formatter import EdmsFormatter
from edms_ai_assistant.utils.format_utils import clean_dict

if TYPE_CHECKING:
    from edms_ai_assistant.services.entity_extractor import EntityExtractor
    from edms_ai_assistant.domain.document import DocumentDto
    from edms_ai_assistant.domain.employee import EmployeeDto
    from edms_ai_assistant.services.query_refiner import QueryRefiner

logger = logging.getLogger(__name__)


class UserIntent(StrEnum):
    """Enumeration of recognized user intents."""
    UNKNOWN = "UNKNOWN"
    FILE_ANALYSIS = "FILE_ANALYSIS"
    DOCUMENT_SUMMARY = "DOCUMENT_SUMMARY"
    TASK_CREATION = "TASK_CREATION"
    CREATE_INTRODUCTION = "CREATE_INTRODUCTION"
    CREATE_TASK = "CREATE_TASK"
    SUMMARIZE = "SUMMARIZE"
    COMPARE = "COMPARE"
    SEARCH = "SEARCH"
    ANALYZE = "ANALYZE"
    QUESTION = "QUESTION"
    CREATE_DOCUMENT = "CREATE_DOCUMENT"
    COMPLIANCE_CHECK = "COMPLIANCE_CHECK"
    CONTROL = "CONTROL"
    APPEAL_AUTOFILL = "APPEAL_AUTOFILL"


class EDMSNaturalLanguageService:
    """High-level service for semantic analysis of EDMS domain objects."""

    def __init__(self, entity_extractor: EntityExtractor, query_refiner: QueryRefiner):
        self.entity_extractor = entity_extractor
        self.query_refiner = query_refiner

    def process_document(self, doc: DocumentDto) -> dict[str, Any]:
        """Produce a full structured analysis of a DocumentDto."""
        if not doc:
            logger.warning("Attempted to process None document")
            return {}

        try:
            # 1. Базовая информация
            base_info = {
                "id": str(doc.id) if doc.id else None,
                "категория": str(doc.doc_category_const or ""),
                "краткое_содержание": doc.short_summary,
                "вид_документа": doc.document_type.type_name if doc.document_type else None,
                "способ_создания": str(doc.create_type or ""),
                "_reg_date_iso": EdmsFormatter.format_date_iso(doc.reg_date),
                "_create_date_iso": EdmsFormatter.format_date_iso(doc.create_date),
            }

            # 2. Регистрация
            registration = {
                "рег_номер": doc.reg_number,
                "дата_регистрации": EdmsFormatter.format_date(doc.reg_date),
                "дата_создания": EdmsFormatter.format_datetime(doc.create_date),
                "исходящий_номер": doc.out_number,
                "исходящая_дата": EdmsFormatter.format_date(doc.out_date),
            }

            # 3. Участники (пример доступа к полям через модель)
            participants = {
                "автор": EdmsFormatter.format_user(getattr(doc, "author", None)),
                "инициатор": EdmsFormatter.format_user(getattr(doc, "initiator", None)),
                "ответственный_исполнитель": EdmsFormatter.format_user(getattr(doc, "responsible_executor", None)),
            }

            # 4. Жизненный цикл
            lifecycle = {
                "текущий_статус": str(doc.status or ""),
            }

            # 5. Контроль
            control_info = {
                "на_контроле": doc.on_control,
            }

            # Собираем все блоки
            result: dict[str, Any] = {
                "базовая_информация": clean_dict(base_info),
                "регистрация": clean_dict(registration),
                "участники": clean_dict(participants),
                "жизненный_цикл": clean_dict(lifecycle),
                "контроль": clean_dict(control_info),
            }

            # Добавляем инфо по обращению если есть
            if doc.document_appeal:
                appeal = doc.document_appeal
                result["информация_об_обращении"] = clean_dict({
                    "тип_заявителя": str(appeal.declarant_type or ""),
                    "фио_заявителя": appeal.fio_applicant,
                    "организация": appeal.organization_name,
                    "адрес": appeal.full_address,
                    "email": appeal.email,
                    "телефон": appeal.phone,
                })

            return result

        except Exception as exc:
            logger.error("Error processing document: %s", exc, exc_info=True)
            return {"error": "Ошибка обработки документа", "details": str(exc)}

    def process_employee_info(self, emp: EmployeeDto) -> dict[str, Any]:
        """Build an analytical employee card from an EmployeeDto."""
        if not emp: return {}
        try:
            return {
                "основное": {
                    "фио": EdmsFormatter.format_user(emp),
                    "должность": emp.post.post_name if emp.post else None,
                    "департамент": emp.department.name if emp.department else None,
                },
                "контакты": {
                    "email": emp.email,
                    "phone": emp.phone,
                },
            }
        except Exception as exc:
            logger.error("Error processing employee info: %s", exc, exc_info=True)
            return {"error": str(exc)}

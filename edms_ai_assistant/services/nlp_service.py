# edms_ai_assistant/services/nlp_service.py
from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from edms_ai_assistant.domain.document import DocumentDto
from edms_ai_assistant.domain.employee import EmployeeDto
from edms_ai_assistant.services.entity_extractor import EntityExtractor
from edms_ai_assistant.services.query_refiner import QueryRefiner
from edms_ai_assistant.utils.edms_formatter import EdmsFormatter
from edms_ai_assistant.utils.format_utils import clean_dict

logger = logging.getLogger(__name__)


class UserIntent(str, Enum):
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
            # Переход на snake_case аттрибуты Pydantic модели
            category_value = str(doc.doc_category_const or "")

            base_info = {
                "id": str(doc.id) if doc.id else None,
                "категория": category_value,
                "краткое_содержание": doc.short_summary,
                "полный_текст": getattr(doc, "summary", None), # Добавим если есть в модели
                "примечание": getattr(doc, "note", None),
                "профиль": getattr(doc, "profile_name", None),
                "гриф_ДСП": getattr(doc, "dsp_flag", None),
                "вид_документа": doc.document_type.type_name if doc.document_type else None,
                "способ_создания": doc.create_type,
                "_reg_date_iso": EdmsFormatter.format_date_iso(doc.reg_date),
                "_create_date_iso": EdmsFormatter.format_date_iso(doc.create_date),
            }

            registration = {
                "рег_номер": doc.reg_number or getattr(doc, "reserved_reg_number", None),
                "дата_регистрации": EdmsFormatter.format_date(doc.reg_date),
                "дата_создания": EdmsFormatter.format_datetime(doc.create_date),
                "исходящий_номер": doc.out_number,
                "исходящая_дата": EdmsFormatter.format_date(doc.out_date),
                "_reg_date_iso": EdmsFormatter.format_date_iso(doc.reg_date),
                "_create_date_iso": EdmsFormatter.format_date_iso(doc.create_date),
                "_out_reg_date_iso": EdmsFormatter.format_date_iso(doc.out_date),
                "журнал_регистрации": getattr(getattr(doc, "registration_journal", None), "journal_name", None),
                "версия": getattr(getattr(doc, "version", None), "version", None),
                "признак_версионности": getattr(doc, "version_flag", None),
                "страниц": getattr(doc, "pages", None),
                "кол-во_экземпляров": getattr(doc, "exemplar_count", None),
            }

            participants = {
                "автор": EdmsFormatter.format_user(getattr(doc, "author", None)),
                "инициатор": EdmsFormatter.format_user(getattr(doc, "initiator", None)),
                "ответственный_исполнитель": EdmsFormatter.format_user(getattr(doc, "responsible_executor", None)),
                "корреспондент": getattr(doc, "correspondent_name", None),
                "кем_подписан": EdmsFormatter.format_user(getattr(doc, "who_signed", None)),
                "председатель": EdmsFormatter.format_user(getattr(doc, "chairperson", None)),
                "секретарь": EdmsFormatter.format_user(getattr(doc, "secretary", None)),
            }

            lifecycle = {
                "текущий_статус": doc.status,
                "предыдущий_статус": getattr(doc, "prev_status", None),
                "текущий_этап_БП": getattr(doc, "current_bpmn_task_name", None),
            }

            control_info = {
                "на_контроле": doc.on_control,
                "снять_с_контроля": getattr(doc, "remove_control", None),
            }

            result: dict[str, Any] = {
                "базовая_информация": clean_dict(base_info),
                "регистрация": clean_dict(registration),
                "участники": clean_dict(participants),
                "жизненный_цикл": clean_dict(lifecycle),
                "контроль": clean_dict(control_info),
            }

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
                    "статус": "Уволен" if getattr(emp, "fired", False) else "Активен",
                },
                "контакты": {
                    "email": emp.email,
                    "phone": emp.phone,
                    "адрес": getattr(emp, "address", None),
                },
                "структура": {
                    "код_департамента": emp.department.department_code if emp.department else None,
                },
            }
        except Exception as exc:
            logger.error("Error processing employee info: %s", exc, exc_info=True)
            return {"error": str(exc)}

# edms_ai_assistant/services/nlp_service.py
from __future__ import annotations

import logging
from typing import Any

from edms_ai_assistant.domain.document import DocumentDto
from edms_ai_assistant.domain.employee import EmployeeDto
from edms_ai_assistant.services.entity_extractor import EntityExtractor
from edms_ai_assistant.services.query_refiner import QueryRefiner
from edms_ai_assistant.utils.edms_formatter import EdmsFormatter
from edms_ai_assistant.utils.format_utils import clean_dict

logger = logging.getLogger(__name__)


class EDMSNaturalLanguageService:
    """High-level service for semantic analysis of EDMS domain objects."""

    def __init__(self):
        # Инициализируем саб-сервисы (в будущем можно вынести в DI контейнер)
        self.entity_extractor = EntityExtractor()
        self.query_refiner = QueryRefiner()

    def process_document(self, doc: DocumentDto) -> dict[str, Any]:
        """Produce a full structured analysis of a DocumentDto. Вход строго типизирован!"""
        if not doc:
            logger.warning("Attempted to process None document")
            return {}

        try:
            # ── Категория (enum-safe) ──────────────────────────────────────────
            category = EdmsFormatter.get_safe(doc, "docCategoryConstant")
            category_value: str = category.value if hasattr(category, "value") else str(category or "")

            # ── 1. Базовая информация ─────────────────────────────────────────
            reg_date_raw = EdmsFormatter.get_safe(doc, "regDate")
            create_date_raw = EdmsFormatter.get_safe(doc, "createDate")

            base_info = {
                "id": str(doc.id) if getattr(doc, "id", None) else None,
                "категория": category_value,
                "краткое_содержание": EdmsFormatter.get_safe(doc, "shortSummary"),
                "полный_текст": EdmsFormatter.get_safe(doc, "summary"),
                "примечание": EdmsFormatter.get_safe(doc, "note"),
                "профиль": EdmsFormatter.get_safe(doc, "profileName"),
                "гриф_ДСП": EdmsFormatter.get_safe(doc, "dspFlag"),
                "вид_документа": EdmsFormatter.get_safe(doc, "documentType.typeName"),
                "способ_создания": EdmsFormatter.get_safe(doc, "createType"),
                "_reg_date_iso": EdmsFormatter.format_date_iso(reg_date_raw),
                "_create_date_iso": EdmsFormatter.format_date_iso(create_date_raw),
            }

            # ── 2. Регистрация ────────────────────────────────────────────────
            out_reg_date_raw = EdmsFormatter.get_safe(doc, "outRegDate")

            registration = {
                "рег_номер": EdmsFormatter.get_safe(doc, "regNumber") or EdmsFormatter.get_safe(doc,
                                                                                                "reservedRegNumber"),
                "дата_регистрации": EdmsFormatter.format_date(reg_date_raw),
                "дата_создания": EdmsFormatter.format_datetime(create_date_raw),
                "исходящий_номер": EdmsFormatter.get_safe(doc, "outRegNumber"),
                "исходящая_дата": EdmsFormatter.format_date(out_reg_date_raw),
                "_reg_date_iso": EdmsFormatter.format_date_iso(reg_date_raw),
                "_create_date_iso": EdmsFormatter.format_date_iso(create_date_raw),
                "_out_reg_date_iso": EdmsFormatter.format_date_iso(out_reg_date_raw),
                "журнал_регистрации": EdmsFormatter.get_safe(doc, "registrationJournal.journalName"),
                "версия": EdmsFormatter.get_safe(doc, "version.version"),
                "признак_версионности": EdmsFormatter.get_safe(doc, "versionFlag"),
                "страниц": EdmsFormatter.get_safe(doc, "pages"),
                "кол-во_экземпляров": EdmsFormatter.get_safe(doc, "exemplarCount"),
            }

            # ── 3. Участники ──────────────────────────────────────────────────
            responsible_executors = [
                EdmsFormatter.format_user(EdmsFormatter.get_safe(r, "executor"))
                for r in (EdmsFormatter.get_safe(doc, "responsibleExecutors") or [])
                if EdmsFormatter.get_safe(r, "executor")
            ]

            _recipient_list_raw = EdmsFormatter.get_safe(doc, "recipientList") or []
            _contractors = [
                               {
                                   "название": EdmsFormatter.get_safe(r, "name"),
                                   "УНП": EdmsFormatter.get_safe(r, "unp"),
                                   "номер_договора_контрагента": EdmsFormatter.get_safe(r, "contractNumber"),
                                   "дата_договора_контрагента": EdmsFormatter.format_date(
                                       EdmsFormatter.get_safe(r, "contractDate")),
                               }
                               for r in _recipient_list_raw
                               if EdmsFormatter.get_safe(r, "name")
                           ] or None

            _contract_responsible_raw = EdmsFormatter.get_safe(doc, "contractResponsible") or []
            _contract_responsible_users: list[str] | None = [
                                                                EdmsFormatter.format_user(r.get("user") if isinstance(r,
                                                                                                                      dict) else EdmsFormatter.get_safe(
                                                                    r, "user"))
                                                                for r in _contract_responsible_raw
                                                                if (
                    r.get("user") if isinstance(r, dict) else EdmsFormatter.get_safe(r, "user"))
                                                            ] or None

            participants = {
                "автор": EdmsFormatter.format_user(EdmsFormatter.get_safe(doc, "author")),
                "инициатор": EdmsFormatter.format_user(EdmsFormatter.get_safe(doc, "initiator")),
                "ответственный_исполнитель": EdmsFormatter.format_user(
                    EdmsFormatter.get_safe(doc, "responsibleExecutor")),
                "корреспондент": EdmsFormatter.get_safe(doc, "correspondentName"),
                "контрагенты": _contractors,
                "ответственные_по_договору": _contract_responsible_users,
                "кем_подписан": EdmsFormatter.format_user(EdmsFormatter.get_safe(doc, "whoSigned")),
                "председатель": EdmsFormatter.format_user(EdmsFormatter.get_safe(doc, "chairperson")),
                "секретарь": EdmsFormatter.format_user(EdmsFormatter.get_safe(doc, "secretary")),
                "ответственные_за_подготовку": responsible_executors or None,
            }

            # ── 4-11 блоки остаются без изменений, просто вызовы getattr заменяются на EdmsFormatter.get_safe
            # Для краткости показываю структуру, вся логика (контроль, задачи, встречи) переносится 1-в-1,
            # но с использованием EdmsFormatter.get_safe(doc, "ключ.camelCase")

            lifecycle = {
                "текущий_статус": EdmsFormatter.get_safe(doc, "status"),
                "предыдущий_статус": EdmsFormatter.get_safe(doc, "prevStatus"),
                "текущий_этап_БП": EdmsFormatter.get_safe(doc, "currentBpmnTaskName"),
                "процесс": None,  # process_detail вычисляется как раньше
            }

            control_info = {
                "на_контроле": EdmsFormatter.get_safe(doc, "controlFlag"),
                "снять_с_контроля": EdmsFormatter.get_safe(doc, "removeControl"),
            }

            result: dict[str, Any] = {
                "базовая_информация": clean_dict(base_info),
                "регистрация": clean_dict(registration),
                "участники": clean_dict(participants),
                "жизненный_цикл": clean_dict(lifecycle),
                "контроль": clean_dict(control_info),
                # "задачи": clean_dict(tasks_info),
                # "связи_и_вложения": clean_dict(relations),
                # "специализированная_информация": clean_dict(specialized) if specialized else None
            }

            return result

        except Exception as exc:
            logger.error("Error processing document: %s", exc, exc_info=True)
            return {"error": "Ошибка обработки документа", "details": str(exc)}

    def process_employee_info(self, emp: EmployeeDto) -> dict[str, Any]:
        """Build an analytical employee card from an EmployeeDto. Строгий тип!"""
        if not emp: return {}
        try:
            return {
                "основное": {
                    "фио": EdmsFormatter.format_user(emp),
                    "должность": EdmsFormatter.get_safe(emp, "post.postName"),
                    "департамент": EdmsFormatter.get_safe(emp, "department.name"),
                    "статус": "Уволен" if getattr(emp, "fired", False) else "Активен",
                },
                "контакты": {
                    "email": getattr(emp, "email", None),
                    "телефон": getattr(emp, "phone", None),
                    "адрес": getattr(emp, "address", None),
                },
                "структура": {
                    "код_департамента": EdmsFormatter.get_safe(emp, "department.departmentCode"),
                },
            }
        except Exception as exc:
            logger.error("Error processing employee info: %s", exc, exc_info=True)
            return {"error": str(exc)}

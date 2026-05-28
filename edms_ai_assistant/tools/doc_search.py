# edms_ai_assistant/tools/doc_search.py
"""
EDMS AI Assistant — Document Search Tool.

Поиск документов в EDMS по широкому набору параметров фильтрации.

Маппинг параметров инструмента -> поля DocumentFilter (Java):
    short_summary        -> shortSummary         (String, like)
    reg_number           -> regNumber            (String, like)
    out_reg_number       -> outRegNumber         (String, like)
    doc_category         -> categoryConstants    (DocumentCategoryConstants[])
    date_from            -> dateRegStart         (Instant ISO 8601)
    date_to              -> dateRegEnd           (Instant ISO 8601)
    date_control_start   -> dateControlStart     (Instant ISO 8601)
    date_control_end     -> dateControlEnd       (Instant ISO 8601)
    author_last_name     -> authorLastName       (String, like)
    correspondent_name   -> correspondentName    (String, like)
    recipient_name       -> recipientName        (String, like)
    task_executor_last_name -> taskExecutorLastName (String, like)
    author_current_user  -> authorCurrentUser    (Boolean)
    process_executor_current_user -> processExecutorCurrentUser (Boolean)
    task_executor_current_user    -> taskExecutorCurrentUser    (Boolean)
    control_user_current_user     -> controlUserCurrentUser     (Boolean)
    introduction_current_user     -> introductionCurrentUser    (Boolean)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.tools import InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field, field_validator, model_validator

from edms_ai_assistant.agent.hitl_primitives import ask_human
from edms_ai_assistant.agent.interrupt_contract import (
    CardSelectInterrupt,
    CardSelectResume,
    InterruptCard,
)
from edms_ai_assistant.agent.runnable_utils import get_token_from_config

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from edms_ai_assistant.clients.document_client import DocumentClient
    from edms_ai_assistant.domain.document import DocumentDto

logger = logging.getLogger(__name__)

_MAX_RESULTS: int = 20
_VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "INTERN",
        "INCOMING",
        "OUTGOING",
        "MEETING",
        "QUESTION",
        "MEETING_QUESTION",
        "APPEAL",
        "CONTRACT",
        "CUSTOM",
    }
)
_VALID_STATUSES: frozenset[str] = frozenset(
    {
        "FORMING",
        "REGISTRATION",
        "IN_PROGRESS",
        "COMPLETED",
        "CANCELLED",
        "ARCHIVE",
        "ON_SIGNING",
        "ON_AGREEMENT",
        "ON_REVIEW",
        "ON_STATEMENT",
    }
)
_DATE_PATTERN: str = r"^\d{4}-\d{2}-\d{2}$|^$"


class DocSearchInput(BaseModel):
    """Validated input schema for the document search tool."""

    short_summary: str | None = Field(
        None, max_length=500, description="Поиск по краткому содержанию."
    )
    reg_number: str | None = Field(
        None, max_length=100, description="Регистрационный номер."
    )
    out_reg_number: str | None = Field(
        None, max_length=100, description="Исходящий регистрационный номер."
    )
    doc_category: str | None = Field(None, description="Категория документа.")
    status: str | None = Field(
        None, description="Статус документа. Передавай ОДНУ строку."
    )
    date_from: str | None = Field(
        None,
        description="Начало дат регистрации. Формат: YYYY-MM-DD.",
        pattern=_DATE_PATTERN,
    )
    date_to: str | None = Field(
        None,
        description="Конец дат регистрации. Формат: YYYY-MM-DD.",
        pattern=_DATE_PATTERN,
    )
    date_control_start: str | None = Field(
        None,
        description="Начало дат контроля. Формат: YYYY-MM-DD.",
        pattern=_DATE_PATTERN,
    )
    date_control_end: str | None = Field(
        None,
        description="Конец дат контроля. Формат: YYYY-MM-DD.",
        pattern=_DATE_PATTERN,
    )
    author_last_name: str | None = Field(
        None, max_length=150, description="Фамилия автора."
    )
    correspondent_name: str | None = Field(
        None, max_length=300, description="Организация-корреспондент."
    )
    recipient_name: str | None = Field(
        None, max_length=300, description="Адресат документа."
    )
    task_executor_last_name: str | None = Field(
        None, max_length=150, description="Фамилия исполнителя поручения."
    )
    author_current_user: bool | None = Field(
        None, description="True — автор = текущий пользователь."
    )
    process_executor_current_user: bool | None = Field(
        None, description="True — участник активного процесса."
    )
    task_executor_current_user: bool | None = Field(
        None, description="True — исполнитель поручения."
    )
    control_user_current_user: bool | None = Field(
        None, description="True — контролёр."
    )
    introduction_current_user: bool | None = Field(
        None, description="True — на ознакомлении."
    )
    display_mode: str | None = Field(
        None,
        description=(
            "Формат вывода результатов. "
            "Установите 'cards', если пользователь просит ПОКАЗАТЬ, НАЙТИ ДЛЯ ПРОСМОТРА или ОТКРЫТЬ документы. "
            "Если параметр опущен или 'data', возвращается JSON для подсчета, статистики или внутреннего анализа."
        ),
    )
    config: Annotated[Any, InjectedToolArg] = None

    @field_validator(
        "short_summary",
        "reg_number",
        "out_reg_number",
        "author_last_name",
        "correspondent_name",
        "recipient_name",
        "task_executor_last_name",
        mode="before",
    )
    @classmethod
    def strip_and_none(cls, v: str | None) -> str | None:
        if v is None:
            return None
        stripped = v.strip()
        return stripped if stripped else None

    @field_validator("doc_category", mode="before")
    @classmethod
    def validate_category(cls, v: str | None) -> str | None:
        if v is None:
            return None
        upper = v.strip().upper()
        if not upper:
            return None
        if upper not in _VALID_CATEGORIES:
            raise ValueError(f"Неизвестная категория: '{upper}'")
        return upper

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, list):
            if not v:
                return None
            v = v[0]
        if not isinstance(v, str):
            v = str(v)
        upper = v.strip().upper()
        if not upper:
            return None
        return upper

    @model_validator(mode="after")
    def at_least_one_search_param(self) -> DocSearchInput:
        fields = [
            self.short_summary,
            self.reg_number,
            self.out_reg_number,
            self.doc_category,
            self.status,
            self.date_from,
            self.date_to,
            self.date_control_start,
            self.date_control_end,
            self.author_last_name,
            self.correspondent_name,
            self.recipient_name,
            self.task_executor_last_name,
            self.author_current_user,
            self.process_executor_current_user,
            self.task_executor_current_user,
            self.control_user_current_user,
            self.introduction_current_user,
        ]
        if not any(v is not None for v in fields):
            raise ValueError("Укажите хотя бы один параметр поиска.")
        return self


# ─── Tool Factory ─────────────────────────────────────────────────────────────


def create_doc_search_tool(document_client: DocumentClient) -> StructuredTool:
    """Фабрика для создания инструмента поиска документов.

    Args:
        document_client: Клиент для работы с документами EDMS.

    Returns:
        Настроенный StructuredTool, готовый к регистрации в агенте.
    """

    async def doc_search_tool(
        short_summary: str | None = None,
        reg_number: str | None = None,
        out_reg_number: str | None = None,
        doc_category: str | None = None,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        date_control_start: str | None = None,
        date_control_end: str | None = None,
        author_last_name: str | None = None,
        correspondent_name: str | None = None,
        recipient_name: str | None = None,
        task_executor_last_name: str | None = None,
        author_current_user: bool | None = None,
        process_executor_current_user: bool | None = None,
        task_executor_current_user: bool | None = None,
        control_user_current_user: bool | None = None,
        introduction_current_user: bool | None = None,
        display_mode: str | None = None,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Searches documents in EDMS by a wide range of filter criteria."""

        try:
            token = get_token_from_config(config)
        except Exception as e:
            logger.error("Failed to get token from config: %s", e)
            return {
                "status": "error",
                "message": f"Ошибка авторизации: токен не найден. {e}",
            }

        doc_filter: dict[str, Any] = {}
        if short_summary:
            doc_filter["shortSummary"] = short_summary
        if reg_number:
            doc_filter["regNumber"] = reg_number
        if out_reg_number:
            doc_filter["outRegNumber"] = out_reg_number
        if doc_category:
            doc_filter["categoryConstants"] = [doc_category]
        if date_from:
            doc_filter["dateRegStart"] = _to_iso_start(date_from)
        if date_to:
            doc_filter["dateRegEnd"] = _to_iso_end(date_to)
        if date_control_start:
            doc_filter["dateControlStart"] = _to_iso_start(date_control_start)
        if date_control_end:
            doc_filter["dateControlEnd"] = _to_iso_end(date_control_end)
        if author_last_name:
            doc_filter["authorLastName"] = author_last_name
        if correspondent_name:
            doc_filter["correspondentName"] = correspondent_name
        if recipient_name:
            doc_filter["recipientName"] = recipient_name
        if task_executor_last_name:
            doc_filter["taskExecutorLastName"] = task_executor_last_name

        # Spring обычно ожидает строку "true"/"false" для Boolean параметров query
        if author_current_user:
            doc_filter["authorCurrentUser"] = "true"
        if process_executor_current_user:
            doc_filter["processExecutorCurrentUser"] = "true"
        if task_executor_current_user:
            doc_filter["taskExecutorCurrentUser"] = "true"
        if control_user_current_user:
            doc_filter["controlUserCurrentUser"] = "true"
        if introduction_current_user:
            doc_filter["introductionCurrentUser"] = "true"

        pageable: dict[str, Any] = {"page": 0, "size": _MAX_RESULTS}
        includes = ["DOCUMENT_TYPE", "CORRESPONDENT", "REGISTRATION_JOURNAL"]

        logger.info(
            "Document search requested",
            extra={
                "filter_keys": list(doc_filter.keys()),
                "status_filter": status,
                "display_mode": display_mode,
            },
        )

        try:
            documents = await document_client.search_documents(
                token=token,
                doc_filter=doc_filter,
                pageable=pageable,
                includes=includes,
            )
        except Exception as exc:
            logger.error("Document search failed: %s", exc, exc_info=True)
            return {"status": "error", "message": f"Ошибка поиска документов: {exc}"}

        # Client-side фильтрация по статусу (если API не поддерживает напрямую)
        if status and documents:
            status_upper = status.strip().upper()
            documents = [
                d
                for d in documents
                if str(getattr(d, "status", "")).upper() == status_upper
            ]

        if not documents:
            return {
                "status": "success",
                "message": "Документы не найдены.",
                "documents": [],
                "total": 0,
            }

        serialized_docs = [_serialize_document(d) for d in documents[:10]]

        # ── Если LLM запросила формат "cards" для UI ──────────────────────
        if display_mode == "cards":
            cards = [
                InterruptCard(
                    id=doc["id"],
                    label=doc["reg_number"],
                    description=doc["short_summary"],
                    badges=[doc["category"], doc["status"]],
                    primary_attrs={
                        "Дата": doc["reg_date"],
                        "Автор": doc["author"],
                    },
                    metadata={
                        "url": f"/document-form/{doc['id']}",
                        "category": doc["category"],
                        "status": doc["status"],
                    },
                )
                for doc in serialized_docs
            ]

            # Выбрасываем Interrupt — это остановит граф и отправит JSON на фронтенд
            resume = ask_human(
                CardSelectInterrupt(
                    prompt=f"Найдено документов: {len(documents)}",
                    cards=cards,
                    multiple=False,
                )
            )

            # Граф возобновлен. Инструмент возвращает ID выбранного документа,
            # чтобы LLM могла продолжить работу (например, ответить "Открываю документ...")
            if isinstance(resume, CardSelectResume) and resume.selected_ids:
                selected_id = resume.selected_ids[0]
                return {
                    "status": "selected",
                    "selected_document_id": selected_id,
                    "message": "Пользователь выбрал документ из списка для просмотра.",
                }

            # Если пользователь прервал выбор (закрыл виджет и т.п.)
            return {
                "status": "cancelled",
                "message": "Пользователь отменил выбор документа.",
            }

        # ── Стандартный возврат JSON для аналитики ──────────────────────
        return {
            "status": "success",
            "total": len(documents),
            "shown": len(serialized_docs),
            "documents": serialized_docs,
            "message": f"Найдено {len(documents)} документ(ов).",
        }

    return StructuredTool.from_function(
        coroutine=doc_search_tool,
        name="doc_search_tool",
        description="Searches documents in EDMS by a wide range of filter criteria.",
        args_schema=DocSearchInput,
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _to_iso_start(date_str: str) -> str:
    return (
        datetime.fromisoformat(date_str)
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .strftime("%Y-%m-%dT%H:%M:%S.000Z")
    )


def _to_iso_end(date_str: str) -> str:
    return (
        datetime.fromisoformat(date_str)
        .replace(hour=23, minute=59, second=59, microsecond=999000)
        .strftime("%Y-%m-%dT%H:%M:%S.999Z")
    )


def _serialize_document(d: DocumentDto) -> dict[str, Any]:
    """
    Сериализует DocumentDto в плоский словарь для возврата инструментом.
    Использует getattr для безопасного доступа к полям, так как модель
    использует extra="allow", и некоторые поля могут не быть явно аннотированы.
    """
    author_obj = getattr(d, "author", None)
    author_name = "—"
    if author_obj:
        parts = [
            getattr(author_obj, "last_name", "") or "",
            getattr(author_obj, "first_name", "") or "",
            getattr(author_obj, "middle_name", "") or "",
        ]
        author_name = " ".join(p for p in parts if p).strip() or "—"

    return {
        "id": str(d.id) if d.id else "",
        "reg_number": d.reg_number or getattr(d, "reserved_reg_number", None) or "—",
        "reg_date": (str(d.reg_date)[:10] if d.reg_date else "—"),
        "category": str(
            getattr(d, "doc_category_const", None)
            or getattr(d, "doc_category_constant", "—")
        ),
        "short_summary": (d.short_summary or "")[:200],
        "author": author_name,
        "status": str(d.status or "—"),
    }

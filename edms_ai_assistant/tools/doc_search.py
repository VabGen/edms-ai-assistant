# edms_ai_assistant/tools/doc_search.py
"""
EDMS AI Assistant — Document Search Tool.

Поиск документов в EDMS по широкому набору параметров фильтрации.

Маппинг параметров инструмента → поля DocumentFilter (Java):
    short_summary        → shortSummary         (String, like)
    reg_number           → regNumber            (String, like)
    out_reg_number       → outRegNumber         (String, like)
    doc_category         → categoryConstants    (DocumentCategoryConstants[])
    date_from            → dateRegStart         (Instant ISO 8601)
    date_to              → dateRegEnd           (Instant ISO 8601)
    date_control_start   → dateControlStart     (Instant ISO 8601)
    date_control_end     → dateControlEnd       (Instant ISO 8601)
    author_last_name     → authorLastName       (String, like)
    correspondent_name   → correspondentName    (String, like)
    recipient_name       → recipientName        (String, like)
    task_executor_last_name → taskExecutorLastName (String, like)
    author_current_user  → authorCurrentUser    (Boolean)
    process_executor_current_user → processExecutorCurrentUser (Boolean)
    task_executor_current_user    → taskExecutorCurrentUser    (Boolean)
    control_user_current_user     → controlUserCurrentUser     (Boolean)
    introduction_current_user     → introductionCurrentUser    (Boolean)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator

from edms_ai_assistant.clients.document_client import DocumentClient

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

    token: str = Field(..., description="JWT токен авторизации пользователя")

    short_summary: str | None = Field(
        None,
        max_length=500,
        description=(
            "Поиск по краткому содержанию документа. "
            "Пример: 'договор с ООО Альфа', 'обращение по вопросу аренды'."
        ),
    )
    reg_number: str | None = Field(
        None,
        max_length=100,
        description="Регистрационный номер (входящий). Пример: 'ВХ-2026-001'.",
    )
    out_reg_number: str | None = Field(
        None,
        max_length=100,
        description="Исходящий регистрационный номер. Пример: 'ИСХ-2025-042'.",
    )
    doc_category: str | None = Field(
        None,
        description=(
            "Категория документа. Допустимые значения: "
            "INCOMING (входящие), OUTGOING (исходящие), INTERN (внутренние), "
            "APPEAL (обращения граждан), CONTRACT (договоры), "
            "MEETING (совещания), QUESTION (вопросы повестки), "
            "MEETING_QUESTION (вопросы заседания), CUSTOM (произвольные)."
        ),
    )
    status: str | None = Field(
        None,
        description=(
            "Статус документа для фильтрации. Допустимые значения: "
            "FORMING (формирование), "
            "REGISTRATION (на регистрации), "
            "IN_PROGRESS (в работе / на исполнении), "
            "COMPLETED (завершён / исполнен), "
            "CANCELLED (аннулирован), "
            "ARCHIVE (в архиве), "
            "ON_SIGNING (на подписании), "
            "ON_AGREEMENT (на согласовании), "
            "ON_REVIEW (на рассмотрении), "
            "ON_STATEMENT (на утверждении). "
            "ВАЖНО: передавай ОДНУ строку, не список. "
            "Если статус не известен точно — не передавай это поле."
        ),
    )

    date_from: str | None = Field(
        None,
        description="Начало диапазона дат регистрации (включительно). Формат: YYYY-MM-DD.",
        pattern=_DATE_PATTERN,
    )
    date_to: str | None = Field(
        None,
        description="Конец диапазона дат регистрации (включительно). Формат: YYYY-MM-DD.",
        pattern=_DATE_PATTERN,
    )

    date_control_start: str | None = Field(
        None,
        description="Начало диапазона дат постановки на контроль. Формат: YYYY-MM-DD.",
        pattern=_DATE_PATTERN,
    )
    date_control_end: str | None = Field(
        None,
        description="Конец диапазона дат контроля. Формат: YYYY-MM-DD.",
        pattern=_DATE_PATTERN,
    )

    author_last_name: str | None = Field(
        None,
        max_length=150,
        description="Фамилия автора документа. Пример: 'Иванов'.",
    )
    correspondent_name: str | None = Field(
        None,
        max_length=300,
        description="Наименование организации-корреспондента. Пример: 'ООО Альфа'.",
    )
    recipient_name: str | None = Field(
        None,
        max_length=300,
        description="Наименование адресата документа.",
    )
    task_executor_last_name: str | None = Field(
        None,
        max_length=150,
        description="Фамилия исполнителя поручения по документу.",
    )

    author_current_user: bool | None = Field(
        None,
        description="True — только документы, где автор = текущий пользователь.",
    )
    process_executor_current_user: bool | None = Field(
        None,
        description="True — документы, где текущий пользователь участник активного процесса.",
    )
    task_executor_current_user: bool | None = Field(
        None,
        description="True — документы, по которым текущий пользователь является исполнителем поручения.",
    )
    control_user_current_user: bool | None = Field(
        None,
        description="True — документы, где текущий пользователь является контролёром.",
    )
    introduction_current_user: bool | None = Field(
        None,
        description="True — документы на ознакомлении у текущего пользователя.",
    )

    # ── Валидаторы ────────────────────────────────────────────────────────────

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
            raise ValueError(
                f"Неизвестная категория документа: '{upper}'. "
                f"Допустимые значения: {sorted(_VALID_CATEGORIES)}"
            )
        return upper

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: Any) -> str | None:
        """
        Нормализует статус в одну строку.

        Защита от случая когда LLM передаёт список вместо строки:
        ["IN_PROGRESS"] → "IN_PROGRESS"
        "IN_PROGRESS"   → "IN_PROGRESS"
        """
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

        if upper not in _VALID_STATUSES and len(upper) > 2:
            logger.warning(
                "Possibly invalid status value '%s'. Valid: %s",
                upper,
                sorted(_VALID_STATUSES),
            )

        return upper

    @model_validator(mode="after")
    def at_least_one_search_param(self) -> DocSearchInput:
        filter_fields = [
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
        if not any(v is not None for v in filter_fields):
            raise ValueError(
                "Укажите хотя бы один параметр поиска: краткое содержание, "
                "номер документа, категорию, статус, диапазон дат, "
                "участника документа или флаг текущего пользователя."
            )
        return self


def _build_params_list(
        doc_filter: dict[str, Any],
        pageable: dict[str, Any],
        includes: list[str],
) -> list[tuple[str, Any]]:
    """
    Строит список кортежей (key, value) для GET-запроса.

    httpx корректно передаёт повторяющиеся параметры через список кортежей:
    [("categoryConstants", "INCOMING"), ("includes", "DOCUMENT_TYPE"), ...]
    → ?categoryConstants=INCOMING&includes=DOCUMENT_TYPE&...

    Spring @RequestParam List<T> / T[] ожидает именно такой формат.

    Args:
        doc_filter: Словарь фильтров (значения могут быть списками).
        pageable: Параметры пагинации.
        includes: Список includes.

    Returns:
        Список кортежей (key, value).
    """
    params: list[tuple[str, Any]] = []

    for key, val in doc_filter.items():
        if val is None:
            continue
        if isinstance(val, list):
            for item in val:
                params.append((key, str(item)))
        elif isinstance(val, bool):
            params.append((key, str(val).lower()))
        else:
            params.append((key, val))

    for key, val in pageable.items():
        if val is not None:
            params.append((key, val))

    for inc in includes:
        params.append(("includes", inc))

    return params


@tool("doc_search_tool", args_schema=DocSearchInput)
async def doc_search_tool(
        token: str,
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
) -> dict[str, Any]:
    """
    Searches documents in EDMS by a wide range of filter criteria.

    ВАЖНО про статусы: передавай ОДНУ строку (не список!):
    «в работе» → IN_PROGRESS, «на согласовании» → ON_AGREEMENT,
    «завершённые» → COMPLETED, «аннулированные» → CANCELLED.
    Если не уверен в точном статусе — НЕ передавай поле status.
    НЕ убирай и не меняй колонку id

    Use when the user asks:
    - «Найди договоры с ООО Альфа»                → correspondent_name + doc_category=CONTRACT
    - «Покажи входящие документы за март 2026»    → doc_category=INCOMING + date_from/date_to
    - «Есть ли документ ВХ-2026-001?»             → reg_number
    - «Входящие в работе за январь»               → doc_category=INCOMING + status=IN_PROGRESS + dates
    - «Мои документы на согласовании»             → process_executor_current_user=True
    """
    # ── Строим фильтр без status (передаём отдельно для пост-фильтрации) ─────
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
    if author_current_user:
        doc_filter["authorCurrentUser"] = True
    if process_executor_current_user:
        doc_filter["processExecutorCurrentUser"] = True
    if task_executor_current_user:
        doc_filter["taskExecutorCurrentUser"] = True
    if control_user_current_user:
        doc_filter["controlUserCurrentUser"] = True
    if introduction_current_user:
        doc_filter["introductionCurrentUser"] = True

    pageable: dict[str, Any] = {"page": 0, "size": _MAX_RESULTS}
    includes = ["DOCUMENT_TYPE", "CORRESPONDENT", "REGISTRATION_JOURNAL"]

    logger.info(
        "Document search requested",
        extra={"filter_keys": list(doc_filter.keys()), "status_filter": status},
    )

    params_list = _build_params_list(doc_filter, pageable, includes)
    logger.debug("Search params: %s", params_list)

    content: list[dict[str, Any]] = []

    try:
        async with DocumentClient() as client:
            result = await client._make_request(
                "GET",
                "api/document",
                token=token,
                params=params_list,
            )

            if isinstance(result, dict):
                content = result.get("content") or []
            elif isinstance(result, list):
                content = result

    except Exception as exc:
        logger.error("Document search failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": f"Ошибка поиска документов: {exc}",
        }

    # ── Пост-фильтрация по статусу на стороне Python ─────────────────────────
    if status and content:
        status_upper = status.strip().upper()
        filtered = [
            d for d in content
            if str(d.get("status", "")).upper() == status_upper
        ]
        logger.info(
            "Post-filter by status=%s: %d → %d documents",
            status_upper, len(content), len(filtered),
        )
        content = filtered

    if not content:
        hint = ""
        if status:
            hint = f" со статусом «{status}»"
        return {
            "status": "success",
            "message": f"По вашему запросу документы не найдены{hint}.",
            "documents": [],
            "total": 0,
        }

    documents = [_serialize_document(d) for d in content[:10]]

    logger.info("Document search completed, found %d", len(documents))

    return {
        "status": "success",
        "total": len(content),
        "shown": len(documents),
        "documents": documents,
        "message": f"Найдено {len(content)} документ(ов), показано {len(documents)}.",
    }


# ── Вспомогательные функции ───────────────────────────────────────────────────


def _to_iso_start(date_str: str) -> str:
    """Converts YYYY-MM-DD to start-of-day ISO 8601 with milliseconds."""
    dt = datetime.fromisoformat(date_str)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )


def _to_iso_end(date_str: str) -> str:
    """Converts YYYY-MM-DD to end-of-day ISO 8601 with milliseconds."""
    dt = datetime.fromisoformat(date_str)
    return dt.replace(hour=23, minute=59, second=59, microsecond=999000).strftime(
        "%Y-%m-%dT%H:%M:%S.999Z"
    )


def _serialize_document(d: dict[str, Any]) -> dict[str, Any]:
    """Converts raw DocumentDto dict into a compact agent-friendly representation."""
    return {
        "id": str(d.get("id", "")),
        "reg_number": d.get("regNumber") or d.get("reservedRegNumber") or "—",
        "reg_date": _extract_date(d.get("regDate")),
        "category": str(d.get("docCategoryConstant", "—")),
        "short_summary": (d.get("shortSummary") or "")[:200],
        "author": _format_author(d.get("author")),
        "status": str(d.get("status", "—")),
    }


def _extract_date(raw: Any) -> str:
    if not raw:
        return "—"
    return str(raw)[:10]


def _format_author(author: dict[str, Any] | None) -> str:
    if not author:
        return "—"
    parts = [
        author.get("lastName", ""),
        author.get("firstName", ""),
        author.get("middleName", ""),
    ]
    return " ".join(p for p in parts if p).strip() or "—"

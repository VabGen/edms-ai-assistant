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
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field, field_validator, model_validator

from edms_ai_assistant.agent.hitl_primitives import ask_human
from edms_ai_assistant.agent.interrupt_contract import (
    CardSelectInterrupt,
    CardSelectResume,
    InterruptCard,
)
from edms_ai_assistant.agent.runnable_utils import get_token_from_config
from edms_ai_assistant.clients.document_client import DocumentClient

logger = logging.getLogger(__name__)

_MAX_RESULTS: int = 20
_VALID_CATEGORIES: frozenset[str] = frozenset(
    {"INTERN", "INCOMING", "OUTGOING", "MEETING", "QUESTION", "MEETING_QUESTION", "APPEAL", "CONTRACT", "CUSTOM"})
_VALID_STATUSES: frozenset[str] = frozenset(
    {"FORMING", "REGISTRATION", "IN_PROGRESS", "COMPLETED", "CANCELLED", "ARCHIVE", "ON_SIGNING", "ON_AGREEMENT",
     "ON_REVIEW", "ON_STATEMENT"})
_DATE_PATTERN: str = r"^\d{4}-\d{2}-\d{2}$|^$"


class DocSearchInput(BaseModel):
    """Validated input schema for the document search tool."""
    short_summary: str | None = Field(None, max_length=500, description="Поиск по краткому содержанию.")
    reg_number: str | None = Field(None, max_length=100, description="Регистрационный номер.")
    out_reg_number: str | None = Field(None, max_length=100, description="Исходящий регистрационный номер.")
    doc_category: str | None = Field(None, description="Категория документа.")
    status: str | None = Field(None, description="Статус документа. Передавай ОДНУ строку.")
    date_from: str | None = Field(None, description="Начало дат регистрации. Формат: YYYY-MM-DD.",
                                  pattern=_DATE_PATTERN)
    date_to: str | None = Field(None, description="Конец дат регистрации. Формат: YYYY-MM-DD.", pattern=_DATE_PATTERN)
    date_control_start: str | None = Field(None, description="Начало дат контроля. Формат: YYYY-MM-DD.",
                                           pattern=_DATE_PATTERN)
    date_control_end: str | None = Field(None, description="Конец дат контроля. Формат: YYYY-MM-DD.",
                                         pattern=_DATE_PATTERN)
    author_last_name: str | None = Field(None, max_length=150, description="Фамилия автора.")
    correspondent_name: str | None = Field(None, max_length=300, description="Организация-корреспондент.")
    recipient_name: str | None = Field(None, max_length=300, description="Адресат документа.")
    task_executor_last_name: str | None = Field(None, max_length=150, description="Фамилия исполнителя поручения.")
    author_current_user: bool | None = Field(None, description="True — автор = текущий пользователь.")
    process_executor_current_user: bool | None = Field(None, description="True — участник активного процесса.")
    task_executor_current_user: bool | None = Field(None, description="True — исполнитель поручения.")
    control_user_current_user: bool | None = Field(None, description="True — контролёр.")
    introduction_current_user: bool | None = Field(None, description="True — на ознакомлении.")
    display_mode: str | None = Field(
        None,
        description=(
            "Формат вывода результатов. "
            "Установите 'cards', если пользователь просит ПОКАЗАТЬ, НАЙТИ ДЛЯ ПРОСМОТРА или ОТКРЫТЬ документы. "
            "Если параметр опущен или 'data', возвращается JSON для подсчета, статистики или внутреннего анализа."
        ),
    )

    @field_validator("short_summary", "reg_number", "out_reg_number", "author_last_name", "correspondent_name",
                     "recipient_name", "task_executor_last_name", mode="before")
    @classmethod
    def strip_and_none(cls, v: str | None) -> str | None:
        if v is None: return None
        stripped = v.strip()
        return stripped if stripped else None

    @field_validator("doc_category", mode="before")
    @classmethod
    def validate_category(cls, v: str | None) -> str | None:
        if v is None: return None
        upper = v.strip().upper()
        if not upper: return None
        if upper not in _VALID_CATEGORIES: raise ValueError(f"Неизвестная категория: '{upper}'")
        return upper

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: Any) -> str | None:
        if v is None: return None
        if isinstance(v, list):
            if not v: return None
            v = v[0]
        if not isinstance(v, str): v = str(v)
        upper = v.strip().upper()
        if not upper: return None
        return upper

    @model_validator(mode="after")
    def at_least_one_search_param(self) -> DocSearchInput:
        fields = [self.short_summary, self.reg_number, self.out_reg_number, self.doc_category, self.status,
                  self.date_from, self.date_to, self.date_control_start, self.date_control_end, self.author_last_name,
                  self.correspondent_name, self.recipient_name, self.task_executor_last_name, self.author_current_user,
                  self.process_executor_current_user, self.task_executor_current_user, self.control_user_current_user,
                  self.introduction_current_user]
        if not any(v is not None for v in fields): raise ValueError("Укажите хотя бы один параметр поиска.")
        return self


@tool("doc_search_tool", args_schema=DocSearchInput)
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
        logger.error("Failed to get token from config: %s | config keys: %s", e,
                     list((config or {}).get("configurable", {}).keys()) if config else "None")
        return {"status": "error", "message": f"Ошибка авторизации: токен не найден. {e}"}

    doc_filter: dict[str, Any] = {}
    if short_summary: doc_filter["shortSummary"] = short_summary
    if reg_number: doc_filter["regNumber"] = reg_number
    if out_reg_number: doc_filter["outRegNumber"] = out_reg_number
    if doc_category: doc_filter["categoryConstants"] = [doc_category]
    if date_from: doc_filter["dateRegStart"] = _to_iso_start(date_from)
    if date_to: doc_filter["dateRegEnd"] = _to_iso_end(date_to)
    if date_control_start: doc_filter["dateControlStart"] = _to_iso_start(date_control_start)
    if date_control_end: doc_filter["dateControlEnd"] = _to_iso_end(date_control_end)
    if author_last_name: doc_filter["authorLastName"] = author_last_name
    if correspondent_name: doc_filter["correspondentName"] = correspondent_name
    if recipient_name: doc_filter["recipientName"] = recipient_name
    if task_executor_last_name: doc_filter["taskExecutorLastName"] = task_executor_last_name
    if author_current_user: doc_filter["authorCurrentUser"] = True
    if process_executor_current_user: doc_filter["processExecutorCurrentUser"] = True
    if task_executor_current_user: doc_filter["taskExecutorCurrentUser"] = True
    if control_user_current_user: doc_filter["controlUserCurrentUser"] = True
    if introduction_current_user: doc_filter["introductionCurrentUser"] = True

    pageable: dict[str, Any] = {"page": 0, "size": _MAX_RESULTS}
    includes = ["DOCUMENT_TYPE", "CORRESPONDENT", "REGISTRATION_JOURNAL"]

    logger.info("Document search requested",
                extra={"filter_keys": list(doc_filter.keys()), "status_filter": status, "display_mode": display_mode})
    params_list = _build_params_list(doc_filter, pageable, includes)

    content: list[dict[str, Any]] = []
    try:
        async with DocumentClient() as client:
            result = await client._make_request("GET", "api/document", token=token, params=params_list)
            if isinstance(result, dict):
                content = result.get("content") or []
            elif isinstance(result, list):
                content = result
    except Exception as exc:
        logger.error("Document search failed: %s", exc, exc_info=True)
        return {"status": "error", "message": f"Ошибка поиска документов: {exc}"}

    if status and content:
        status_upper = status.strip().upper()
        content = [d for d in content if str(d.get("status", "")).upper() == status_upper]

    if not content:
        return {"status": "success", "message": "Документы не найдены.", "documents": [], "total": 0}

    documents = [_serialize_document(d) for d in content[:10]]

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
                }
            )
            for doc in documents
        ]

        resume = ask_human(CardSelectInterrupt(
            prompt=f"Найдено документов: {len(content)}",
            cards=cards,
            multiple=False,
        ))

        if isinstance(resume, CardSelectResume) and resume.selected_ids:
            selected_id = resume.selected_ids[0]
            return {
                "status": "selected",
                "selected_document_id": selected_id,
                "message": "Пользователь выбрал документ из списка для просмотра."
            }

        # Если пользователь прервал выбор (закрыл виджет и т.п.)
        return {"status": "cancelled", "message": "Пользователь отменил выбор документа."}

    # ── Стандартный возврат JSON для аналитики ──────────────────────
    return {"status": "success", "total": len(content), "shown": len(documents), "documents": documents,
            "message": f"Найдено {len(content)} документ(ов)."}


def _build_params_list(doc_filter: dict[str, Any], pageable: dict[str, Any], includes: list[str]) -> list[
    tuple[str, Any]]:
    params: list[tuple[str, Any]] = []
    for key, val in doc_filter.items():
        if val is None: continue
        if isinstance(val, list):
            for item in val: params.append((key, str(item)))
        elif isinstance(val, bool):
            params.append((key, str(val).lower()))
        else:
            params.append((key, val))
    for key, val in pageable.items():
        if val is not None: params.append((key, val))
    for inc in includes: params.append(("includes", inc))
    return params


def _to_iso_start(date_str: str) -> str:
    return datetime.fromisoformat(date_str).replace(hour=0, minute=0, second=0, microsecond=0).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z")


def _to_iso_end(date_str: str) -> str:
    return datetime.fromisoformat(date_str).replace(hour=23, minute=59, second=59, microsecond=999000).strftime(
        "%Y-%m-%dT%H:%M:%S.999Z")


def _serialize_document(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(d.get("id", "")), "reg_number": d.get("regNumber") or d.get("reservedRegNumber") or "—",
        "reg_date": (str(d.get("regDate"))[:10] if d.get("regDate") else "—"),
        "category": str(d.get("docCategoryConstant", "—")), "short_summary": (d.get("shortSummary") or "")[:200],
        "author": " ".join(p for p in
                           [(d.get("author") or {}).get("lastName", ""), (d.get("author") or {}).get("firstName", ""),
                            (d.get("author") or {}).get("middleName", "")] if p).strip() or "—",
        "status": str(d.get("status", "—")),
    }
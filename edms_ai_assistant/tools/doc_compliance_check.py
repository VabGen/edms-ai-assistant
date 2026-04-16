# edms_ai_assistant/tools/doc_compliance_check.py
"""
EDMS AI Assistant — Document Compliance Check Tool.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.clients.attachment_client import EdmsAttachmentClient
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.services.file_processor import FileProcessorService
from edms_ai_assistant.tools.attachment import (
    _get_attachment_id,
    _get_attachment_name,
    _resolve_attachment,
)

logger = logging.getLogger(__name__)

_MAX_ATTACHMENT_CHARS: int = 12_000
_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".doc", ".txt", ".rtf", ".odt", ".md"}
)

# ══════════════════════════════════════════════════════════════════════════════
# Input schema
# ══════════════════════════════════════════════════════════════════════════════


class DocComplianceCheckInput(BaseModel):
    document_id: str = Field(..., description="UUID документа в СЭД")
    token: str = Field(..., description="JWT токен авторизации")
    attachment_id: str | None = Field(
        None,
        description=(
            "UUID или имя конкретного вложения. "
            "Если не указан и вложений несколько — возвращается disambiguation."
        ),
    )
    check_all: bool = Field(False, description="True — анализировать все вложения.")

    @field_validator("attachment_id")
    @classmethod
    def strip_empty(cls, v: str | None) -> str | None:
        return v.strip() or None if v else None


# ══════════════════════════════════════════════════════════════════════════════
# Поля по категориям
# check_type: "text" — ищем в файле | "presence" — просто факт заполнения
# ══════════════════════════════════════════════════════════════════════════════

FieldDef = dict[str, str]

_FIELDS_BY_CATEGORY: dict[str, list[FieldDef]] = {
    "APPEAL": [
        {
            "field_key": "fioApplicant",
            "label": "ФИО заявителя",
            "source": "appeal",
            "update_field": "fioApplicant",
            "check_type": "text",
        },
        {
            "field_key": "organizationName",
            "label": "Организация заявителя",
            "source": "appeal",
            "update_field": "organizationName",
            "check_type": "text",
        },
        {
            "field_key": "signed",
            "label": "Подписант",
            "source": "appeal",
            "update_field": "signed",
            "check_type": "text",
        },
        {
            "field_key": "correspondentOrgNumber",
            "label": "Исходящий номер организации",
            "source": "appeal",
            "update_field": "correspondentOrgNumber",
            "check_type": "text",
        },
        {
            "field_key": "phone",
            "label": "Телефон",
            "source": "appeal",
            "update_field": "phone",
            "check_type": "text",
        },
        {
            "field_key": "email",
            "label": "Email",
            "source": "appeal",
            "update_field": "email",
            "check_type": "text",
        },
        {
            "field_key": "fullAddress",
            "label": "Адрес заявителя",
            "source": "appeal",
            "update_field": "fullAddress",
            "check_type": "text",
        },
        {
            "field_key": "index",
            "label": "Почтовый индекс",
            "source": "appeal",
            "update_field": "index",
            "check_type": "text",
        },
        {
            "field_key": "shortSummary",
            "label": "Краткое содержание",
            "source": "root",
            "update_field": "shortSummary",
            "check_type": "presence",
        },
        {
            "field_key": "correspondentAppeal",
            "label": "Пересылающая организация",
            "source": "appeal",
            "update_field": "correspondentAppeal",
            "check_type": "text",
        },
        {
            "field_key": "indexDateCoverLetter",
            "label": "Индекс сопроводительного",
            "source": "appeal",
            "update_field": "indexDateCoverLetter",
            "check_type": "text",
        },
        {
            "field_key": "reviewProgress",
            "label": "Ход рассмотрения",
            "source": "appeal",
            "update_field": "reviewProgress",
            "check_type": "text",
        },
    ],
    "INCOMING": [
        {
            "field_key": "outRegNumber",
            "label": "Исходящий номер",
            "source": "root",
            "update_field": "outRegNumber",
            "check_type": "text",
        },
        {
            "field_key": "outRegDate",
            "label": "Исходящая дата",
            "source": "root",
            "update_field": "outRegDate",
            "check_type": "text",
        },
        {
            "field_key": "correspondentName",
            "label": "Корреспондент",
            "source": "root",
            "update_field": "correspondentName",
            "check_type": "text",
        },
        {
            "field_key": "regDate",
            "label": "Дата регистрации",
            "source": "root",
            "update_field": "regDate",
            "check_type": "text",
        },
        {
            "field_key": "regNumber",
            "label": "Регистрационный номер",
            "source": "root",
            "update_field": "regNumber",
            "check_type": "presence",
        },
        {
            "field_key": "shortSummary",
            "label": "Краткое содержание",
            "source": "root",
            "update_field": "shortSummary",
            "check_type": "presence",
        },
        {
            "field_key": "note",
            "label": "Примечание",
            "source": "root",
            "update_field": "note",
            "check_type": "presence",
        },
    ],
    "OUTGOING": [
        {
            "field_key": "regNumber",
            "label": "Регистрационный номер",
            "source": "root",
            "update_field": "regNumber",
            "check_type": "presence",
        },
        {
            "field_key": "regDate",
            "label": "Дата регистрации",
            "source": "root",
            "update_field": "regDate",
            "check_type": "presence",
        },
        {
            "field_key": "correspondentName",
            "label": "Адресат",
            "source": "root",
            "update_field": "correspondentName",
            "check_type": "text",
        },
        {
            "field_key": "shortSummary",
            "label": "Краткое содержание",
            "source": "root",
            "update_field": "shortSummary",
            "check_type": "presence",
        },
        {
            "field_key": "note",
            "label": "Примечание",
            "source": "root",
            "update_field": "note",
            "check_type": "presence",
        },
    ],
    "INTERN": [
        {
            "field_key": "shortSummary",
            "label": "Краткое содержание",
            "source": "root",
            "update_field": "shortSummary",
            "check_type": "presence",
        },
        {
            "field_key": "regDate",
            "label": "Дата документа",
            "source": "root",
            "update_field": "regDate",
            "check_type": "presence",
        },
        {
            "field_key": "note",
            "label": "Примечание",
            "source": "root",
            "update_field": "note",
            "check_type": "presence",
        },
    ],
    "CONTRACT": [
        {
            "field_key": "contractNumber",
            "label": "Номер договора",
            "source": "root",
            "update_field": "contractNumber",
            "check_type": "text",
        },
        {
            "field_key": "contractDate",
            "label": "Дата договора",
            "source": "root",
            "update_field": "contractDate",
            "check_type": "text",
        },
        {
            "field_key": "contractSum",
            "label": "Сумма договора",
            "source": "root",
            "update_field": "contractSum",
            "check_type": "text",
        },
        {
            "field_key": "contractSigningDate",
            "label": "Дата подписания",
            "source": "root",
            "update_field": "contractSigningDate",
            "check_type": "text",
        },
        {
            "field_key": "contractDurationStart",
            "label": "Начало действия",
            "source": "root",
            "update_field": "contractDurationStart",
            "check_type": "text",
        },
        {
            "field_key": "contractDurationEnd",
            "label": "Окончание действия",
            "source": "root",
            "update_field": "contractDurationEnd",
            "check_type": "text",
        },
        {
            "field_key": "correspondentName",
            "label": "Контрагент",
            "source": "root",
            "update_field": "correspondentName",
            "check_type": "text",
        },
        {
            "field_key": "shortSummary",
            "label": "Краткое содержание",
            "source": "root",
            "update_field": "shortSummary",
            "check_type": "presence",
        },
        {
            "field_key": "note",
            "label": "Примечание",
            "source": "root",
            "update_field": "note",
            "check_type": "presence",
        },
    ],
    "MEETING": [
        {
            "field_key": "dateMeeting",
            "label": "Дата совещания",
            "source": "root",
            "update_field": "dateMeeting",
            "check_type": "text",
        },
        {
            "field_key": "startMeeting",
            "label": "Время начала",
            "source": "root",
            "update_field": "startMeeting",
            "check_type": "text",
        },
        {
            "field_key": "endMeeting",
            "label": "Время окончания",
            "source": "root",
            "update_field": "endMeeting",
            "check_type": "text",
        },
        {
            "field_key": "placeMeeting",
            "label": "Место проведения",
            "source": "root",
            "update_field": "placeMeeting",
            "check_type": "text",
        },
        {
            "field_key": "externalInvitees",
            "label": "Внешние приглашённые",
            "source": "root",
            "update_field": "externalInvitees",
            "check_type": "presence",
        },
        {
            "field_key": "shortSummary",
            "label": "Краткое содержание",
            "source": "root",
            "update_field": "shortSummary",
            "check_type": "presence",
        },
    ],
    "MEETING_QUESTION": [
        {
            "field_key": "dateMeetingQuestion",
            "label": "Дата заседания",
            "source": "root",
            "update_field": "dateMeetingQuestion",
            "check_type": "text",
        },
        {
            "field_key": "numberQuestion",
            "label": "Номер вопроса",
            "source": "root",
            "update_field": "numberQuestion",
            "check_type": "presence",
        },
        {
            "field_key": "shortSummary",
            "label": "Краткое содержание",
            "source": "root",
            "update_field": "shortSummary",
            "check_type": "presence",
        },
        {
            "field_key": "commentQuestion",
            "label": "Комментарий",
            "source": "root",
            "update_field": "commentQuestion",
            "check_type": "presence",
        },
    ],
    "QUESTION": [
        {
            "field_key": "dateQuestion",
            "label": "Дата вопроса",
            "source": "root",
            "update_field": "dateQuestion",
            "check_type": "text",
        },
        {
            "field_key": "shortSummary",
            "label": "Краткое содержание",
            "source": "root",
            "update_field": "shortSummary",
            "check_type": "presence",
        },
    ],
    "CUSTOM": [
        {
            "field_key": "shortSummary",
            "label": "Краткое содержание",
            "source": "root",
            "update_field": "shortSummary",
            "check_type": "presence",
        },
        {
            "field_key": "regDate",
            "label": "Дата документа",
            "source": "root",
            "update_field": "regDate",
            "check_type": "presence",
        },
    ],
}

_FIELDS_DEFAULT: list[FieldDef] = [
    {
        "field_key": "shortSummary",
        "label": "Краткое содержание",
        "source": "root",
        "update_field": "shortSummary",
        "check_type": "presence",
    },
]


def _get_field_defs(category: str) -> list[FieldDef]:
    return _FIELDS_BY_CATEGORY.get(category.upper(), _FIELDS_DEFAULT)


# ══════════════════════════════════════════════════════════════════════════════
# Извлечение значений из DocumentDto
# ══════════════════════════════════════════════════════════════════════════════


def _format_value(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    if hasattr(value, "strftime"):
        return value.strftime("%d.%m.%Y")
    if hasattr(value, "isoformat"):
        return value.isoformat()[:10]
    s = str(value).strip()
    return s or None


def _extract_card_fields(
    doc: DocumentDto,
    category: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Возвращает (text_fields, presence_fields).
    text_fields     — поля для LLM-проверки по тексту файла.
    presence_fields — поля которые просто заполнены (факт наличия → ok).
    """
    appeal = getattr(doc, "documentAppeal", None)
    field_defs = _get_field_defs(category)

    text_fields: list[dict[str, Any]] = []
    presence_fields: list[dict[str, Any]] = []

    for fd in field_defs:
        fk = fd["field_key"]
        src = fd["source"]

        raw = None
        if src == "appeal" and appeal is not None:
            raw = getattr(appeal, fk, None)
            if raw is None:
                raw = getattr(doc, fk, None)
        else:
            raw = getattr(doc, fk, None)
            if hasattr(raw, "value"):
                raw = raw.value

        formatted = _format_value(raw)
        if not formatted:
            continue

        entry = {
            "field_key": fk,
            "label": fd["label"],
            "card_value": formatted,
            "update_field": fd["update_field"],
        }

        if fd["check_type"] == "presence":
            presence_fields.append(entry)
        else:
            text_fields.append(entry)

    return text_fields, presence_fields


# ══════════════════════════════════════════════════════════════════════════════
# Скачивание + извлечение текста
# ══════════════════════════════════════════════════════════════════════════════


async def _extract_text(
    token: str,
    document_id: str,
    attachment_id: str,
    attachment_name: str,
) -> str | None:
    suffix = Path(attachment_name).suffix.lower() or ".tmp"
    if suffix not in _SUPPORTED_EXTENSIONS:
        return None

    try:
        async with EdmsAttachmentClient() as client:
            content_bytes = await client.get_attachment_content(
                token, document_id, attachment_id
            )
    except Exception as exc:
        logger.error("Download failed for '%s': %s", attachment_name, exc)
        return None

    if not content_bytes:
        return None

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content_bytes)
            tmp_path = tmp.name
        text = await FileProcessorService.extract_text_async(tmp_path)
        if not text or text.startswith(("Ошибка:", "Формат файла")):
            return None
        return text[:_MAX_ATTACHMENT_CHARS]
    except Exception as exc:
        logger.error("Extraction failed for '%s': %s", attachment_name, exc)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# LLM проверка text-полей
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """
Ты — эксперт по документообороту. Проверяешь соответствие полей карточки документа
содержимому вложенного файла.

ПРАВИЛА:
1. Проверяй ТОЛЬКО переданные поля карточки.
2. Статусы:
   "ok"        — значения совпадают (форматы дат/чисел могут отличаться)
   "mismatch"  — явное расхождение; укажи correct_value из файла
   "not_found" — поле не упоминается в файле
3. Для дат: "15.03.2026" = "2026-03-15" = "15 марта 2026" — это ok
4. Для чисел: "1 000 000" = "1000000" — это ok
5. Для телефонов сравнивай цифры: "+375 29 000-00-01" != "+375 29 000-00-00"
6. Для ФИО сравнивай полностью — "Иванов Иван" != "Иванов Иван Иванович"

Верни ТОЛЬКО JSON без markdown-блоков и без лишних ключей:
{{"fields": [{{"field_key": "...", "label": "...", "card_value": "...", "correct_value": null, "status": "ok", "update_field": "...", "recommendation": null}}]}}
"""

_USER_PROMPT = """
Категория: {category}

Поля карточки:
{card_fields_text}

Текст вложения «{attachment_name}»:
---
{attachment_text}
---
"""


async def _run_llm(
    category: str,
    text_fields: list[dict[str, Any]],
    attachment_text: str,
    attachment_name: str,
) -> list[dict[str, Any]]:
    """LLM-проверка text-полей. Возвращает список field-объектов."""
    if not text_fields:
        return []

    card_fields_text = "\n".join(
        f"- {f['label']} ({f['field_key']}): {f['card_value']}" for f in text_fields
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", _USER_PROMPT),
        ]
    )

    llm = get_chat_model()
    try:
        llm_clean = llm.bind_tools([])
    except Exception:
        llm_clean = llm

    parser = JsonOutputParser()
    chain = prompt | llm_clean | parser

    try:
        result = await chain.ainvoke(
            {
                "category": category,
                "card_fields_text": card_fields_text,
                "attachment_text": attachment_text,
                "attachment_name": attachment_name,
            }
        )

        meta = {f["field_key"]: f for f in text_fields}
        fields_out: list[dict[str, Any]] = []
        for f in result.get("fields", []):
            key = f.get("field_key", "")
            if key in meta:
                f["update_field"] = meta[key]["update_field"]
                if not f.get("card_value"):
                    f["card_value"] = meta[key]["card_value"]
            fields_out.append(f)
        return fields_out

    except Exception as exc:
        logger.error("LLM check failed: %s", exc, exc_info=True)
        return [
            {
                "field_key": f["field_key"],
                "label": f["label"],
                "card_value": f["card_value"],
                "correct_value": None,
                "status": "not_found",
                "update_field": f["update_field"],
                "recommendation": "Не удалось проверить (ошибка анализа)",
            }
            for f in text_fields
        ]


def _build_presence_results(
    presence_fields: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Presence-поля всегда ok — они заполнены оператором, не ищем в файле."""
    return [
        {
            "field_key": f["field_key"],
            "label": f["label"],
            "card_value": f["card_value"],
            "correct_value": None,
            "status": "ok",
            "update_field": f["update_field"],
            "recommendation": None,
        }
        for f in presence_fields
    ]


def _compute_overall(fields: list[dict[str, Any]]) -> str:
    statuses = {f.get("status") for f in fields}
    if "mismatch" in statuses:
        return "has_mismatches"
    if "not_found" in statuses:
        return "cannot_verify"
    return "ok"


def _compute_summary(overall: str, names: list[str], stats: dict[str, int]) -> str:
    names_str = ", ".join(f"«{n}»" for n in names[:3])
    parts = [f"Проверено: {names_str}."]
    if stats["mismatches"]:
        parts.append(f"Расхождений: {stats['mismatches']}.")
    if stats["not_found"]:
        parts.append(f"Не найдено в файле: {stats['not_found']}.")
    if overall == "ok":
        parts.append("Все поля совпадают.")
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Tool
# ══════════════════════════════════════════════════════════════════════════════


@tool("doc_compliance_check", args_schema=DocComplianceCheckInput)
async def doc_compliance_check(
    document_id: str,
    token: str,
    attachment_id: str | None = None,
    check_all: bool = False,
) -> dict[str, Any]:
    """
    Проверяет соответствие заполненных полей карточки документа содержимому вложения.

    Используй когда пользователь просит:
    - «Проверить документ перед отправкой»
    - «Всё ли правильно заполнено»
    - «Соответствует ли карточка файлу»

    Возвращает поля со статусами:
    - "ok"        — совпадает
    - "mismatch"  — расхождение (содержит correct_value для автоисправления)
    - "not_found" — не найдено в файле

    ВАЖНО: после получения результата сразу формулируй ответ пользователю.
    НЕ вызывай этот инструмент повторно — данные уже получены.
    """
    logger.info(
        "doc_compliance_check: doc=%s... att=%s check_all=%s",
        document_id[:8],
        attachment_id,
        check_all,
    )

    # ── 1. Документ ───────────────────────────────────────────────────────────
    try:
        async with DocumentClient() as client:
            raw = await client.get_document_metadata(token, document_id)
            doc = DocumentDto.model_validate(raw)
    except Exception as exc:
        return {"status": "error", "message": f"Не удалось получить документ: {exc}"}

    cat_raw = getattr(doc, "docCategoryConstant", None)
    category = (
        cat_raw.value if hasattr(cat_raw, "value") else str(cat_raw or "CUSTOM")
    ).upper()

    attachments = list(getattr(doc, "attachmentDocument", None) or [])
    if not attachments:
        return {"status": "error", "message": "В документе нет вложений для проверки."}

    # ── 2. Выбор вложений ─────────────────────────────────────────────────────
    if attachment_id:
        target = _resolve_attachment(attachments, attachment_id)
        if target is None:
            available = [
                {"id": _get_attachment_id(a), "name": _get_attachment_name(a)}
                for a in attachments
                if _get_attachment_id(a)
            ]
            return {
                "status": "requires_disambiguation",
                "message": f"Вложение «{attachment_id}» не найдено. Выберите из списка:",
                "available_attachments": available,
            }
        selected = [target]

    elif check_all or len(attachments) == 1:
        selected = list(attachments)

    else:
        available = [
            {
                "id": _get_attachment_id(a),
                "name": _get_attachment_name(a) or "без имени",
                "size_kb": round((getattr(a, "size", 0) or 0) / 1024, 1),
            }
            for a in attachments
            if _get_attachment_id(a)
        ]
        return {
            "status": "requires_disambiguation",
            "message": "В документе несколько вложений. Выберите одно или проверьте все:",
            "available_attachments": available,
            "check_all_option": True,
        }

    # ── 3. Поля карточки ──────────────────────────────────────────────────────
    text_fields, presence_fields = _extract_card_fields(doc, category)

    if not text_fields and not presence_fields:
        return {
            "status": "error",
            "message": f"Нет заполненных полей для проверки (категория {category}).",
        }

    presence_results = _build_presence_results(presence_fields)

    # ── 4. Если нет text-полей — возвращаем только presence ───────────────────
    if not text_fields:
        fields = presence_results
        fields.sort(
            key=lambda f: {"mismatch": 0, "not_found": 1, "ok": 2}.get(f["status"], 3)
        )
        overall = _compute_overall(fields)
        stats = {
            "total": len(fields),
            "ok": sum(1 for f in fields if f["status"] == "ok"),
            "mismatches": 0,
            "not_found": 0,
        }
        used_names = [_get_attachment_name(selected[0])] if selected else ["—"]
        return {
            "status": "success",
            "document_id": document_id,
            "document_category": category,
            "attachments_checked": used_names,
            "overall": overall,
            "summary": _compute_summary(overall, used_names, stats),
            "fields": fields,
            "stats": stats,
        }

    # ── 5. Скачиваем текст ────────────────────────────────────────────────────
    async def _process_one(att: Any) -> tuple[str, str | None]:
        att_id = _get_attachment_id(att)
        att_name = _get_attachment_name(att) or "attachment"
        att_doc_id = str(getattr(att, "documentId", None) or document_id)
        text = await _extract_text(token, att_doc_id, att_id, att_name)
        return att_name, text

    gathered = await asyncio.gather(
        *[_process_one(a) for a in selected], return_exceptions=True
    )

    texts: list[tuple[str, str]] = []
    for item in gathered:
        if isinstance(item, Exception):
            logger.error("Attachment error: %s", item)
            continue
        name, text = item
        if text:
            texts.append((name, text))

    if not texts:
        fallback_text = [
            {
                "field_key": f["field_key"],
                "label": f["label"],
                "card_value": f["card_value"],
                "correct_value": None,
                "status": "not_found",
                "update_field": f["update_field"],
                "recommendation": "Не удалось извлечь текст из вложения",
            }
            for f in text_fields
        ]
        fields = fallback_text + presence_results
        fields.sort(
            key=lambda f: {"mismatch": 0, "not_found": 1, "ok": 2}.get(f["status"], 3)
        )
        overall = _compute_overall(fields)
        stats = {
            "total": len(fields),
            "ok": sum(1 for f in fields if f["status"] == "ok"),
            "mismatches": 0,
            "not_found": sum(1 for f in fields if f["status"] == "not_found"),
        }
        unsupported = [_get_attachment_name(a) for a in selected]
        return {
            "status": "success",
            "document_id": document_id,
            "document_category": category,
            "attachments_checked": unsupported,
            "overall": overall,
            "summary": f"Не удалось извлечь текст из: {', '.join(unsupported)}. Поддерживаемые форматы: PDF, DOCX, DOC, TXT.",
            "fields": fields,
            "stats": stats,
        }

    # ── 6. LLM анализ для каждого вложения ────────────────────────────────────
    llm_tasks = [_run_llm(category, text_fields, text, name) for name, text in texts]
    llm_raw = await asyncio.gather(*llm_tasks, return_exceptions=True)

    all_llm_fields: list[dict[str, Any]] = []
    used_names: list[str] = []

    for (name, _), result in zip(texts, llm_raw):
        if isinstance(result, Exception):
            logger.error("LLM failed for '%s': %s", name, result)
            all_llm_fields.extend(
                [
                    {
                        **f,
                        "status": "not_found",
                        "correct_value": None,
                        "recommendation": "Ошибка анализа",
                    }
                    for f in text_fields
                ]
            )
        else:
            all_llm_fields.extend(result)
        used_names.append(name)

    # ── 7. Агрегация если несколько вложений ──────────────────────────────────
    if len(texts) > 1:
        _priority = {"mismatch": 2, "not_found": 1, "ok": 0}
        agg: dict[str, dict[str, Any]] = {}
        for f in all_llm_fields:
            key = f.get("field_key", "")
            if not key:
                continue
            existing = agg.get(key)
            if existing is None:
                agg[key] = dict(f)
            else:
                if _priority.get(f["status"], 0) > _priority.get(existing["status"], 0):
                    agg[key].update(
                        {
                            "status": f["status"],
                            "correct_value": f.get("correct_value"),
                            "recommendation": f.get("recommendation"),
                        }
                    )
                elif f["status"] == "ok" and existing["status"] == "not_found":
                    agg[key].update(
                        {
                            "status": "ok",
                            "correct_value": None,
                            "recommendation": None,
                        }
                    )
        merged_llm = list(agg.values())
    else:
        merged_llm = all_llm_fields

    # ── 8. Объединяем LLM + presence ──────────────────────────────────────────
    fields = merged_llm + presence_results
    fields.sort(
        key=lambda f: {"mismatch": 0, "not_found": 1, "ok": 2}.get(
            f.get("status", "ok"), 3
        )
    )

    overall = _compute_overall(fields)
    stats = {
        "total": len(fields),
        "ok": sum(1 for f in fields if f["status"] == "ok"),
        "mismatches": sum(1 for f in fields if f["status"] == "mismatch"),
        "not_found": sum(1 for f in fields if f["status"] == "not_found"),
    }

    logger.info("Compliance done: overall=%s %s", overall, stats)

    return {
        "status": "success",
        "document_id": document_id,
        "document_category": category,
        "attachments_checked": used_names,
        "overall": overall,
        "summary": _compute_summary(overall, used_names, stats),
        "fields": fields,
        "stats": stats,
        "fix_hint": (
            (
                "Нажми на карточку расхождения для автоисправления "
                "или используй кнопку «Исправить все»."
            )
            if overall == "has_mismatches"
            else None
        ),
    }

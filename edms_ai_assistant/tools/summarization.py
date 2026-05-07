"""
EDMS AI Assistant — Document Summarisation Tool (v2 Integration).
"""

from __future__ import annotations

import json
import logging
import re as _re
import uuid
from enum import StrEnum
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class SummarizeType(StrEnum):
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    THESIS = "thesis"


_SUMMARY_TYPE_ALIASES: dict[str, str] = {
    "факты": "extractive",
    "ключевые факты": "extractive",
    "extractive": "extractive",
    "1": "extractive",
    "пересказ": "abstractive",
    "краткий пересказ": "abstractive",
    "abstractive": "abstractive",
    "2": "abstractive",
    "тезисы": "thesis",
    "тезисный план": "thesis",
    "thesis": "thesis",
    "3": "thesis",
}


def _normalise_summary_type(value: Any) -> SummarizeType:
    if isinstance(value, SummarizeType):
        return value
    raw = str(value).strip().lower() if value else ""
    canonical = _SUMMARY_TYPE_ALIASES.get(raw, raw)
    try:
        return SummarizeType(canonical)
    except ValueError:
        logger.warning("Unknown summary_type '%s' — falling back to extractive", value)
        return SummarizeType.EXTRACTIVE


class SummarizeInput(BaseModel):
    text: str = Field(
        ...,
        description=(
            "Текст документа для суммаризации "
            "(или JSON-обёртка от doc_get_file_content)"
        ),
        min_length=1,
        max_length=50_000,
    )
    summary_type: SummarizeType | None = Field(
        None,
        description=(
            "Формат суммаризации: "
            "extractive (ключевые факты и данные), "
            "abstractive (краткий пересказ своими словами), "
            "thesis (структурированный тезисный план). "
            "Если None — пользователь ОБЯЗАН выбрать формат (Human-in-the-Loop)."
        ),
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Текст документа не может быть пустым.")
        return stripped


def _unwrap_json_envelope(text: str) -> str:
    clean = text.strip()
    if not (clean.startswith("{") and clean.endswith("}")):
        return clean
    try:
        data: dict[str, Any] = json.loads(clean)
        for key in ("content", "text", "document_info", "text_preview"):
            extracted = data.get(key)
            if extracted and isinstance(extracted, str) and len(extracted) > 10:
                return extracted.strip()
    except (json.JSONDecodeError, TypeError):
        pass
    return clean


def _heuristic_recommendation(text: str) -> dict[str, str]:
    if not text:
        return {
            "recommended": "abstractive",
            "reason": "Текст пуст или очень короткий.",
        }
    chars = len(text)
    lines = text.count("\n")
    numeric_groups = len(_re.findall(r"\d+", text))
    if chars > 5_000 or numeric_groups > 20:
        return {
            "recommended": "thesis",
            "reason": (
                f"Объёмный документ ({chars} симв.) или много числовых данных "
                f"({numeric_groups} чисел) — тезисный план удобнее для навигации."
            ),
        }
    if lines < 5:
        return {
            "recommended": "abstractive",
            "reason": f"Компактный текст ({lines} строк) — краткого пересказа достаточно.",
        }
    return {
        "recommended": "extractive",
        "reason": "Структурированный текст с конкретными данными — список фактов будет полезнее.",
    }


def _build_requires_choice_response(text: str) -> dict[str, Any]:
    hint = _heuristic_recommendation(text)
    return {
        "status": "requires_choice",
        "message": "Выберите формат анализа документа:",
        "options": [
            {
                "key": "extractive",
                "label": "Ключевые факты",
                "description": "Конкретные данные, даты, суммы, имена — нумерованным списком.",
            },
            {
                "key": "abstractive",
                "label": "Краткий пересказ",
                "description": "Суть документа своими словами в 1–2 абзацах.",
            },
            {
                "key": "thesis",
                "label": "Тезисный план",
                "description": "Структурированный план с разделами и подпунктами.",
            },
        ],
        "hint": hint["recommended"],
        "hint_reason": hint["reason"],
    }


@tool("doc_summarize_text", args_schema=SummarizeInput)
async def doc_summarize_text(
    text: str,
    summary_type: SummarizeType | None = None,
) -> dict[str, Any]:
    """Use this tool ONLY when the user EXPLICITLY asks to summarize, analyze, extract facts, or make a thesis plan of a document.
    Do NOT use this tool for simple questions about the document (like 'who is the author?', 'what is the date?', 'find a specific word').
    For simple queries, answer directly from the text.

    Perform intelligent summarisation of document text via LLM v2 pipeline.

    Human-in-the-Loop contract:
        When ``summary_type`` is None, ALWAYS call this tool immediately —
        do NOT ask the user for the format in your text response.
        The tool returns ``requires_choice`` and the system automatically
        presents interactive selection buttons to the user.

    Supported formats when ``summary_type`` is provided:
    - ``extractive`` : Key facts, dates, amounts as a numbered list.
    - ``abstractive``: Concise 1–2 paragraph plain-language retelling.
    - ``thesis``     : Numbered section-by-section thesis plan.

    Args:
        text: Document text (plain or JSON-wrapped).
        summary_type: Desired format or None to trigger user selection.

    Returns:
        Dict with one of three statuses:
        - ``requires_choice``: summary_type was None.
        - ``success``: Analysis completed. Contains `content` and `meta`.
        - ``error``: Validation or LLM call failed.
    """
    logger.info(
        "doc_summarize_text called",
        extra={
            "text_length": len(text),
            "summary_type": summary_type.value if summary_type else None,
        },
    )

    clean_text = _unwrap_json_envelope(text)

    # ── Human-in-the-Loop selection ────────────────────────────────────
    if summary_type is None:
        return _build_requires_choice_response(clean_text)

    normalised = _normalise_summary_type(summary_type)

    # ── Lazy imports to prevent circular dependency ────────────────────
    try:
        from edms_ai_assistant.main import app
        from edms_ai_assistant.summarizer.service import (
            SummarizationRequest,
            SummarizationService,
        )
        from edms_ai_assistant.summarizer.structured.models import SummaryMode
    except ImportError as imp_err:
        logger.error("Failed to import SummarizationService: %s", imp_err)
        return {
            "status": "error",
            "message": "Внутренняя ошибка: модуль суммаризации недоступен.",
        }

    # ── Retrieve service from app state ────────────────────────────────
    service: SummarizationService | None = getattr(
        app.state, "summarization_service", None
    )
    if service is None:
        return {
            "status": "error",
            "message": "Сервис суммаризации не инициализирован. Попробуйте позже.",
        }

    # ── Map tool enums to service enums ────────────────────────────────
    _mode_map = {
        SummarizeType.EXTRACTIVE: SummaryMode.EXTRACTIVE,
        SummarizeType.ABSTRACTIVE: SummaryMode.ABSTRACTIVE,
        SummarizeType.THESIS: SummaryMode.THESIS,
    }
    mode = _mode_map.get(normalised, SummaryMode.ABSTRACTIVE)

    # ── Execute v2 pipeline ────────────────────────────────────────────
    try:
        # Service expects bytes, so we encode the extracted text
        file_bytes = clean_text.encode("utf-8")

        req = SummarizationRequest(
            file_content=file_bytes,
            file_name="agent_tool_input.txt",
            mode=mode,
            language="ru",
            request_id=str(uuid.uuid4()),
            force_refresh=False,
        )

        resp = await service.summarize(req)

        # The output is a dict (serialized Pydantic model).
        # Convert it to JSON string so the LLM agent can read it natively.
        content_str = json.dumps(resp.output, ensure_ascii=False, indent=2)

        return {
            "status": "success",
            "content": content_str,
            "meta": {
                "format_used": resp.mode.value,
                "text_length": len(clean_text),
                "pipeline": resp.chunking_strategy,
                "chunks_processed": resp.chunk_count,
                "processing_time_ms": resp.latency_ms,
                "cost_usd": resp.cost_usd,
                "from_cache": resp.cache_hit,
                "model": resp.model,
            },
        }

    except ValueError as exc:
        logger.warning("Validation error in doc_summarize_text: %s", exc)
        return {"status": "error", "message": f"Ошибка валидации: {exc}"}
    except Exception as exc:
        logger.error("doc_summarize_text failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": f"Не удалось проанализировать документ: {exc}",
        }
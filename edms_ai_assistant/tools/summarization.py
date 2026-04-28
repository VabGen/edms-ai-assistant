# edms_ai_assistant/tools/summarization.py
"""EDMS AI Assistant — Document Summarisation LangChain Tool.

Human-in-the-Loop contract
--------------------------
When ``summary_type`` is None this tool returns a ``requires_choice`` payload
with three labelled options and a heuristic recommendation.
The agent MUST present the selection to the user and wait for their explicit
choice before invoking the tool again with a non-None ``summary_type``.
"""

from __future__ import annotations

import json
import logging
import re as _re
from enum import StrEnum
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.services.summarization_orchestrator import get_orchestrator

logger = logging.getLogger(__name__)


# ── Summary type enum + alias resolution ──────────────────────────────────────


class SummarizeType(StrEnum):
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    THESIS = "thesis"


# Aliases support Russian colloquial names the agent may emit
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
    """Resolve alias → canonical SummarizeType, falling back to EXTRACTIVE.

    Args:
        value: Raw summary_type value from the agent or user.

    Returns:
        Canonical SummarizeType enum member.
    """
    if isinstance(value, SummarizeType):
        return value
    raw = str(value).strip().lower() if value else ""
    canonical = _SUMMARY_TYPE_ALIASES.get(raw, raw)
    try:
        return SummarizeType(canonical)
    except ValueError:
        logger.warning("Unknown summary_type '%s' — falling back to extractive", value)
        return SummarizeType.EXTRACTIVE


# ── Input schema ───────────────────────────────────────────────────────────────


class SummarizeInput(BaseModel):
    """Validated input for the doc_summarize_text tool."""

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


# ── JSON envelope unwrapper ───────────────────────────────────────────────────


def _unwrap_json_envelope(text: str) -> str:
    """Extract plain text from a JSON envelope returned by doc_get_file_content.

    Tries known payload keys in order; returns original string if not a valid
    JSON object or no recognised key found.

    Args:
        text: Raw text that may be a JSON-serialised envelope.

    Returns:
        Unwrapped plain text, or the original string.
    """
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


# ── Heuristic recommendation ──────────────────────────────────────────────────


def _heuristic_recommendation(text: str) -> dict[str, str]:
    """Suggest a summary type based on text structure heuristics.

    Rules (in priority order):
        - Long or data-heavy  → thesis
        - Short / few lines   → abstractive
        - Otherwise           → extractive

    Args:
        text: Clean document text (after envelope unwrapping).

    Returns:
        Dict with ``recommended`` key and human-readable ``reason``.
    """
    if not text:
        return {"recommended": "abstractive", "reason": "Текст пуст или очень короткий."}

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
    """Build the Human-in-the-Loop selection payload.

    Args:
        text: Clean document text for heuristic analysis.

    Returns:
        Dict with status='requires_choice', options list, hint and hint_reason.
    """
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


# ── Tool implementation ────────────────────────────────────────────────────────


@tool("doc_summarize_text", args_schema=SummarizeInput)
async def doc_summarize_text(
    text: str,
    summary_type: SummarizeType | None = None,
) -> dict[str, Any]:
    """Perform intelligent summarisation of document text via LLM.

    Human-in-the-Loop contract:
        When ``summary_type`` is None this tool returns ``requires_choice``
        with three labelled options and a heuristic recommendation.
        The agent MUST show this selection to the user and wait for their
        explicit choice before calling the tool again.

    Supported formats when ``summary_type`` is provided:
        - ``extractive``  : Key facts, dates, amounts as a numbered list.
        - ``abstractive`` : Concise 1–2 paragraph plain-language retelling.
        - ``thesis``      : Numbered section-by-section thesis plan.

    Internal pipeline:
        1. Unwrap JSON envelope if text arrived from doc_get_file_content.
        2. Normalise summary_type aliases (e.g. «тезисы» → thesis).
        3. If summary_type is None → return requires_choice (user must choose).
        4. Invoke orchestrator.summarize() and return structured result.

    Args:
        text: Document text (plain or JSON-wrapped).
        summary_type: Desired format or None to trigger user selection.

    Returns:
        Dict with one of three top-level statuses:
        - ``requires_choice``: summary_type was None.
        - ``success``: Analysis completed. Contains ``content`` and ``meta``.
        - ``error``: Validation or LLM failure. Contains ``message``.
    """
    logger.info(
        "doc_summarize_text invoked",
        extra={
            "text_length": len(text),
            "summary_type": summary_type.value if summary_type else None,
        },
    )

    clean_text = _unwrap_json_envelope(text)

    if summary_type is None:
        return _build_requires_choice_response(clean_text)

    normalised = _normalise_summary_type(summary_type)

    try:
        orchestrator = get_orchestrator()
        result = await orchestrator.summarize(
            text=clean_text,
            summary_type=str(normalised),
            file_identifier=None,  # tool layer never owns file_identifier
        )

        if result.status == "error":
            return {"status": "error", "message": result.content}

        return {
            "status": result.status,
            "content": result.content,
            "meta": {
                "format_used": result.format_used,
                "text_length": result.text_length,
                "was_truncated": result.was_truncated,
                "pipeline": result.pipeline,
                "chunks_processed": result.chunks_processed,
                "processing_time_ms": result.processing_time_ms,
                "quality_score": result.quality_score,
                "confidence": result.confidence,
                "degraded": result.degraded,
            },
        }

    except ValueError as exc:
        logger.warning("Validation error in doc_summarize_text: %s", exc)
        return {"status": "error", "message": f"Ошибка валидации: {exc}"}
    except Exception as exc:
        logger.error("doc_summarize_text failed", exc_info=True)
        return {
            "status": "error",
            "message": f"Не удалось проанализировать документ: {exc}",
        }
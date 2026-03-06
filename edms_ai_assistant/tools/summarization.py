from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import Any, Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.llm import get_chat_model

logger = logging.getLogger(__name__)

_CHOICE_NORM: dict[str, str] = {
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

_MAX_TEXT_LENGTH = 12000
_HEAD_FRACTION = 0.67


class SummarizeType(StrEnum):
    """Supported document summarisation formats."""

    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    THESIS = "thesis"


class SummarizeInput(BaseModel):
    """Validated input schema for the doc_summarize_text tool.

    Attributes:
        text: Document text to summarise (1 – 50 000 chars).
        summary_type: Optional format; if None the tool asks the user to choose.
    """

    text: str = Field(
        ...,
        description="Текст документа для суммаризации",
        min_length=1,
        max_length=50000,
    )
    summary_type: Optional[SummarizeType] = Field(
        None,
        description=(
            "Формат суммаризации: extractive (ключевые факты), "
            "abstractive (краткий пересказ), thesis (тезисный план). "
            "Оставь None только если пользователь действительно не указал формат — "
            "система спросит у него сама."
        ),
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Strip and validate text.

        Args:
            v: Raw text value.

        Returns:
            Stripped text.

        Raises:
            ValueError: If text is blank.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("Текст не может быть пустым")
        return stripped


@tool("doc_summarize_text", args_schema=SummarizeInput)
async def doc_summarize_text(
    text: str,
    summary_type: Optional[SummarizeType] = None,
) -> Dict[str, Any]:
    """Perform intelligent summarisation of document text using the LLM.

    Supported formats:
    - extractive : Key facts, dates, amounts as a structured list.
    - abstractive: Concise 1-2 paragraph plain-language summary.
    - thesis     : Numbered thesis plan with sub-items.

    Workflow:
    1. Clean JSON wrappers if text arrived wrapped in a JSON envelope.
    2. Normalise summary_type aliases (e.g. «факты» → extractive).
    3. If summary_type is still None → return requires_choice so the UI
       can show the selection dropdown to the user.
    4. If summary_type is provided → execute and return the summary.

    Args:
        text: Document text (may be JSON-wrapped content from doc_get_file_content).
        summary_type: Desired summarisation format or None.

    Returns:
        Dict with keys:
        - status  : «success» | «requires_choice» | «error»
        - content : Result string (for success).
        - message : Human-readable message.
        - suggestion: Recommended format hint (for requires_choice).
        - meta    : Processing metadata.
    """
    logger.info(
        "Summarization requested",
        extra={
            "text_length": len(text),
            "summary_type": summary_type.value if summary_type else None,
        },
    )

    try:
        clean_text = _extract_text_from_json(text)

        if summary_type is None:
            return _handle_format_selection(clean_text)

        normalised = _normalise_summary_type(summary_type)
        return await _perform_summarization(clean_text, normalised)

    except ValueError as exc:
        logger.warning("Validation error in summarization: %s", exc)
        return {"status": "error", "message": f"Ошибка валидации: {exc}"}
    except Exception as exc:
        logger.error("Summarization failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": f"Не удалось проанализировать текст: {exc}",
        }


def _normalise_summary_type(value: Any) -> SummarizeType:
    """Resolve aliases and string values to a canonical SummarizeType.

    Args:
        value: Raw summary_type value (str, SummarizeType, or None).

    Returns:
        Canonical SummarizeType, defaulting to EXTRACTIVE for unknowns.
    """
    if isinstance(value, SummarizeType):
        return value
    raw = str(value).strip().lower() if value else ""
    mapped = _CHOICE_NORM.get(raw, raw)
    try:
        return SummarizeType(mapped)
    except ValueError:
        logger.warning("Unknown summary_type '%s' – defaulting to extractive", value)
        return SummarizeType.EXTRACTIVE


def _extract_text_from_json(text: str) -> str:
    """Extract plain text from a JSON-wrapped content envelope.

    Some tools return content as ``{"content": "...", ...}`` — this function
    unwraps it transparently.

    Args:
        text: Raw text, potentially JSON-encoded.

    Returns:
        Clean plain-text string.
    """
    clean = text.strip()
    if clean.startswith("{") and clean.endswith("}"):
        try:
            data = json.loads(clean)
            extracted = (
                data.get("content")
                or data.get("text")
                or data.get("document_info")
                or data.get("text_preview")
            )
            if extracted and isinstance(extracted, str):
                logger.debug(
                    "Extracted text from JSON wrapper (%d chars)", len(extracted)
                )
                return extracted.strip()
        except (json.JSONDecodeError, TypeError):
            pass
    return clean


def _handle_format_selection(text: str) -> Dict[str, Any]:
    """Return a requires_choice response with format recommendations.

    Args:
        text: The document text to analyse for format hints.

    Returns:
        Dict with status=requires_choice and recommendation.
    """
    logger.info("summary_type not specified – returning format suggestion")
    recommendation = _recommend_format(text)
    return {
        "status": "requires_choice",
        "message": "Выберите формат анализа документа:",
        "suggestion": recommendation,
    }


def _recommend_format(text: str) -> Dict[str, Any]:
    """Heuristically recommend a summarisation format for the given text.

    Args:
        text: Source text to analyse.

    Returns:
        Dict with recommended format, reason and stats.
    """
    if not text:
        return {
            "recommended": "abstractive",
            "reason": "Текст пуст",
            "stats": {"chars": 0, "lines": 0},
        }
    import re as _re

    chars = len(text)
    lines = text.count("\n")
    digit_groups = len(_re.findall(r"\d+", text))

    if chars > 5000 or digit_groups > 20:
        return {
            "recommended": "thesis",
            "reason": "Объёмный текст или много данных – тезисный план удобнее.",
            "stats": {"chars": chars, "lines": lines},
        }
    if lines < 5:
        return {
            "recommended": "abstractive",
            "reason": "Компактный текст – краткий пересказ достаточен.",
            "stats": {"chars": chars, "lines": lines},
        }
    return {
        "recommended": "extractive",
        "reason": "Много конкретики – ключевые факты будут полезнее.",
        "stats": {"chars": chars, "lines": lines},
    }


async def _perform_summarization(
    text: str,
    summary_type: SummarizeType,
) -> Dict[str, Any]:
    """Execute LLM-powered summarisation.

    Args:
        text: Pre-cleaned document text.
        summary_type: Canonical summarisation format.

    Returns:
        Dict with status=success and the summary content.
    """
    if len(text) < 50:
        return {
            "status": "success",
            "content": "Текст слишком мал для глубокого анализа.",
            "meta": {"format_used": summary_type.value, "text_length": len(text)},
        }

    processing_text = _truncate_large_text(text)
    llm = get_chat_model()

    try:
        summ_llm = llm.bind_tools([])
    except Exception:
        summ_llm = llm

    prompt = _build_summarization_prompt(summary_type)
    chain = prompt | summ_llm | StrOutputParser()

    logger.info(
        "Invoking LLM for summarization",
        extra={
            "summary_type": summary_type.value,
            "text_length": len(processing_text),
            "was_truncated": len(text) > _MAX_TEXT_LENGTH,
        },
    )

    summary: str = await chain.ainvoke({"text": processing_text})

    logger.info(
        "Summarization completed",
        extra={
            "summary_type": summary_type.value,
            "summary_length": len(summary),
        },
    )

    return {
        "status": "success",
        "content": summary.strip(),
        "meta": {
            "format_used": summary_type.value,
            "text_length": len(text),
            "was_truncated": len(text) > _MAX_TEXT_LENGTH,
        },
    }


def _truncate_large_text(text: str, max_length: int = _MAX_TEXT_LENGTH) -> str:
    """Truncate large texts while preserving head and tail for context.

    Strategy: first 67 % + last 33 % of max_length characters.

    Args:
        text: Source text.
        max_length: Maximum character budget.

    Returns:
        Original text if within budget, otherwise head + separator + tail.
    """
    if len(text) <= max_length:
        return text

    head = int(max_length * _HEAD_FRACTION)
    tail = max_length - head
    truncated = (
        text[:head]
        + "\n\n[... контент пропущен для оптимизации ...]\n\n"
        + text[-tail:]
    )
    logger.debug(
        "Text truncated: %d → %d chars (head=%d, tail=%d)",
        len(text),
        len(truncated),
        head,
        tail,
    )
    return truncated


def _build_summarization_prompt(summary_type: SummarizeType) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate tailored for the requested format.

    Args:
        summary_type: Canonical summarisation format.

    Returns:
        Ready-to-use ChatPromptTemplate.
    """
    instructions: dict[SummarizeType, str] = {
        SummarizeType.EXTRACTIVE: (
            "Выдели ключевые факты, конкретные даты, суммы, имена и обязательства. "
            "Оформи нумерованным списком. Каждый пункт — одна конкретная мысль."
        ),
        SummarizeType.ABSTRACTIVE: (
            "Напиши связный краткий пересказ сути документа своими словами "
            "(1–2 абзаца). Сохрани ключевую информацию, перефразируй без технических деталей."
        ),
        SummarizeType.THESIS: (
            "Сформируй структурированный тезисный план документа. "
            "Используй нумерацию разделов и подпункты. Каждый тезис — 1 предложение."
        ),
    }

    system_message = (
        "Ты — ведущий аналитик системы электронного документооборота (СЭД). "
        f"Задача: {instructions[summary_type]} "
        "Пиши строго по делу, на русском языке. "
        "НЕ добавляй фраз типа «В данном документе», «Данный текст» — сразу к сути."
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", "ИСХОДНЫЙ ТЕКСТ:\n{text}\n\nРЕЗУЛЬТАТ:"),
        ]
    )

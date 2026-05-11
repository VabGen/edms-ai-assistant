"""
EDMS AI Assistant — Document Summarisation Tool.
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
        description="Текст документа для суммаризации",
        min_length=1,
        max_length=50_000,
    )
    summary_type: SummarizeType | None = Field(
        None,
        description=(
            "Формат суммаризации: "
            "extractive (ключевые факты), "
            "abstractive (краткий пересказ), "
            "thesis (тезисный план). "
            "Если None — пользователь выбирает формат (Human-in-the-Loop)."
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
    """Извлекает текст из JSON-обёртки если doc_get_file_content вернул JSON."""
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
                f"({numeric_groups} чисел) — тезисный план удобнее."
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


# ---------------------------------------------------------------------------
# Глобальный синглтон сервиса (без import cycle)
# ---------------------------------------------------------------------------

_service_instance: Any | None = None


def set_summarization_service(service: Any) -> None:
    """Вызывается из lifespan при инициализации приложения."""
    global _service_instance
    _service_instance = service
    logger.info("SummarizationService зарегистрирован в tool")


def get_summarization_service() -> Any | None:
    return _service_instance


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool("doc_summarize_text", args_schema=SummarizeInput)
async def doc_summarize_text(
    text: str,
    summary_type: SummarizeType | None = None,
) -> dict[str, Any]:
    """Анализирует текст документа через LLM-пайплайн суммаризации.

    Используй ТОЛЬКО когда пользователь явно просит:
    - суммаризировать / кратко изложить / пересказать документ
    - извлечь ключевые факты
    - составить тезисный план

    НЕ используй для простых вопросов о документе (кто автор, какая дата и т.д.).

    Human-in-the-Loop:
        Когда summary_type = None — возвращает requires_choice с тремя вариантами.
        Агент показывает выбор пользователю и ждёт ответа перед повторным вызовом.

    Args:
        text: Текст документа (plain или JSON-обёртка от doc_get_file_content).
        summary_type: Формат или None для запроса у пользователя.

    Returns:
        Dict со статусом:
        - requires_choice: нужно выбрать формат
        - success: анализ выполнен, поле content содержит Markdown
        - error: ошибка выполнения
    """
    logger.info(
        "doc_summarize_text: text_length=%d type=%s",
        len(text),
        summary_type.value if summary_type else None,
    )

    clean_text = _unwrap_json_envelope(text)

    # Human-in-the-Loop — запрашиваем выбор формата
    if summary_type is None:
        return _build_requires_choice_response(clean_text)

    normalised = _normalise_summary_type(summary_type)

    # Получаем сервис через синглтон (без import cycle)
    service = get_summarization_service()
    if service is None:
        # Fallback: используем простой LLM без пайплайна
        return await _llm_fallback(clean_text, normalised)

    try:
        from edms_ai_assistant.summarizer.service import (
            SummarizationRequest,
            format_output_as_markdown,
        )
        from edms_ai_assistant.summarizer.structured.models import SummaryMode

        _mode_map = {
            SummarizeType.EXTRACTIVE: SummaryMode.EXTRACTIVE,
            SummarizeType.ABSTRACTIVE: SummaryMode.ABSTRACTIVE,
            SummarizeType.THESIS: SummaryMode.THESIS,
        }
        mode = _mode_map.get(normalised, SummaryMode.ABSTRACTIVE)

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
        content = format_output_as_markdown(resp)

        return {
            "status": "success",
            "content": content,
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
        # Пробуем fallback
        try:
            return await _llm_fallback(clean_text, normalised)
        except Exception:
            return {
                "status": "error",
                "message": f"Не удалось проанализировать документ: {exc}",
            }


async def _llm_fallback(text: str, summary_type: SummarizeType) -> dict[str, Any]:
    """
    Простой LLM fallback без пайплайна суммаризации.
    Используется когда сервис недоступен или упал пайплайн.
    """
    from edms_ai_assistant.llm import get_chat_model

    _prompts = {
        SummarizeType.EXTRACTIVE: (
            "Извлеки ключевые факты из документа. "
            "Формат: список фактов с категориями (ДАТА, ПЕРСОНА, ОРГАНИЗАЦИЯ, СУММА, ТРЕБОВАНИЕ). "
            "Язык ответа: русский."
        ),
        SummarizeType.ABSTRACTIVE: (
            "Напиши краткое изложение документа своими словами. "
            "2-4 абзаца, профессиональный стиль. "
            "Язык ответа: русский."
        ),
        SummarizeType.THESIS: (
            "Составь тезисный план документа. "
            "Формат: пронумерованные разделы с подпунктами. "
            "Язык ответа: русский."
        ),
    }

    prompt = _prompts.get(summary_type, _prompts[SummarizeType.ABSTRACTIVE])
    llm = get_chat_model()

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=f"{prompt}\n\nОтвечай ТОЛЬКО на русском языке."),
        HumanMessage(content=f"Документ:\n\n{text[:8000]}"),
    ]

    response = await llm.ainvoke(messages)
    content = str(response.content).strip()

    return {
        "status": "success",
        "content": content,
        "meta": {
            "format_used": summary_type.value,
            "text_length": len(text),
            "pipeline": "llm_fallback",
            "chunks_processed": 1,
            "from_cache": False,
        },
    }

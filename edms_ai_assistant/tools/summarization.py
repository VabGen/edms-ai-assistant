"""
EDMS AI Assistant - Document Summarization Tool.

LLM-powered инструмент для интеллектуального анализа и сжатия текстов.
"""

import json
import logging
from enum import StrEnum
from typing import Any, Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.services.nlp_service import EDMSNaturalLanguageService

logger = logging.getLogger(__name__)


class SummarizeType(StrEnum):
    """Типы суммаризации документов."""

    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    THESIS = "thesis"


class SummarizeInput(BaseModel):
    """Валидированная схема входных данных для суммаризации."""

    text: str = Field(
        ...,
        description="Текст документа для суммаризации",
        min_length=1,
        max_length=50000,
    )
    summary_type: Optional[SummarizeType] = Field(
        None,
        description=(
            "ОБЯЗАТЕЛЬНО оставь это поле пустым (null), если пользователь "
            "не указал конкретный формат (например, 'тезисы' или 'факты'). "
            "Если поле пустое, система предложит пользователю выбрать формат."
        ),
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Валидация и очистка входного текста."""
        if not v or not v.strip():
            raise ValueError("Текст не может быть пустым")
        return v.strip()


@tool("doc_summarize_text", args_schema=SummarizeInput)
async def doc_summarize_text(
    text: str, summary_type: Optional[SummarizeType] = None
) -> Dict[str, Any]:
    """
    Выполняет интеллектуальный анализ и сжатие текста документа.

    Поддерживаемые форматы:
    - EXTRACTIVE: Ключевые факты, даты, суммы (список)
    - ABSTRACTIVE: Краткий пересказ (1-2 абзаца)
    - THESIS: Структурированный тезисный план

    Workflow:
    1. Если summary_type=None → возврат requires_choice с рекомендацией
    2. Если summary_type указан → выполнение суммаризации
    3. Автоматическая очистка JSON-артефактов из входного текста
    4. Truncation больших документов (8K начало + 4K конец)

    Args:
        text: Текст документа для анализа
        summary_type: Тип суммаризации (опционально)

    Returns:
        Dict с ключами:
        - status: "success" | "requires_choice" | "error"
        - content: Результат суммаризации (для success)
        - message: Информационное сообщение
        - suggestion: Рекомендация формата (для requires_choice)
        - meta: Метаданные обработки

    Examples:
         # Без указания типа (требует выбора)
         result = await doc_summarize_text(text="Длинный текст...")
         # {"status": "requires_choice", "suggestion": {...}, ...}

         # С указанием типа
         result = await doc_summarize_text(
             text="Длинный текст...",
             summary_type=SummarizeType.EXTRACTIVE
         )
         # {"status": "success", "content": "Ключевые факты:...", ...}
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

        return await _perform_summarization(clean_text, summary_type)

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return {
            "status": "error",
            "message": f"Ошибка валидации: {str(e)}",
        }
    except Exception as e:
        logger.error(
            f"Summarization failed: {e}",
            exc_info=True,
            extra={"summary_type": summary_type},
        )
        return {
            "status": "error",
            "message": f"Не удалось проанализировать текст: {str(e)}",
        }


def _extract_text_from_json(text: str) -> str:
    """
    Извлекает чистый текст из JSON-обернутого контента.

    Обрабатывает случаи, когда текст приходит в формате:
    {"content": "...", ...} или {"document_info": "...", ...}

    Args:
        text: Исходный текст (может быть JSON)

    Returns:
        Очищенный текст
    """
    clean_text = text.strip()

    if clean_text.startswith("{") and clean_text.endswith("}"):
        try:
            data = json.loads(clean_text)
            clean_text = data.get("content") or data.get("document_info") or clean_text
            logger.debug("Extracted text from JSON wrapper")
        except json.JSONDecodeError:
            pass

    return clean_text


def _handle_format_selection(text: str) -> Dict[str, Any]:
    """
    Обрабатывает случай, когда пользователь не выбрал формат суммаризации.

    Анализирует текст и предлагает оптимальный формат на основе характеристик.

    Args:
        text: Текст для анализа

    Returns:
        Dict со статусом requires_choice и рекомендацией
    """
    logger.info("Summary type not specified - returning format suggestion")

    nlp = EDMSNaturalLanguageService()
    analysis = nlp.suggest_summarize_format(text)

    return {
        "status": "requires_choice",
        "message": "Выберите формат анализа документа:",
        "suggestion": analysis,
    }


async def _perform_summarization(
    text: str, summary_type: SummarizeType
) -> Dict[str, Any]:
    """
    Выполняет суммаризацию текста с использованием LLM.

    Args:
        text: Очищенный текст для суммаризации
        summary_type: Тип суммаризации

    Returns:
        Dict с результатом суммаризации
    """
    if len(text) < 50:
        logger.info("Text too short for deep analysis")
        return {
            "status": "success",
            "content": "Текст слишком мал для глубокого анализа.",
        }

    processing_text = _truncate_large_text(text)

    llm = get_chat_model()
    summ_llm = llm.copy(update={"parallel_tool_calls": None}).bind_tools([])

    prompt = _build_summarization_prompt(summary_type)
    chain = prompt | summ_llm | StrOutputParser()

    logger.debug(
        "Invoking LLM for summarization",
        extra={
            "summary_type": summary_type.value,
            "text_length": len(processing_text),
        },
    )

    summary = await chain.ainvoke({"text": processing_text})

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
            "was_truncated": len(text) > 12000,
        },
    }


def _truncate_large_text(text: str, max_length: int = 12000) -> str:
    """
    Обрезает большие тексты с сохранением начала и конца.

    Strategy: Первые 8K символов + последние 4K символов для контекста.

    Args:
        text: Исходный текст
        max_length: Максимальная длина (default: 12000)

    Returns:
        Обрезанный текст или исходный если длина в пределах лимита
    """
    if len(text) <= max_length:
        return text

    head_length = int(max_length * 0.67)
    tail_length = int(max_length * 0.33)

    truncated = (
        text[:head_length]
        + "\n\n[... контент пропущен для оптимизации ...]\n\n"
        + text[-tail_length:]
    )

    logger.debug(
        f"Text truncated: {len(text)} → {len(truncated)} chars "
        f"(head: {head_length}, tail: {tail_length})"
    )

    return truncated


def _build_summarization_prompt(summary_type: SummarizeType) -> ChatPromptTemplate:
    """
    Строит prompt template для суммаризации с учетом типа.

    Args:
        summary_type: Тип суммаризации

    Returns:
        ChatPromptTemplate с инструкциями для LLM
    """
    instructions = {
        SummarizeType.EXTRACTIVE: (
            "Выдели ключевые факты, даты, суммы и конкретные обязательства. "
            "Оформи списком с краткими пояснениями."
        ),
        SummarizeType.ABSTRACTIVE: (
            "Напиши связный краткий пересказ сути документа своими словами "
            "(1-2 абзаца). Сохрани ключевую информацию, но перефразируй."
        ),
        SummarizeType.THESIS: (
            "Сформируй структурированный тезисный план документа с выделением "
            "главных мыслей. Используй нумерацию и подпункты."
        ),
    }

    system_message = (
        f"Ты — ведущий аналитик СЭД. "
        f"Задача: {instructions[summary_type]} "
        f"Пиши строго по делу, на русском языке."
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", "ИСХОДНЫЙ ТЕКСТ:\n{text}\n\nРЕЗУЛЬТАТ:"),
        ]
    )

# edms_ai_assistant\tools\summarization.py
import logging
from enum import StrEnum
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from edms_ai_assistant.llm import get_chat_model

logger = logging.getLogger(__name__)


class SummarizeType(StrEnum):
    EXTRACTIVE = "extractive"  # Извлечение ключевых фактов без изменений
    ABSTRACTIVE = "abstractive"  # Пересказ своими словами (краткое содержание)
    THESIS = "thesis"  # Основные положения (тезисный план)


class SummarizeInput(BaseModel):
    text: str = Field(..., description="Текст документа для суммаризации")
    summary_type: Optional[SummarizeType] = Field(
        SummarizeType.ABSTRACTIVE,
        description="Тип суммаризации: экстрактивная (факты), абстрактивная (пересказ) или тезисная."
    )


@tool("doc_summarize_text", args_schema=SummarizeInput)
async def doc_summarize_text(
        text: str,
        summary_type: SummarizeType = SummarizeType.ABSTRACTIVE
) -> Dict[str, Any]:
    """
    Создает краткую выжимку (summary) предоставленного текста выбранным методом.
    """
    try:
        llm = get_chat_model()

        instructions = {
            SummarizeType.EXTRACTIVE: (
                "действуй как метод экстрактивной суммаризации. Выдели и процитируй наиболее важные "
                "факты, сущности и предложения из текста без искажения их оригинального смысла."
            ),
            SummarizeType.ABSTRACTIVE: (
                "действуй как метод абстрактивной суммаризации. Напиши лаконичный пересказ сути "
                "документа своими словами, сохраняя логическую связь и основные идеи."
            ),
            SummarizeType.THESIS: (
                "сформируй тезисный план документа. Выдели ключевые положения и идеи в виде "
                "структурированного списка (тезисов)."
            )
        }

        target_instruction = instructions.get(summary_type, instructions[SummarizeType.ABSTRACTIVE])

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Ты — эксперт-аналитик EDMS. Твоя задача: {target_instruction}"),
            ("user", "Текст для обработки:\n\n{text}\n\nРезультат:")
        ])

        chain = prompt | llm | StrOutputParser()

        # Логика обрезки контекста
        if len(text) > 40000:
            logger.info("Текст слишком длинный, используем комбинированное окно")
            processing_text = text[:20000] + "\n... [пропуск части текста] ...\n" + text[-10000:]
        else:
            processing_text = text

        summary = await chain.ainvoke({"text": processing_text})

        return {
            "status": "success",
            "type_used": summary_type,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Ошибка при суммаризации: {e}", exc_info=True)
        return {"error": f"Не удалось сжать текст: {str(e)}"}
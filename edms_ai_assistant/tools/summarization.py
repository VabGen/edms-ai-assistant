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
    ВАЖНО: Используй этот инструмент ТОЛЬКО когда пользователь просит сделать 'сводку', 'выжимку',
    'резюме' или 'краткое содержание' текста файла.
    Этот инструмент берет сырой текст документа и превращает его в краткий аналитический отчет.
    Поддерживает экстрактивный, абстрактивный и тезисный типы.
    """
    try:
        llm = get_chat_model()

        instructions = {
            SummarizeType.EXTRACTIVE: "выдели ключевые факты и цитаты",
            SummarizeType.ABSTRACTIVE: "напиши краткий пересказ сути документа своими словами",
            SummarizeType.THESIS: "сформируй структурированный тезисный план"
        }

        target_instruction = instructions.get(summary_type, instructions[SummarizeType.ABSTRACTIVE])

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Ты — эксперт-аналитик СЭД. Твоя задача: {target_instruction}. "
                       f"Отвечай кратко и только по делу."),
            ("user", "ТЕКСТ ДОКУМЕНТА:\n{text}\n\nРЕЗУЛЬТАТ:")
        ])

        if len(text) > 30000:
            processing_text = text[:15000] + "\n... [ТЕКСТ ОБРЕЗАН ДЛЯ АНАЛИЗА] ...\n" + text[-10000:]
        else:
            processing_text = text

        chain = prompt | llm | StrOutputParser()
        summary = await chain.ainvoke({"text": processing_text})

        return {
            "status": "success",
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return {"error": str(e)}

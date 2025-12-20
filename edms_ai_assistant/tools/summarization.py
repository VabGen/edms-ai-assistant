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


class SummarizeFocus(StrEnum):
    GENERAL = "general"
    LEGAL = "legal"
    FINANCE = "finance"
    DATES = "dates"


class SummarizeInput(BaseModel):
    text: str = Field(..., description="Текст документа для суммаризации")
    focus: Optional[SummarizeFocus] = Field(
        SummarizeFocus.GENERAL,
        description="Фокус анализа: общие тезисы, юр. детали, финансы или даты"
    )


@tool("doc_summarize_text", args_schema=SummarizeInput)
async def doc_summarize_text(
        text: str,
        focus: SummarizeFocus = SummarizeFocus.GENERAL
) -> Dict[str, Any]:
    """
    Создает краткую выжимку (summary) предоставленного текста.
    Используется, когда файл слишком большой для прямого чтения агентом.
    """
    try:
        llm = get_chat_model()

        instructions = {
            SummarizeFocus.GENERAL: "основные тезисы, суть и цель документа",
            SummarizeFocus.LEGAL: "стороны, права, ключевые обязательства и штрафы",
            SummarizeFocus.FINANCE: "денежные суммы, валюты, условия оплаты и реквизиты",
            SummarizeFocus.DATES: "важные даты, сроки действия и дедлайны"
        }
        target_instruction = instructions.get(focus, instructions[SummarizeFocus.GENERAL])

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Ты — ведущий аналитик EDMS. Твоя задача: прочитать текст и выделить {target_instruction}."),
            ("user", "Текст документа для анализа:\n\n{text}\n\nНапиши выжимку кратко, по пунктам:")
        ])

        chain = prompt | llm | StrOutputParser()

        if len(text) > 40000:
            logger.info("Текст слишком длинный, используем комбинированное окно")
            processing_text = text[:20000] + "\n... [пропуск части текста] ...\n" + text[-10000:]
        else:
            processing_text = text

        summary = await chain.ainvoke({"text": processing_text})

        return {
            "status": "success",
            "focus_used": focus,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Ошибка при суммаризации через LCEL: {e}", exc_info=True)
        return {"error": f"Не удалось сжать текст: {str(e)}"}

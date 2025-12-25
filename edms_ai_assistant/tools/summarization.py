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
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    THESIS = "thesis"


class SummarizeInput(BaseModel):
    text: str = Field(..., description="Текст документа для суммаризации")
    summary_type: Optional[SummarizeType] = Field(
        None,
        description=(
            "ОБЯЗАТЕЛЬНО оставь это поле пустым (null), если пользователь "
            "не указал конкретный формат (например, 'тезисы' или 'факты'). "
            "Если поле пустое, система предложит пользователю выбрать формат кнопками."
        )
    )


@tool("doc_summarize_text", args_schema=SummarizeInput)
async def doc_summarize_text(
        text: str,
        summary_type: Optional[SummarizeType] = None
) -> Dict[str, Any]:
    """
    Инструмент для глубокого анализа и суммаризации текста.
    ВАЖНО: Если ты вызываешь этот инструмент БЕЗ указания summary_type (summary_type=null),
    выполнение будет немедленно ПРЕРВАНО для запроса формата у пользователя.
    Используй это поведение для всех общих запросов типа 'сделай сводку', 'о чем файл', 'о чем вложения' или 'проанализируй'.
    """
    if summary_type is None:
        return {
            "status": "requires_choice",
            "message": "Формат суммаризации не определен."
        }

    actual_type = summary_type

    logger.info(f"Executing summarization. Type: {actual_type}")

    try:
        clean_text = text.strip()
        if clean_text.startswith("{") and clean_text.endswith("}"):
            import json
            try:
                data = json.loads(clean_text)
                clean_text = data.get("content", clean_text)
            except json.JSONDecodeError:
                pass

        if not clean_text or len(clean_text) < 50:
            return {
                "status": "success",
                "content": f"Текст слишком краткий для анализа:\n\n{clean_text}"
            }

        if len(clean_text) > 12000:
            processing_text = clean_text[:8000] + "\n[... контент обрезан ...]\n" + clean_text[-4000:]
        else:
            processing_text = clean_text

        llm = get_chat_model()
        summ_llm = llm.copy(update={"parallel_tool_calls": None}).bind_tools([])

        instructions = {
            SummarizeType.EXTRACTIVE: "выдели ключевые факты и цитаты",
            SummarizeType.ABSTRACTIVE: "напиши краткий пересказ сути документа своими словами",
            SummarizeType.THESIS: "сформируй структурированный тезисный план"
        }

        target_instruction = instructions.get(actual_type, instructions[SummarizeType.ABSTRACTIVE])

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"Ты — эксперт-аналитик СЭД. Твоя задача: {target_instruction}. Отвечай кратко, структурировано, на русском языке."),
            ("user", "ТЕКСТ ДЛЯ АНАЛИЗА:\n{text}\n\nРЕЗУЛЬТАТ:")
        ])

        chain = prompt | summ_llm | StrOutputParser()
        summary = await chain.ainvoke({"text": processing_text})

        return {
            "status": "success",
            "content": summary.strip(),
            "summary_type": actual_type
        }

    except Exception as e:
        logger.error(f"Summarization error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
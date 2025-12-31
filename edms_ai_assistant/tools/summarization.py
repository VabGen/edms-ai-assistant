import logging
from enum import StrEnum
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.services.nlp_service import EDMSNaturalLanguageService

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
        ),
    )


@tool("doc_summarize_text", args_schema=SummarizeInput)
async def doc_summarize_text(
    text: str, summary_type: Optional[SummarizeType] = None
) -> Dict[str, Any]:
    """
    Выполняет интеллектуальный анализ и сжатие текста.
    Если summary_type не указан, возвращает статус 'requires_choice' с рекомендацией формата.
    """
    nlp = EDMSNaturalLanguageService()
    clean_text = text.strip()
    if clean_text.startswith("{") and clean_text.endswith("}"):
        try:
            import json

            data = json.loads(clean_text)
            clean_text = data.get("content", data.get("document_info", clean_text))
        except:
            pass

    if summary_type is None:
        analysis = nlp.suggest_summarize_format(clean_text)
        return {
            "status": "requires_choice",
            "message": "Выберите формат анализа документа.",
            "suggestion": analysis,
        }

    logger.info(f"[NLP-SUMMARIZE] Обработка типа: {summary_type}")

    try:
        if not clean_text or len(clean_text) < 50:
            return {
                "status": "success",
                "content": "Текст слишком мал для глубокого анализа.",
            }

        if len(clean_text) > 12000:
            processing_text = (
                clean_text[:8000]
                + "\n[... контент пропущен для оптимизации ...]\n"
                + clean_text[-4000:]
            )
        else:
            processing_text = clean_text

        llm = get_chat_model()
        summ_llm = llm.copy(update={"parallel_tool_calls": None}).bind_tools([])

        instructions = {
            SummarizeType.EXTRACTIVE: "Выдели ключевые факты, даты, суммы и конкретные обязательства. Оформи списком.",
            SummarizeType.ABSTRACTIVE: "Напиши связный краткий пересказ сути документа своими словами (1-2 абзаца).",
            SummarizeType.THESIS: "Сформируй структурированный тезисный план документа с выделением главных мыслей.",
        }

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"Ты — ведущий аналитик СЭД. Задача: {instructions[summary_type]}. Пиши строго по делу, на русском языке.",
                ),
                ("user", "ИСХОДНЫЙ ТЕКСТ:\n{text}\n\nРЕЗУЛЬТАТ:"),
            ]
        )

        chain = prompt | summ_llm | StrOutputParser()
        summary = await chain.ainvoke({"text": processing_text})

        return {
            "status": "success",
            "content": summary.strip(),
            "meta": {"format_used": summary_type, "text_length": len(clean_text)},
        }

    except Exception as e:
        logger.error(f"Ошибка суммаризации: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Не удалось проанализировать текст: {str(e)}",
        }

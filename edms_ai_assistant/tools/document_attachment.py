# edms_ai_assistant/tools/document_attachment.py

import json
import logging
from pydantic import BaseModel, Field
from .base import EdmsApiClient
# Предполагаем, что у вас есть этот утилитный файл
from edms_ai_assistant.utils.file_utils import extract_text_from_bytes

logger = logging.getLogger(__name__)

# Максимальная длина текста для LLM (переиспользуем из вашего старого кода)
MAX_LLM_TEXT_LENGTH = 8000


class SummarizeAttachmentArgs(BaseModel):
    document_id: str = Field(description="UUID документа.")
    attachment_id: str = Field(description="UUID вложения.")
    file_name: str = Field(description="Имя файла для распознавания его типа.")

    # Поля контракта для шаблонизации
    contract_sum: float | None = Field(None, description="Сумма договора.")
    reg_date: str | None = Field(None, description="Дата регистрации/заключения.")
    duration_end: str | None = Field(None, description="Срок окончания действия.")


class DocumentAttachmentTools:
    """Инструменты для работы с вложениями документов (скачивание, сводка)."""

    def __init__(self, api_client: EdmsApiClient, llm_summarizer):
        self.api_client = api_client
        self.llm_summarizer = llm_summarizer

    async def summarize(
            self,
            document_id: str,
            attachment_id: str,
            file_name: str,
            contract_sum: float | None = None,
            reg_date: str | None = None,
            duration_end: str | None = None
    ) -> str:
        """
        Скачать и прочитать содержимое вложения, а затем создать его сводку,
        автоматически подставляя метаданные в текст шаблона.
        """
        logger.info(f"Вызов API: doc_attachment.summarize для {attachment_id}")

        # 1. Скачивание файла
        endpoint = f"api/document/{document_id}/attachment/{attachment_id}"
        file_bytes = await self.api_client.download(endpoint)

        if file_bytes is None:
            return json.dumps({"error": "Ошибка при скачивании файла."}, ensure_ascii=False)

        # 2. ИЗВЛЕЧЕНИЕ ТЕКСТА
        try:
            text_content = extract_text_from_bytes(file_bytes, file_name)
        except Exception as e:
            logger.error(f"Ошибка извлечения текста: {e}")
            return json.dumps({"error": f"Ошибка извлечения текста из файла: {type(e).__name__}"}, ensure_ascii=False)

        # 3. Формирование списка метаданных для шаблонизации
        template_data = {
            "CONTRACT_SUM": contract_sum,
            "REG_DATE": reg_date,
            "CONTRACT_DURATION_END": duration_end,
            # Добавьте другие поля, которые могут быть в шаблонах
        }

        template_instructions = "\n".join(
            f"- {{'{key}'}} = {value if value is not None else 'не заполнено (null)'}"
            for key, value in template_data.items()
        )

        # 4. Формирование промпта с инструкциями по шаблонизации
        summarizer_prompt = f"""
Создай краткое содержание (3-5 предложений) на русском языке. 
Выдели суть, ключевые условия, стороны, суммы, даты. 
Перед созданием резюме **замени все плейсхолдеры в тексте** (вида {{KEY}}) на соответствующие значения из предоставленных метаданных. 
Если значение метаданных 'null' или отсутствует, укажи это явно.

**Доступные метаданные для замены:**
{template_instructions}

СТРОГО ОТВЕТЬ ТОЛЬКО РЕЗЮМЕ, без пояснений или заголовков.
Текст:
{text_content[:MAX_LLM_TEXT_LENGTH]}
"""

        # 5. Вызов LLM для суммаризации
        try:
            response_message = await self.llm_summarizer.ainvoke([{"role": "user", "content": summarizer_prompt}])
            summary = getattr(response_message, "content", str(response_message))
        except Exception as e:
            logger.error(f"Ошибка суммаризации LLM: {e}")
            summary = f"Не удалось получить резюме от модели: {type(e).__name__}"

        # 6. ВОЗВРАТ РЕЗУЛЬТАТА
        return json.dumps({
            "status": "success",
            "file_name": file_name,
            "summary": summary,
            "size_bytes": len(file_bytes)
        }, ensure_ascii=False)
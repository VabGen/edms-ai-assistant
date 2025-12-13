# tools/document_tools.py
import json
from typing import Optional, Type, Dict, Any, List
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from .base import EdmsApiClient
import logging

logger = logging.getLogger(__name__)

from edms_ai_assistant.llm import get_chat_model
# Предполагаем, что этот файл существует после предыдущих шагов
from edms_ai_assistant.utils.file_utils import extract_text_from_bytes

# Максимальная длина текста для LLM
MAX_LLM_TEXT_LENGTH = 8000
# Универсальный промпт для суммаризации
DEFAULT_SUMMARY_PROMPT = "Создай краткое содержание (3-5 предложений) на русском языке. Выдели суть, ключевые условия, стороны, суммы, даты. СТРОГО ОТВЕТЬ ТОЛЬКО РЕЗЮМЕ, без пояснений или заголовков."


# --- Input Schemas (Inputs for LLM) ---

class GetDocumentByIdInput(BaseModel):
    document_id: str = Field(..., description="UUID документа.")


class DownloadAttachmentInput(BaseModel):
    document_id: str = Field(..., description="UUID документа.")
    attachment_id: str = Field(..., description="UUID вложения.")
    file_name: Optional[str] = Field(None, description="Имя файла (если известно).")  # <--- Важное поле


# --- Tool Implementations ---

class GetDocumentByIdTool(BaseTool):
    name: str = "get_document_by_id"
    description: str = "Получить детальную информацию о документе по его ID (UUID). Используй как первый шаг для поиска ID вложения (attachment_id)."
    args_schema: Type[BaseModel] = GetDocumentByIdInput
    api_client: EdmsApiClient = Field(exclude=True)

    def _run(self, document_id: str):
        raise NotImplementedError("Use async _arun")

    async def _arun(self, document_id: str) -> str:
        logger.info(f"Вызов API: Получение документа по ID: {document_id}")
        endpoint = f"api/document/{document_id}"

        data = await self.api_client.get(endpoint)

        if not data or data.get("error"):
            return json.dumps({"error": "Документ не найден или ошибка API.", "details": data}, ensure_ascii=False)

        attachments: List[Dict[str, Any]] = data.get("attachmentDocument", [])

        # Фильтруем важные поля для Planner (включая ID в корне и name в attachment)
        filtered_data = {
            "id": data.get("id"),  # <--- Ключ .id нужен для маппинга document_id
            "regNumber": data.get("regNumber"),
            "docType": data.get("documentType", {}).get("name"),
            "attachments_count": len(attachments),
            "attachments": [
                {
                    "attachment_id": att.get("id"),
                    "name": att.get("name"),  # <--- Ключ .name нужен для маппинга file_name
                    "mimeType": att.get("mimeType")
                }
                for att in attachments
            ]
        }
        return json.dumps(filtered_data, ensure_ascii=False)


class DownloadAndSummarizeAttachmentTool(BaseTool):
    name: str = "download_and_summarize_attachment"
    description: str = "Скачать и прочитать содержимое вложения документа. Требует document_id и attachment_id. ИСПОЛЬЗУЙ ТОЛЬКО ПОСЛЕ ПОЛУЧЕНИЯ attachment_id ИЗ get_document_by_id."
    args_schema: Type[BaseModel] = DownloadAttachmentInput
    api_client: EdmsApiClient = Field(exclude=True)

    def _run(self, document_id: str, attachment_id: str, file_name: str = None):
        raise NotImplementedError("Use async _arun")

    async def _arun(self, document_id: str, attachment_id: str, file_name: str = None) -> str:
        logger.info(f"Вызов API: Скачивание вложения {attachment_id} для документа {document_id}")
        endpoint = f"api/document/{document_id}/attachment/{attachment_id}"
        file_bytes = await self.api_client.download(endpoint)

        # Используем переданное имя файла
        file_name_clean = file_name or "без имени"

        if file_bytes is None:
            return json.dumps({"error": "Ошибка при скачивании файла. Возможно, технический сбой или файл недоступен."},
                              ensure_ascii=False)

        # 1. ИЗВЛЕЧЕНИЕ ТЕКСТА
        try:
            text = extract_text_from_bytes(file_bytes, file_name_clean)
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из {file_name_clean}: {e}")
            return json.dumps({"error": f"Ошибка извлечения текста из файла: {type(e).__name__}"}, ensure_ascii=False)

        if not text or len(text) < 50:
            return json.dumps({
                "summary": f"Файл '{file_name_clean}' содержит слишком мало текста или его формат не поддерживается для суммаризации. Размер: {len(file_bytes)} байт."},
                ensure_ascii=False)

        # 2. СУММАРИЗАЦИЯ ТЕКСТА С ПОМОЩЬЮ LLM
        llm = get_chat_model()
        text_to_summarize = text[:MAX_LLM_TEXT_LENGTH]

        prompt = DEFAULT_SUMMARY_PROMPT + f"\nТекст:\n{text_to_summarize}"

        try:
            response = await llm.ainvoke([{"role": "user", "content": prompt}])
            summary = getattr(response, "content", str(response))
        except Exception as e:
            logger.error(f"Ошибка суммаризации LLM: {e}")
            summary = f"Не удалось получить резюме от модели: {type(e).__name__}"

        # 3. ВОЗВРАТ РЕЗУЛЬТАТА
        return json.dumps({
            "status": "success",
            "file_name": file_name_clean,
            "size_bytes": len(file_bytes),
            "summary": summary
        }, ensure_ascii=False)
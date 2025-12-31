# edms_ai_assistant\tools\attachment.py
import logging
import os
import tempfile
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.clients.attachment_client import EdmsAttachmentClient
from edms_ai_assistant.generated.resources_openapi import (
    DocumentDto,
    AttachmentDocumentType,
)
from edms_ai_assistant.services.file_processor import FileProcessorService
from edms_ai_assistant.services.nlp_service import EDMSNaturalLanguageService

logger = logging.getLogger(__name__)


class AttachmentFetchInput(BaseModel):
    document_id: str = Field(..., description="UUID документа")
    token: str = Field(..., description="Токен авторизации пользователя")
    attachment_id: Optional[str] = Field(
        None,
        description="UUID конкретного файла. Если не указан, будет выбран основной документ (печатная форма).",
    )


@tool("doc_get_file_content", args_schema=AttachmentFetchInput)
async def doc_get_file_content(
    document_id: str, token: str, attachment_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Извлекает содержимое файла для анализа.
    Если ID не указан, возвращает список доступных файлов с их описанием.
    """
    try:
        async with DocumentClient() as doc_client:
            raw_data = await doc_client.get_document_metadata(token, document_id)
            doc = DocumentDto.model_validate(raw_data)
            attachments = doc.attachmentDocument

        if not attachments:
            return {"status": "info", "message": "В документе отсутствуют вложения."}

        nlp = EDMSNaturalLanguageService()

        if not attachment_id:
            files_info = [nlp.analyze_attachment_meta(a) for a in attachments]
            return {
                "status": "need_selection",
                "message": "Укажите attachment_id для анализа текста. Список доступных файлов:",
                "files": files_info,
            }

        target = next((a for a in attachments if str(a.id) == attachment_id), None)
        if not target:
            return {"error": f"Файл с ID {attachment_id} не найден."}

        async with EdmsAttachmentClient() as attach_client:
            content_bytes = await attach_client.get_attachment_content(
                token, document_id, attachment_id
            )

        file_info = nlp.analyze_attachment_meta(target)
        suffix = os.path.splitext(target.name)[1] or ".tmp"

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content_bytes)
                tmp_path = tmp.name

            try:
                text_content = FileProcessorService.extract_text(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except Exception as parse_err:
            logger.warning(f"Ошибка парсинга {target.name}: {parse_err}")
            text_content = f"[Ошибка извлечения текста: файл может быть поврежден или является сканом без OCR]"

        return {
            "status": "success",
            "meta": file_info,
            "text_preview": text_content[:15000],
            "is_truncated": len(text_content) > 15000,
        }

    except Exception as e:
        logger.error(f"Ошибка doc_get_file_content: {e}", exc_info=True)
        return {"error": f"Не удалось прочитать файл: {str(e)}"}

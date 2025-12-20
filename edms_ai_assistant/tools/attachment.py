# edms_ai_assistant\tools\attachment.py
import logging
import os
import tempfile
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.clients.attachment_client import EdmsAttachmentClient
from edms_ai_assistant.generated.resources_openapi import DocumentDto, AttachmentDocumentType
from edms_ai_assistant.services.file_processor import FileProcessorService

logger = logging.getLogger(__name__)


class AttachmentFetchInput(BaseModel):
    document_id: str = Field(..., description="UUID документа")
    token: str = Field(..., description="Токен авторизации пользователя")
    attachment_id: Optional[str] = Field(
        None,
        description="UUID конкретного файла. Если не указан, будет выбран основной документ (печатная форма)."
    )


@tool("doc_get_file_content", args_schema=AttachmentFetchInput)
async def doc_get_file_content(document_id: str, token: str, attachment_id: Optional[str] = None) -> Dict[str, Any]:
    """Скачивает файл и извлекает текст. Если ID не указан, возвращает список файлов."""
    try:
        async with DocumentClient() as doc_client:
            raw_data = await doc_client.get_document_metadata(token, document_id)
            attachments = raw_data.get("attachmentDocument", []) if isinstance(raw_data, dict) else (
                        raw_data.attachmentDocument or [])

        if not attachments:
            return {"error": "В документе нет прикрепленных файлов."}

        if not attachment_id:
            file_list = "\n".join([f"- {a.get('name', 'Без имени')} (ID: {a.get('id')})" for a in attachments])
            return {
                "status": "error",
                "message": f"Необходимо указать attachment_id. Список файлов:\n{file_list}"
            }

        target_file = next((a for a in attachments if str(a.get('id')) == attachment_id), None)
        if not target_file:
            return {"error": f"Файл с ID {attachment_id} не найден в метаданных документа."}

        file_name = target_file.get("name", "document.tmp")
        suffix = os.path.splitext(file_name)[1] or ".tmp"

        async with EdmsAttachmentClient() as attach_client:
            content_bytes = await attach_client.get_attachment_content(token, document_id, attachment_id)

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
            logger.warning(f"Ошибка парсинга файла {file_name}: {parse_err}")
            text_content = f"Не удалось извлечь текст из файла {file_name}."

        return {
            "status": "success",
            "file_name": file_name,
            "content": text_content[:25000],
            "is_truncated": len(text_content) > 25000
        }

    except Exception as e:
        logger.error(f"Ошибка в doc_get_file_content: {e}", exc_info=True)
        return {"error": f"Техническая ошибка при получении файла: {str(e)}"}

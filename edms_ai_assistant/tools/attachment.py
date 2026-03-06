from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.clients.attachment_client import EdmsAttachmentClient
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.services.file_processor import FileProcessorService

logger = logging.getLogger(__name__)

_SUPPORTED_TEXT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".rtf",
    ".odt",
    ".md",
}
_SUPPORTED_TABLE_EXTENSIONS = {".xlsx", ".xls", ".csv"}
_ALL_SUPPORTED = _SUPPORTED_TEXT_EXTENSIONS | _SUPPORTED_TABLE_EXTENSIONS


class AttachmentFetchInput(BaseModel):
    """Input schema for the doc_get_file_content tool.

    Attributes:
        document_id: EDMS document UUID.
        token: User bearer token.
        attachment_id: Specific attachment UUID; if omitted the first attachment is used.
        analysis_mode: Extraction mode – text | tables | metadata | full.
        summary_type: Summarisation format to pass downstream (extractive | abstractive | thesis).
    """

    document_id: str = Field(..., description="UUID документа")
    token: str = Field(..., description="Токен авторизации пользователя")
    attachment_id: Optional[str] = Field(
        None,
        description="UUID вложения. Если не указан — берётся первое вложение.",
    )
    analysis_mode: Optional[str] = Field(
        "text",
        description=(
            "Режим анализа: 'text' (текст), 'tables' (таблицы из Excel), "
            "'metadata' (метаданные), 'full' (всё сразу)"
        ),
    )
    summary_type: Optional[str] = Field(
        None,
        description=(
            "Тип суммаризации для передачи следующему инструменту: "
            "extractive | abstractive | thesis"
        ),
    )


def _build_attachment_meta(attachment: Any) -> Dict[str, Any]:
    """Build a metadata dict from an attachment domain object.

    Args:
        attachment: Attachment object from DocumentDto (has name, size, id etc.).

    Returns:
        Dict with human-readable attachment metadata.
    """
    name = (
        getattr(attachment, "name", None)
        or getattr(attachment, "fileName", None)
        or "unknown"
    )
    size_bytes = getattr(attachment, "size", None) or 0
    upload_date = getattr(attachment, "uploadDate", None)
    signs = getattr(attachment, "signs", None) or []

    att_type_obj = getattr(attachment, "attachmentDocumentType", None)
    att_type = None
    if att_type_obj:
        att_type = getattr(att_type_obj, "name", None) or str(att_type_obj)

    formatted_date: Optional[str] = None
    if upload_date:
        try:
            if hasattr(upload_date, "strftime"):
                formatted_date = upload_date.strftime("%d.%m.%Y %H:%M")
            else:
                s = str(upload_date)
                formatted_date = s[:16] if len(s) >= 16 else s
        except Exception:
            formatted_date = str(upload_date)

    return {
        "название": name,
        "тип_вложения": att_type,
        "размер_кб": round(size_bytes / 1024, 2) if size_bytes else 0,
        "дата_загрузки": formatted_date,
        "есть_эцп": bool(signs),
        "id": str(getattr(attachment, "id", "")) or None,
    }


@tool("doc_get_file_content", args_schema=AttachmentFetchInput)
async def doc_get_file_content(
    document_id: str,
    token: str,
    attachment_id: Optional[str] = None,
    analysis_mode: str = "text",
    summary_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract and analyse the content of a document attachment from EDMS.

    Analysis modes:
    - text     : Extract plain text (default, works for PDF/DOCX/TXT/RTF).
    - tables   : Extract tabular data from Excel files.
    - metadata : Return only file metadata without downloading.
    - full     : Return text + tables + metadata together.

    The ``summary_type`` parameter is forwarded in the response so the next
    tool in the pipeline (doc_summarize_text) can use it without the agent
    needing to rediscover it.

    Args:
        document_id: EDMS document UUID.
        token: User bearer JWT token.
        attachment_id: Specific attachment UUID to read.
        analysis_mode: One of text | tables | metadata | full.
        summary_type: Summarisation format to forward downstream.

    Returns:
        Dict with status, content/metadata, and forwarded summary_type.
    """
    logger.info(
        "doc_get_file_content called",
        extra={
            "document_id": document_id,
            "attachment_id": attachment_id,
            "analysis_mode": analysis_mode,
            "summary_type": summary_type,
        },
    )

    try:
        async with DocumentClient() as doc_client:
            raw_data = await doc_client.get_document_metadata(token, document_id)
            doc = DocumentDto.model_validate(raw_data)
            attachments = doc.attachmentDocument or []

        if not attachments:
            return {
                "status": "info",
                "message": "В документе нет вложений.",
                "summary_type": summary_type,
            }

        if attachment_id:
            target = next(
                (a for a in attachments if str(getattr(a, "id", "")) == attachment_id),
                None,
            )
            if target is None:
                names = [
                    getattr(a, "name", None)
                    or getattr(a, "fileName", None)
                    or str(getattr(a, "id", ""))
                    for a in attachments
                ]
                return {
                    "status": "error",
                    "message": (
                        f"Вложение с ID '{attachment_id}' не найдено. "
                        f"Доступные вложения: {', '.join(names)}"
                    ),
                    "available_attachments": [
                        {
                            "id": str(getattr(a, "id", "")),
                            "name": getattr(a, "name", None)
                            or getattr(a, "fileName", None),
                        }
                        for a in attachments
                    ],
                    "summary_type": summary_type,
                }
        else:
            target = attachments[0]
            attachment_id = str(getattr(target, "id", ""))
            logger.info("Auto-selected first attachment: %s", attachment_id)

        file_info = _build_attachment_meta(target)
        file_name = file_info["название"]
        suffix = os.path.splitext(file_name)[1].lower() if file_name else ".tmp"

        if analysis_mode == "metadata":
            return {
                "status": "success",
                "mode": "metadata",
                "data": file_info,
                "summary_type": summary_type,
            }

        async with EdmsAttachmentClient() as attach_client:
            content_bytes = await attach_client.get_attachment_content(
                token, document_id, attachment_id
            )

        if not content_bytes:
            return {
                "status": "error",
                "message": f"Файл '{file_name}' пустой или недоступен для скачивания.",
                "summary_type": summary_type,
            }

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content_bytes)
                tmp_path = tmp.name

            try:
                if analysis_mode == "full":
                    structured = await FileProcessorService.extract_structured_data(
                        tmp_path
                    )
                    return {
                        "status": "success",
                        "mode": "full",
                        "file_info": file_info,
                        "content": structured.get("text", "")[:15000],
                        "is_truncated": len(structured.get("text", "")) > 15000,
                        "total_chars": len(structured.get("text", "")),
                        "metadata": structured.get("metadata"),
                        "stats": structured.get("stats"),
                        "tables": structured.get("tables"),
                        "summary_type": summary_type,
                    }

                elif analysis_mode == "tables":
                    if suffix not in _SUPPORTED_TABLE_EXTENSIONS:
                        return {
                            "status": "error",
                            "message": (
                                f"Режим 'tables' поддерживается только для Excel/CSV файлов. "
                                f"Текущий формат: '{suffix}'. "
                                f"Для текстовых документов используй режим 'text'."
                            ),
                            "summary_type": summary_type,
                        }
                    structured = await FileProcessorService.extract_structured_data(
                        tmp_path
                    )
                    return {
                        "status": "success",
                        "mode": "tables",
                        "file_info": file_info,
                        "tables": structured.get("tables", []),
                        "tables_count": len(structured.get("tables", [])),
                        "summary_type": summary_type,
                    }

                else:
                    if suffix not in _ALL_SUPPORTED:
                        return {
                            "status": "warning",
                            "message": (
                                f"Формат '{suffix}' не поддерживается для извлечения текста. "
                                f"Поддерживаемые форматы: {', '.join(sorted(_ALL_SUPPORTED))}. "
                                f"Метаданные файла возвращены."
                            ),
                            "file_info": file_info,
                            "summary_type": summary_type,
                        }

                    text_content = await FileProcessorService.extract_text_async(
                        tmp_path
                    )

                    if not text_content or text_content.startswith("Ошибка:"):
                        return {
                            "status": "error",
                            "message": (
                                f"Не удалось извлечь текст из '{file_name}'. "
                                f"Возможно, файл является сканом без текстового слоя. "
                                f"Подробности: {text_content}"
                            ),
                            "file_info": file_info,
                            "summary_type": summary_type,
                        }

                    return {
                        "status": "success",
                        "mode": "text",
                        "file_info": file_info,
                        "content": text_content[:15000],
                        "is_truncated": len(text_content) > 15000,
                        "total_chars": len(text_content),
                        "summary_type": summary_type,
                    }

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as parse_err:
            logger.warning("File processing error for '%s': %s", file_name, parse_err)
            return {
                "status": "error",
                "message": (
                    f"Не удалось обработать файл '{file_name}'. "
                    f"Возможно, файл повреждён или защищён паролем. "
                    f"Ошибка: {parse_err}"
                ),
                "file_info": file_info,
                "summary_type": summary_type,
            }

    except Exception as exc:
        logger.error("doc_get_file_content failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": f"Не удалось прочитать файл: {exc}",
            "summary_type": summary_type,
        }

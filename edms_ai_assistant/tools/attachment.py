import logging
import os
import tempfile
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.clients.attachment_client import EdmsAttachmentClient
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.services.file_processor import FileProcessorService
from edms_ai_assistant.services.nlp_service import EDMSNaturalLanguageService

logger = logging.getLogger(__name__)


class AttachmentFetchInput(BaseModel):
    document_id: str = Field(..., description="UUID документа")
    token: str = Field(..., description="Токен авторизации пользователя")
    attachment_id: Optional[str] = Field(
        None,
        description="UUID конкретного файла. Если не указан, будет выбран первый доступный.",
    )
    analysis_mode: Optional[str] = Field(
        "text",
        description=(
            "Режим анализа: 'text' (текст), 'tables' (таблицы из Excel), "
            "'metadata' (метаданные), 'full' (все сразу)"
        ),
    )


@tool("doc_get_file_content", args_schema=AttachmentFetchInput)
async def doc_get_file_content(
    document_id: str,
    token: str,
    attachment_id: Optional[str] = None,
    analysis_mode: str = "text",
) -> Dict[str, Any]:
    """
    Извлекает и анализирует содержимое файла документа.

    Режимы анализа:
    - text: Извлекает текст (по умолчанию)
    - tables: Извлекает таблицы из Excel
    - metadata: Возвращает только метаданные
    - full: Возвращает всё (текст + таблицы + метаданные)

    Args:
        document_id: UUID документа
        token: Токен авторизации
        attachment_id: UUID файла (опционально)
        analysis_mode: Режим анализа

    Returns:
        Dict с извлеченными данными
    """
    try:
        # Получаем метаданные документа
        async with DocumentClient() as doc_client:
            raw_data = await doc_client.get_document_metadata(token, document_id)
            doc = DocumentDto.model_validate(raw_data)
            attachments = doc.attachmentDocument

        if not attachments:
            return {
                "status": "info",
                "message": "В документе отсутствуют вложения.",
            }

        nlp = EDMSNaturalLanguageService()

        # Если ID не указан, берем первое вложение
        if not attachment_id:
            target = attachments[0]
            attachment_id = str(target.id)
            logger.info(f"Auto-selected first attachment: {target.name}")
        else:
            target = next((a for a in attachments if str(a.id) == attachment_id), None)
            if not target:
                return {
                    "status": "error",
                    "message": f"Файл с ID {attachment_id} не найден.",
                }

        # Анализ метаданных файла
        file_info = nlp.analyze_attachment_meta(target)

        # Режим "только метаданные"
        if analysis_mode == "metadata":
            return {
                "status": "success",
                "mode": "metadata",
                "data": file_info,
            }

        # Скачиваем файл
        async with EdmsAttachmentClient() as attach_client:
            content_bytes = await attach_client.get_attachment_content(
                token, document_id, attachment_id
            )

        suffix = os.path.splitext(target.name)[1] or ".tmp"

        # Сохраняем во временный файл
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content_bytes)
                tmp_path = tmp.name

            try:
                if analysis_mode == "full":
                    # Извлекаем всё: текст + таблицы + метаданные
                    structured_data = (
                        await FileProcessorService.extract_structured_data(tmp_path)
                    )

                    return {
                        "status": "success",
                        "mode": "full",
                        "file_info": file_info,
                        "text": structured_data["text"][:15000],
                        "is_truncated": len(structured_data["text"]) > 15000,
                        "total_chars": len(structured_data["text"]),
                        "metadata": structured_data["metadata"],
                        "stats": structured_data["stats"],
                        "tables": structured_data.get("tables"),
                    }

                elif analysis_mode == "tables":
                    # Только таблицы (для Excel)
                    if suffix.lower() not in [".xlsx", ".xls"]:
                        return {
                            "status": "error",
                            "message": f"Режим 'tables' поддерживается только для Excel файлов. Текущий формат: {suffix}",
                        }

                    structured_data = (
                        await FileProcessorService.extract_structured_data(tmp_path)
                    )

                    return {
                        "status": "success",
                        "mode": "tables",
                        "file_info": file_info,
                        "tables": structured_data.get("tables", []),
                        "tables_count": len(structured_data.get("tables", [])),
                    }

                else:
                    text_content = await FileProcessorService.extract_text_async(
                        tmp_path
                    )

                    if text_content.startswith("Ошибка:"):
                        return {
                            "status": "error",
                            "message": text_content,
                        }

                    return {
                        "status": "success",
                        "mode": "text",
                        "file_info": file_info,
                        "content": text_content[:15000],
                        "is_truncated": len(text_content) > 15000,
                        "total_chars": len(text_content),
                    }

            finally:
                # Удаляем временный файл
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        except Exception as parse_err:
            logger.warning(f"Ошибка обработки {target.name}: {parse_err}")
            return {
                "status": "error",
                "message": (
                    f"Не удалось обработать файл {target.name}. "
                    f"Возможно, файл поврежден или является сканом без текстового слоя. "
                    f"Ошибка: {str(parse_err)}"
                ),
            }

    except Exception as e:
        logger.error(f"Ошибка doc_get_file_content: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Не удалось прочитать файл: {str(e)}",
        }

# edms_ai_assistant\tools\local_file_tool.py
import os
from typing import Dict, Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from edms_ai_assistant.services.file_processor import FileProcessorService, logger
from edms_ai_assistant.services.nlp_service import EDMSNaturalLanguageService


class LocalFileInput(BaseModel):
    file_path: str = Field(
        ...,
        description="ПОЛНЫЙ путь к локальному файлу. Возьми его ИЗ ТЕКСТА ЗАПРОСА в блоке [ДОСТУПЕН ЛОКАЛЬНЫЙ ФАЙЛ].",
    )


@tool("read_local_file_content", args_schema=LocalFileInput)
def read_local_file_content(file_path: str) -> Dict[str, Any]:
    """
    Извлекает текст из локального файла (PDF, DOCX, TXT).
    Используй для анализа документов, загруженных пользователем напрямую.
    """
    logger.info(f"[NLP-TOOL] Анализ локального файла: {file_path}")

    # Защита от передачи плейсхолдера
    if "file_path" in file_path or not file_path.strip():
        return {
            "status": "error",
            "message": "Передан некорректный путь. Найдите реальный путь в блоке [ДОСТУПЕН ЛОКАЛЬНЫЙ ФАЙЛ].",
        }

    if not os.path.exists(file_path):
        return {
            "status": "error",
            "message": f"Файл не найден по указанному пути. Проверьте доступность файла.",
        }

    try:
        nlp = EDMSNaturalLanguageService()
        file_meta = nlp.analyze_local_file(file_path)
        text_content = FileProcessorService.extract_text(file_path)

        limit = 15000
        truncated_content = text_content[:limit]

        return {
            "status": "success",
            "meta": file_meta,
            "content": truncated_content,
            "is_truncated": len(text_content) > limit,
            "total_chars": len(text_content),
        }

    except Exception as e:
        logger.error(f"[NLP-TOOL] Ошибка обработки файла: {e}")
        return {
            "status": "error",
            "message": f"Не удалось извлечь данные из файла: {str(e)}",
        }

"""
EDMS AI Assistant - Local File Tool.

Инструмент для извлечения и анализа локальных файлов.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.services.file_processor import FileProcessorService
from edms_ai_assistant.services.nlp_service import EDMSNaturalLanguageService

logger = logging.getLogger(__name__)


class LocalFileInput(BaseModel):
    """Валидированная схема входных данных для чтения локального файла."""

    file_path: str = Field(
        ...,
        description=(
            "ПОЛНЫЙ абсолютный путь к локальному файлу. "
            "Возьми его ИЗ ТЕКСТА ЗАПРОСА в блоке [ДОСТУПЕН ЛОКАЛЬНЫЙ ФАЙЛ]."
        ),
        min_length=1,
        max_length=500,
    )

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """
        Валидирует путь к файлу.

        Проверки:
        - Не является плейсхолдером
        - Не пустая строка
        - Валидный путь

        Raises:
            ValueError: Если путь невалиден
        """
        cleaned = v.strip()

        if not cleaned:
            raise ValueError("Путь к файлу не может быть пустым")

        if "file_path" in cleaned.lower():
            raise ValueError(
                "Передан плейсхолдер вместо реального пути. "
                "Найдите путь в блоке [ДОСТУПЕН ЛОКАЛЬНЫЙ ФАЙЛ]."
            )

        return cleaned


@tool("read_local_file_content", args_schema=LocalFileInput)
def read_local_file_content(file_path: str) -> Dict[str, Any]:
    """
    Извлекает текст и метаданные из локального файла.

    Поддерживаемые форматы:
    - PDF (.pdf)
    - Microsoft Word (.docx, .doc)
    - Plain Text (.txt)

    Workflow:
    1. Валидация пути к файлу
    2. Извлечение метаданных через NLP service
    3. Извлечение текстового содержимого
    4. Truncation при необходимости (15K chars limit)

    Args:
        file_path: Абсолютный путь к файлу

    Returns:
        Dict с ключами:
        - status: "success" | "error"
        - meta: Метаданные файла (размер, тип, etc.)
        - content: Извлеченный текст
        - is_truncated: bool (был ли обрезан текст)
        - total_chars: int (полная длина текста)
        - message: Сообщение об ошибке (для status="error")

    Examples:
         result = read_local_file_content("/path/to/document.pdf")
         # {
         #   "status": "success",
         #   "meta": {"extension": ".pdf", "size_bytes": 12345, ...},
         #   "content": "Extracted text...",
         #   "is_truncated": False,
         #   "total_chars": 5000
         # }

         result = read_local_file_content("")
         # {"status": "error", "message": "Путь к файлу не может быть пустым"}
    """
    logger.info(
        "Processing local file",
        extra={"file_path": file_path},
    )

    try:
        validation_error = _validate_file_access(file_path)
        if validation_error:
            return validation_error

        nlp = EDMSNaturalLanguageService()
        file_meta = nlp.analyze_local_file(file_path)

        text_content = FileProcessorService.extract_text(file_path)

        if text_content.startswith("Ошибка:") or text_content.startswith(
            "Формат файла"
        ):
            logger.error(
                "File extraction failed",
                extra={"file_path": file_path, "error": text_content},
            )
            return {
                "status": "error",
                "message": text_content,
            }

        truncated_content, is_truncated = _truncate_content(text_content)

        logger.info(
            "File processed successfully",
            extra={
                "file_path": file_path,
                "total_chars": len(text_content),
                "is_truncated": is_truncated,
            },
        )

        return {
            "status": "success",
            "meta": file_meta,
            "content": truncated_content,
            "is_truncated": is_truncated,
            "total_chars": len(text_content),
        }

    except ValueError as e:
        logger.warning(
            f"Validation error: {e}",
            extra={"file_path": file_path},
        )
        return {
            "status": "error",
            "message": str(e),
        }
    except Exception as e:
        logger.error(
            f"File processing error: {e}",
            exc_info=True,
            extra={"file_path": file_path},
        )
        return {
            "status": "error",
            "message": f"Не удалось извлечь данные из файла: {str(e)}",
        }


def _validate_file_access(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Валидирует доступ к файлу перед обработкой.

    Args:
        file_path: Путь к файлу

    Returns:
        Dict с ошибкой если валидация не прошла, None если все ок
    """
    path = Path(file_path)

    if not path.exists():
        logger.error(
            "File not found",
            extra={"file_path": file_path},
        )
        return {
            "status": "error",
            "message": "Файл не найден по указанному пути. Проверьте доступность файла.",
        }

    if not path.is_file():
        logger.error(
            "Path is not a file",
            extra={"file_path": file_path},
        )
        return {
            "status": "error",
            "message": "Указанный путь не является файлом.",
        }

    return None


def _truncate_content(content: str, max_length: int = 15000) -> tuple[str, bool]:
    """
    Обрезает контент до максимальной длины.

    Args:
        content: Исходный текст
        max_length: Максимальная длина (default: 15000)

    Returns:
        Tuple[str, bool]: (truncated_content, is_truncated)
    """
    if len(content) <= max_length:
        return content, False

    logger.debug(f"Content truncated: {len(content)} → {max_length} chars")

    return content[:max_length], True

# edms_ai_assistant\tools\local_file_tool.py
import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from edms_ai_assistant.services.file_processor import FileProcessorService, logger


class LocalFileInput(BaseModel):
    file_path: str = Field(
        ...,
        description="ПОЛНЫЙ путь к локальному файлу. Возьми его ИЗ ТЕКСТА ЗАПРОСА в блоке [ДОСТУПЕН ЛОКАЛЬНЫЙ ФАЙЛ]."
    )


@tool("read_local_file_content", args_schema=LocalFileInput)
def read_local_file_content(file_path: str):
    """
    Инструмент для извлечения текстового содержимого из локально загруженного файла (PDF, DOCX, TXT).
    Используй его, когда пользователь спрашивает о 'загруженном файле' или 'этом файле'.
    """
    logger.info(f"[TOOL] Вызов read_local_file_content для пути: {file_path}")

    if file_path.strip() == "file_path" or file_path.strip() == "":
        return (
            "ОШИБКА: Вы передали техническое имя 'file_path'. "
            "Пожалуйста, посмотрите в историю сообщений, найдите строку '[ПУТЬ К ФАЙЛУ]' "
            "и передайте её значение (реальный путь на диске)."
        )

    if not os.path.exists(file_path):
        logger.error(f"[TOOL] Файл не найден. Текущая директория: {os.getcwd()}")
        return f"ОШИБКА: Файл не найден по пути: {file_path}. Убедитесь, что путь указан верно."

    try:
        return FileProcessorService.extract_text(file_path)
    except Exception as e:
        logger.error(f"[TOOL] Ошибка при чтении файла: {e}")
        return f"Произошла ошибка при чтении содержимого файла: {str(e)}"

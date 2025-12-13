import io
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# --- Импорты ---
try:
    import docx2txt
    from PyPDF2 import PdfReader  # В старых версиях PyPDF2 это может быть PdfFileReader
except ImportError:
    # ⚠️ Важно: Установите необходимые пакеты: pip install docx2txt pypdf2
    logger.warning("Не удалось импортировать docx2txt или PyPDF2. Поддерживаются только .txt")
    docx2txt = None
    PdfReader = None


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> Optional[str]:
    """
    Извлекает текст из байтов файла (поддержка .docx, .pdf, .txt).

    :param file_bytes: Содержимое файла в байтах.
    :param filename: Имя файла (для определения формата).
    :return: Извлеченный текст или None в случае ошибки/неподдерживаемого формата.
    """
    if not file_bytes:
        logger.warning(f"Пустые байты переданы для файла: {filename}")
        return None

    try:
        file_stream = io.BytesIO(file_bytes)

        # --- Корректное извлечение расширения ---
        # Используем os.path.splitext для надежного извлечения расширения
        # и приводим его к нижнему регистру.

        # NOTE: os.path.splitext работает с именами файлов, содержащими кириллицу.
        _, ext_with_dot = os.path.splitext(filename)
        ext = ext_with_dot.lstrip('.').lower()

        if not ext:
            logger.warning(f"Не удалось определить расширение для файла: {filename}")
            return None

        # --- PDF ---
        if ext == "pdf":
            if PdfReader is None:
                logger.error("Ошибка: PyPDF2 не импортирован.")
                return None

            reader = PdfReader(file_stream)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text.strip()

        # --- DOCX ---
        elif ext == "docx":
            if docx2txt is None:
                logger.error("Ошибка: docx2txt не импортирован.")
                return None

            # docx2txt.process ожидает объект, похожий на файл (file_stream)
            return docx2txt.process(file_stream).strip()

        # --- TXT ---
        elif ext == "txt":
            # Используем игнорирование ошибок для максимальной совместимости
            return file_bytes.decode("utf-8", errors="ignore").strip()

        # --- НЕПОДДЕРЖИВАЕМЫЙ ---
        else:
            logger.warning(f"Неподдерживаемый формат файла: .{ext} (файл: {filename}).")
            return None

    except Exception as e:
        # Ловим общие ошибки парсинга
        logger.error(f"Критическая ошибка извлечения текста из {filename} ({ext}): {type(e).__name__}: {e}")
        return None
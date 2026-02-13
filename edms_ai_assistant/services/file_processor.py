"""
EDMS AI Assistant - Enhanced File Processor Service.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)

logger = logging.getLogger(__name__)

# Thread pool для CPU-intensive операций
_executor = ThreadPoolExecutor(max_workers=4)


class FileProcessorService:
    """
    Сервис для извлечения текста из файлов с поддержкой множества форматов.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".txt": lambda path: TextLoader(path, encoding="utf-8"),
        ".xlsx": "excel_modern",
        ".xls": "excel_legacy",
    }

    @classmethod
    async def extract_text_async(cls, file_path: str) -> str:
        """
        Асинхронное извлечение текста из файла.

        Использует thread pool для CPU-intensive операций,
        освобождая event loop для других задач.

        Args:
            file_path: Путь к файлу

        Returns:
            Извлеченный текст
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, cls.extract_text, file_path)

    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """
        Извлекает текст из файла (синхронная версия для обратной совместимости).

        Args:
            file_path: Абсолютный путь к файлу

        Returns:
            Извлеченный текст или сообщение об ошибке
        """
        path = Path(file_path)

        if not path.exists():
            error_msg = f"Файл не найден по пути: {file_path}"
            logger.error(error_msg, extra={"file_path": file_path})
            return f"Ошибка: {error_msg}"

        ext = path.suffix.lower()

        if ext not in cls.SUPPORTED_EXTENSIONS:
            warning_msg = f"Формат файла {ext} пока не поддерживается для анализа."
            logger.warning(
                f"Unsupported file format: {ext}",
                extra={"file_path": file_path, "extension": ext},
            )
            return warning_msg

        try:
            # Обработка Excel файлов
            if ext in [".xlsx", ".xls"]:
                return cls._extract_from_excel(file_path, ext)

            # Стандартная обработка через LangChain
            loader_class = cls.SUPPORTED_EXTENSIONS[ext]
            loader = loader_class(file_path)

            logger.debug(
                f"Loading file with {loader_class.__name__}",
                extra={"file_path": file_path, "extension": ext},
            )

            docs = loader.load()

            if not docs:
                logger.warning(
                    "File loaded but no documents found",
                    extra={"file_path": file_path},
                )
                return "Файл успешно прочитан, но в нем не обнаружено текстового содержимого."

            full_text = "\n\n".join([doc.page_content for doc in docs]).strip()

            if not full_text:
                logger.warning(
                    "Documents loaded but text is empty",
                    extra={"file_path": file_path, "docs_count": len(docs)},
                )
                return "Содержимое файла пусто или состоит из изображений без текстового слоя."

            logger.info(
                "Text extracted successfully",
                extra={
                    "file_path": file_path,
                    "extension": ext,
                    "text_length": len(full_text),
                    "pages_count": len(docs),
                },
            )

            return full_text

        except Exception as e:
            error_msg = f"Произошла техническая ошибка при чтении файла {ext}: {str(e)}"
            logger.error(
                f"File parsing error: {e}",
                exc_info=True,
                extra={"file_path": file_path, "extension": ext},
            )
            return error_msg

    @classmethod
    def _extract_from_excel(cls, file_path: str, ext: str) -> str:
        """
        Извлекает текст из Excel файлов с сохранением структуры таблиц.

        Args:
            file_path: Путь к Excel файлу
            ext: Расширение (.xlsx или .xls)

        Returns:
            Форматированный текст с таблицами
        """
        try:
            if ext == ".xlsx":
                import openpyxl

                wb = openpyxl.load_workbook(file_path, data_only=True)
            else:
                import xlrd

                wb = xlrd.open_workbook(file_path)

            extracted_text = []

            if ext == ".xlsx":
                for sheet_name in wb.sheetnames:
                    sheet = wb[sheet_name]
                    extracted_text.append(f"\n{'='*50}\nЛИСТ: {sheet_name}\n{'='*50}\n")

                    # Извлекаем данные с сохранением структуры
                    for row in sheet.iter_rows(values_only=True):
                        # Пропускаем полностью пустые строки
                        if any(cell is not None for cell in row):
                            row_text = " | ".join(
                                str(cell) if cell is not None else "" for cell in row
                            )
                            extracted_text.append(row_text)
            else:
                for sheet_idx in range(wb.nsheets):
                    sheet = wb.sheet_by_index(sheet_idx)
                    extracted_text.append(f"\n{'='*50}\nЛИСТ: {sheet.name}\n{'='*50}\n")

                    for row_idx in range(sheet.nrows):
                        row = sheet.row_values(row_idx)
                        if any(cell for cell in row):
                            row_text = " | ".join(
                                str(cell) if cell else "" for cell in row
                            )
                            extracted_text.append(row_text)

            result = "\n".join(extracted_text).strip()

            logger.info(
                "Excel file extracted successfully",
                extra={
                    "file_path": file_path,
                    "extension": ext,
                    "text_length": len(result),
                    "sheets_count": (
                        len(wb.sheetnames) if ext == ".xlsx" else wb.nsheets
                    ),
                },
            )

            return result

        except ImportError as e:
            logger.error(f"Missing dependency for Excel: {e}")
            return (
                "Ошибка: Для обработки Excel файлов требуется установить библиотеки "
                "'openpyxl' (для .xlsx) или 'xlrd' (для .xls). "
                "Выполните: pip install openpyxl xlrd --break-system-packages"
            )
        except Exception as e:
            logger.error(f"Excel extraction error: {e}", exc_info=True)
            return f"Ошибка при чтении Excel файла: {str(e)}"

    @classmethod
    async def extract_structured_data(cls, file_path: str) -> Dict[str, Any]:
        """
        Извлекает структурированные данные из файла (метаданные + контент).

        Args:
            file_path: Путь к файлу

        Returns:
            Dict с ключами:
            - text: Извлеченный текст
            - metadata: Метаданные (страницы, размер, формат)
            - tables: Список таблиц (для Excel)
            - stats: Статистика (слова, символы, числа)
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        text = await cls.extract_text_async(file_path)

        stats = {
            "chars": len(text),
            "words": len(text.split()),
            "lines": text.count("\n"),
            "digits": len([c for c in text if c.isdigit()]),
        }

        # Метаданные файла
        file_stat = path.stat()
        metadata = {
            "filename": path.name,
            "extension": ext,
            "size_bytes": file_stat.st_size,
            "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
        }

        result = {
            "text": text,
            "metadata": metadata,
            "stats": stats,
            "tables": None,
        }

        if ext in [".xlsx", ".xls"]:
            result["tables"] = await cls._extract_excel_tables(file_path, ext)

        return result

    @classmethod
    async def _extract_excel_tables(
        cls, file_path: str, ext: str
    ) -> List[Dict[str, Any]]:
        """
        Извлекает таблицы из Excel как структурированные данные.

        Returns:
            Список таблиц, где каждая таблица - это Dict с ключами:
            - sheet_name: Название листа
            - headers: Заголовки столбцов
            - data: Данные (список списков)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, cls._extract_tables_sync, file_path, ext
        )

    @classmethod
    def _extract_tables_sync(cls, file_path: str, ext: str) -> List[Dict[str, Any]]:
        """Синхронная версия извлечения таблиц."""
        try:
            if ext == ".xlsx":
                import openpyxl

                wb = openpyxl.load_workbook(file_path, data_only=True)
            else:
                import xlrd

                wb = xlrd.open_workbook(file_path)

            tables = []

            if ext == ".xlsx":
                for sheet_name in wb.sheetnames:
                    sheet = wb[sheet_name]
                    data = []

                    for row in sheet.iter_rows(values_only=True):
                        if any(cell is not None for cell in row):
                            data.append(list(row))

                    if data:
                        tables.append(
                            {
                                "sheet_name": sheet_name,
                                "headers": data[0] if data else [],
                                "data": data[1:] if len(data) > 1 else [],
                                "rows_count": len(data) - 1,
                            }
                        )
            else:
                for sheet_idx in range(wb.nsheets):
                    sheet = wb.sheet_by_index(sheet_idx)
                    data = []

                    for row_idx in range(sheet.nrows):
                        row = sheet.row_values(row_idx)
                        if any(cell for cell in row):
                            data.append(row)

                    if data:
                        tables.append(
                            {
                                "sheet_name": sheet.name,
                                "headers": data[0] if data else [],
                                "data": data[1:] if len(data) > 1 else [],
                                "rows_count": len(data) - 1,
                            }
                        )

            return tables

        except Exception as e:
            logger.error(f"Table extraction error: {e}", exc_info=True)
            return []

    @classmethod
    def validate_file_path(cls, file_path: str) -> Optional[str]:
        """Валидирует путь к файлу перед обработкой."""
        if not file_path or not file_path.strip():
            return "Путь к файлу не может быть пустым"

        path = Path(file_path)

        if not path.exists():
            return f"Файл не найден: {file_path}"

        if not path.is_file():
            return f"Указанный путь не является файлом: {file_path}"

        ext = path.suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            supported = ", ".join(cls.SUPPORTED_EXTENSIONS.keys())
            return f"Неподдерживаемый формат {ext}. Поддерживаются: {supported}"

        return None

    @classmethod
    def get_file_info(cls, file_path: str) -> dict:
        """Получает метаданные файла без извлечения содержимого."""
        path = Path(file_path)

        if not path.exists():
            return {
                "exists": False,
                "error": f"Файл не найден: {file_path}",
            }

        try:
            stat = path.stat()

            return {
                "exists": True,
                "name": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "is_supported": path.suffix.lower() in cls.SUPPORTED_EXTENSIONS,
                "absolute_path": str(path.absolute()),
            }

        except Exception as e:
            logger.error(
                f"Failed to get file info: {e}",
                exc_info=True,
                extra={"file_path": file_path},
            )
            return {
                "exists": True,
                "error": f"Не удалось получить информацию о файле: {str(e)}",
            }

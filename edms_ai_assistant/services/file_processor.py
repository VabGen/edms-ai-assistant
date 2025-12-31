import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

logger = logging.getLogger(__name__)


class FileProcessorService:
    @staticmethod
    def extract_text(file_path: str) -> str:
        """Извлекает текст из файла в зависимости от его расширения."""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Файл не найден по пути: {file_path}")
            return f"Ошибка: Файл не найден по пути {file_path}"

        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                logger.warning(f"Попытка парсинга неподдерживаемого формата: {ext}")
                return (
                    f"Формат файла {ext} пока не поддерживается для глубокого анализа."
                )

            docs = loader.load()

            if not docs:
                return "Файл успешно прочитан, но в нем не обнаружено текстового содержимого."

            full_text = "\n\n".join([doc.page_content for doc in docs]).strip()

            if not full_text:
                return "Содержимое файла пусто или состоит из изображений без текстового слоя."

            return full_text

        except Exception as e:
            logger.error(
                f"Ошибка при парсинге файла {ext} ({file_path}): {e}", exc_info=True
            )
            return f"Произошла техническая ошибка при чтении файла {ext}: {str(e)}"

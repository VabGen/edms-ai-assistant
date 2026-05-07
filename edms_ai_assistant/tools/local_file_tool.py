from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.services.file_processor import FileProcessorService

logger = logging.getLogger(__name__)

# Лимит для LLM (120k символов ~ 30k-40k токенов — безопасный размер для сохранения внимания модели)
_MAX_CONTENT_CHARS = 120_000
_PAGE_MARKER_PATTERN = re.compile(r"--- Страница (\d+) ---")

_ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".rtf", ".odt",
    ".md", ".xlsx", ".xls", ".csv",
}


class LocalFileInput(BaseModel):
    """Validated input for read_local_file_content."""
    file_path: str = Field(
        ...,
        description="ПОЛНЫЙ абсолютный путь к локальному файлу.",
        min_length=1, max_length=500,
    )
    target_pages: list[int] | None = Field(
        default=None,
        description="Список номеров страниц для извлечения (нумерация с 1). Используй, если знаешь нужные страницы.",
    )
    search_keywords: list[str] | None = Field(
        default=None,
        description="Ключевые слова для поиска. Инструмент вернет только те страницы, где они встречаются.",
    )

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Путь к файлу не может быть пустым")
        if "file_path" in cleaned.lower() or "<" in cleaned or ">" in cleaned:
            raise ValueError("Передан плейсхолдер вместо реального пути.")
        return cleaned


def _split_text_to_pages(text: str) -> dict[int, str]:
    """Разбивает текст с маркерами '--- Страница N ---' в словарь."""
    splits = _PAGE_MARKER_PATTERN.split(text)
    pages = {}
    for i in range(1, len(splits), 2):
        page_num = int(splits[i])
        page_text = splits[i + 1].strip() if i + 1 < len(splits) else ""
        pages[page_num] = f"--- Страница {page_num} ---\n{page_text}"
    return pages


@tool("read_local_file_content", args_schema=LocalFileInput)
async def read_local_file_content(
        file_path: str,
        target_pages: list[int] | None = None,
        search_keywords: list[str] | None = None
) -> dict[str, Any]:
    """Extract text and metadata from a local file on disk.

    Supports Smart Truncation (Head + Mini-Index + Tail) for large docs,
    Targeted Page Extraction, and Keyword Search.
    """
    logger.info("read_local_file_content called",
                extra={"file_path": file_path, "target_pages": target_pages, "search_keywords": search_keywords})

    path = Path(file_path)
    if not path.exists():
        return {"status": "error", "message": f"Файл не найден: '{file_path}'."}

    suffix = path.suffix.lower()
    size_mb = round(path.stat().st_size / (1024 * 1024), 2)
    meta = {"имя_файла": path.name, "расширение": suffix, "размер_мб": size_mb, "путь": str(path)}

    if suffix not in _ALLOWED_EXTENSIONS:
        return {"status": "error", "message": f"Формат '{suffix}' не поддерживается.", "meta": meta}

    full_text = await FileProcessorService.extract_text_async(str(path))

    if not full_text or full_text.startswith("Ошибка:") or full_text.startswith("Формат файла"):
        return {"status": "error", "message": f"Не удалось извлечь текст: {full_text[:300]}", "meta": meta}

    total_chars = len(full_text)

    # ── Режим 1: Поиск по ключевым словам ────────────────────────────────
    if search_keywords:
        pages_dict = _split_text_to_pages(full_text)
        matched_pages = set()

        for page_num, page_text in pages_dict.items():
            for kw in search_keywords:
                if kw.lower() in page_text.lower():
                    matched_pages.add(page_num)

        if not matched_pages:
            return {
                "status": "success", "meta": meta, "content": "Ключевые слова не найдены в документе.",
                "is_truncated": False, "total_chars": total_chars, "matched_pages": []
            }

        result_text = "\n\n".join(pages_dict[p] for p in sorted(matched_pages))
        return {
            "status": "success", "meta": meta, "content": result_text,
            "is_truncated": False, "total_chars": total_chars, "matched_pages": sorted(matched_pages)
        }

    # ── Режим 2: Извлечение конкретных страниц ───────────────────────────
    if target_pages:
        pages_dict = _split_text_to_pages(full_text)
        result_text = "\n\n".join(pages_dict.get(p, f"[Страница {p} не найдена]") for p in target_pages)
        return {
            "status": "success", "meta": meta, "content": result_text,
            "is_truncated": False, "total_chars": total_chars
        }

    # ── Режим 3: Умная обрезка (Начало + Мини-Индекс + Конец) ───────────
    is_truncated = total_chars > _MAX_CONTENT_CHARS
    content = full_text

    if is_truncated:
        pages_dict = _split_text_to_pages(full_text)
        all_page_nums = sorted(pages_dict.keys())

        if not all_page_nums:
            content = full_text[:_MAX_CONTENT_CHARS] + "\n\n... [ТЕКСТ ОБРЕЗАН]"
        else:
            head_size = int(_MAX_CONTENT_CHARS * 0.5)  # 50% под начало
            tail_size = int(_MAX_CONTENT_CHARS * 0.4)  # 40% под конец
            # 10% резервируем под Мини-Индекс

            head_text = ""
            head_pages = []
            for p in all_page_nums:
                if len(head_text) + len(pages_dict[p]) > head_size:
                    break
                head_text += pages_dict[p] + "\n\n"
                head_pages.append(p)

            tail_text = ""
            tail_pages = []
            for p in reversed(all_page_nums):
                if p in head_pages:
                    break
                if len(tail_text) + len(pages_dict[p]) > tail_size:
                    break
                tail_text = pages_dict[p] + "\n\n" + tail_text
                tail_pages.insert(0, p)

            # Формируем Мини-Индекс для пропущенных страниц
            middle_pages = [p for p in all_page_nums if p not in head_pages and p not in tail_pages]
            index_text = "... [ПРОПУЩЕННЫЕ СТРАНИЦЫ. Краткое содержание пропущенных частей для навигации]:\n"

            for p in middle_pages:
                # Берем первые 100 символов текста страницы как аннотацию
                page_content_clean = pages_dict[p].replace(f"--- Страница {p} ---", "").strip()
                snippet = page_content_clean[:100].replace("\n", " ")
                index_text += f"  Стр. {p}: {snippet}...\n"

            index_text += "\nДля чтения полных пропущенных страниц используй параметры target_pages или search_keywords.\n\n"

            content = head_text + index_text + tail_text
            logger.info("Smart truncation with Mini-Index applied to %s. Head: %s, Tail: %s, Indexed: %s",
                        path.name, len(head_pages), len(tail_pages), len(middle_pages))

    return {
        "status": "success", "meta": meta, "content": content,
        "is_truncated": is_truncated, "total_chars": total_chars
    }
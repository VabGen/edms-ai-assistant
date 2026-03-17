# edms_ai_assistant/tools/file_compare_tool.py
"""
EDMS AI Assistant — File Comparison Tool.

Сравнивает локально загруженный файл с вложением документа EDMS.
"""

from __future__ import annotations

import contextlib
import difflib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.clients.attachment_client import EdmsAttachmentClient
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.services.file_processor import FileProcessorService
from edms_ai_assistant.utils.regex_utils import UUID_RE

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS: int = 20_000
_MAX_DIFF_LINES: int = 60

_PATH_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "",
        "local_file",
        "local_file_path",
        "/path/to/file",
        "path/to/file",
        "none",
        "null",
        "<local_file_path>",
        "<path>",
    }
)


class FileCompareInput(BaseModel):
    """Validated input for the doc_compare_with_local tool."""

    token: str = Field(..., description="JWT токен авторизации пользователя")
    document_id: str = Field(..., description="UUID документа в СЭД")
    local_file_path: str = Field(
        ...,
        description=(
            "Абсолютный путь к загруженному файлу. "
            "Инжектируется автоматически из контекста агента."
        ),
    )
    attachment_id: Optional[str] = Field(
        None,
        description=(
            "UUID вложения или его имя для сравнения. "
            "Если не указан — ищем по оригинальному имени локального файла. "
            "Если совпадение не найдено — возвращается список вложений для выбора."
        ),
    )
    original_filename: Optional[str] = Field(
        None,
        description=(
            "Оригинальное имя загруженного файла (например: «Шаблон обложки.docx»). "
            "Инжектируется агентом из контекста. Используется для автоматического "
            "поиска вложения по имени и для понятных сообщений пользователю."
        ),
    )

    @field_validator("local_file_path")
    @classmethod
    def reject_placeholder_path(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned or cleaned.lower() in _PATH_PLACEHOLDERS:
            raise ValueError(
                f"Получен placeholder '{v}' вместо реального пути. "
                "local_file_path инжектируется автоматически из контекста агента."
            )
        return cleaned


def _att_name(attachment: Any) -> str:
    return (
        getattr(attachment, "name", None) or getattr(attachment, "fileName", None) or ""
    )


def _att_id(attachment: Any) -> str:
    return str(getattr(attachment, "id", "") or "")


def _resolve_attachment(attachments: List[Any], hint: str) -> Optional[Any]:
    """Resolve attachment by UUID or filename hint (4-level fallback)."""
    if not hint or not attachments:
        return None

    hint_s = hint.strip()

    if UUID_RE.match(hint_s):
        found = next((a for a in attachments if _att_id(a) == hint_s), None)
        if found is not None:
            logger.debug("Attachment resolved by UUID: %s…", hint_s[:8])
            return found

    hint_lower = hint_s.lower()
    hint_stem = Path(hint_lower).stem

    for att in attachments:
        if _att_name(att).lower() == hint_lower:
            logger.info("Attachment resolved by exact name: '%s'", hint_s)
            return att

    for att in attachments:
        if Path(_att_name(att).lower()).stem == hint_stem and hint_stem:
            logger.info("Attachment resolved by stem: '%s'", hint_s)
            return att

    for att in attachments:
        att_stem = Path(_att_name(att).lower()).stem
        if hint_stem and att_stem and (hint_stem in att_stem or att_stem in hint_stem):
            logger.info(
                "Attachment resolved by partial stem: '%s' ~ '%s'", hint_stem, att_stem
            )
            return att

    return None


def _disambiguation_response(
    attachments: List[Any],
    local_filename: str,
    hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Build requires_disambiguation response consumed by _detect_interactive_status."""
    available = [
        {"id": _att_id(a), "name": _att_name(a) or "без имени"}
        for a in attachments
        if _att_id(a)
    ]
    message = (
        f"Вложение «{hint}» не найдено в документе. "
        f"Выберите вложение для сравнения с «{local_filename}»:"
        if hint
        else f"Не удалось автоматически определить вложение для сравнения "
        f"с «{local_filename}». Выберите из списка:"
    )
    return {
        "status": "requires_disambiguation",
        "message": message,
        "available_attachments": available,
    }


def _normalise(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _compute_diff(
    local_text: str,
    att_text: str,
    local_label: str,
    att_label: str,
) -> List[Dict[str, str]]:
    raw_diff = difflib.unified_diff(
        local_text.splitlines(keepends=True),
        att_text.splitlines(keepends=True),
        fromfile=local_label,
        tofile=att_label,
        lineterm="",
        n=2,
    )
    changes: List[Dict[str, str]] = []
    for line in raw_diff:
        if line.startswith(("---", "+++", "@@")):
            continue
        stripped = line[1:].strip()
        if not stripped:
            continue
        if line.startswith("+"):
            changes.append({"type": "added_in_attachment", "content": stripped})
        elif line.startswith("-"):
            changes.append({"type": "removed_from_local", "content": stripped})
    return changes[:_MAX_DIFF_LINES]


def _build_summary(
    are_identical: bool,
    similarity: float,
    local_name: str,
    att_name: str,
    local_stats: Dict[str, int],
    att_stats: Dict[str, int],
    diff_result: List[Dict[str, str]],
) -> str:
    if are_identical:
        return (
            f"Файлы идентичны по содержимому (схожесть: {similarity}%). "
            f"«{local_name}»: {local_stats['chars']} симв., "
            f"«{att_name}»: {att_stats['chars']} симв."
        )
    added = sum(1 for d in diff_result if d["type"] == "added_in_attachment")
    removed = sum(1 for d in diff_result if d["type"] == "removed_from_local")
    return (
        f"Файлы различаются (схожесть: {similarity}%). "
        f"Строк только во вложении «{att_name}»: {added}. "
        f"Строк только в «{local_name}»: {removed}. "
        f"«{local_name}»: {local_stats['chars']} симв., "
        f"«{att_name}»: {att_stats['chars']} симв."
    )


@tool("doc_compare_with_local", args_schema=FileCompareInput)
async def doc_compare_with_local(
    token: str,
    document_id: str,
    local_file_path: str,
    attachment_id: Optional[str] = None,
    original_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare a locally uploaded file with an EDMS document attachment.

    Attachment resolution (in order, first match wins):
    1. attachment_id is a valid UUID → exact ID match.
    2. attachment_id is a filename → case-insensitive name/stem lookup.
    3. attachment_id is None, original_filename provided → auto-match by original name.
    4. attachment_id is None, no original_filename → auto-match by temp file path name.
    5. No match found → requires_disambiguation with full attachment list.
       No silent random fallback ever.

    original_filename is used both for attachment auto-resolution and for
    human-readable disambiguation messages. It must be the user's original
    file name (e.g. "Шаблон обложки.docx"), NOT the internal temp file path.

    Args:
        token: JWT bearer token (injected by orchestrator).
        document_id: EDMS document UUID (injected by orchestrator).
        local_file_path: Absolute path to the uploaded local file (temp path).
        attachment_id: Attachment UUID or filename hint (optional).
        original_filename: Original user-visible file name (injected by orchestrator).

    Returns:
        Dict with status: success | requires_disambiguation | error.
    """
    local_path = Path(local_file_path)
    display_name: str = (
        original_filename.strip()
        if original_filename and original_filename.strip()
        else local_path.name
    )

    logger.info(
        "doc_compare_with_local called",
        extra={
            "document_id": document_id[:8] + "…",
            "local_file": local_file_path,
            "display_name": display_name,
            "attachment_id": attachment_id,
        },
    )

    # ── 1. Валидация локального файла ─────────────────────────────────────────
    if not local_path.exists():
        logger.error("Local file not found: %s", local_file_path)
        return {
            "status": "error",
            "message": (
                f"Загруженный файл «{display_name}» не найден на сервере. "
                "Возможно, файл был удалён — загрузите его заново."
            ),
        }

    # ── 2. Получение метаданных документа и списка вложений ───────────────────
    try:
        async with DocumentClient() as doc_client:
            raw_data = await doc_client.get_document_metadata(token, document_id)
            doc = DocumentDto.model_validate(raw_data)
            attachments: List[Any] = doc.attachmentDocument or []
    except Exception as exc:
        logger.error("Document metadata fetch failed: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": f"Не удалось получить метаданные документа: {exc}",
        }

    if not attachments:
        return {
            "status": "error",
            "message": "В документе нет вложений для сравнения.",
        }

    # ── 3. Разрешение целевого вложения ───────────────────────────────────────
    if attachment_id:
        target = _resolve_attachment(attachments, attachment_id)
        if target is None:
            logger.info(
                "attachment_id '%s' not resolved → disambiguation", attachment_id
            )
            return _disambiguation_response(
                attachments, display_name, hint=attachment_id
            )
    else:
        target = _resolve_attachment(attachments, display_name)
        if target is None and display_name != local_path.name:
            target = _resolve_attachment(attachments, local_path.name)
        if target is None:
            logger.info(
                "Auto-match failed for '%s' → disambiguation",
                display_name,
            )
            return _disambiguation_response(attachments, display_name)

    resolved_id = _att_id(target)
    resolved_name = _att_name(target) or "attachment"
    resolved_suffix = Path(resolved_name).suffix.lower() or ".tmp"

    att_doc_id: str = str(getattr(target, "documentId", None) or document_id)
    if att_doc_id and att_doc_id != document_id:
        logger.debug(
            "Attachment belongs to version document_id=%s…, context=%s…",
            att_doc_id[:8],
            document_id[:8],
        )

    logger.info("Attachment resolved: '%s' (%s…)", resolved_name, resolved_id[:8])

    # ── 4. Скачивание вложения ────────────────────────────────────────────────
    try:
        async with EdmsAttachmentClient() as att_client:
            att_bytes: bytes = await att_client.get_attachment_content(
                token, att_doc_id, resolved_id
            )
    except Exception as exc:
        logger.error("Attachment download failed '%s': %s", resolved_name, exc)
        return {
            "status": "error",
            "message": f"Ошибка скачивания вложения «{resolved_name}»: {exc}",
        }

    if not att_bytes:
        return {
            "status": "error",
            "message": (
                f"Вложение «{resolved_name}» вернуло пустой ответ — "
                "файл недоступен или удалён."
            ),
        }

    # ── 5. Извлечение текста локального файла ────────────────────────────────
    local_text_raw: str = await FileProcessorService.extract_text_async(str(local_path))
    if not local_text_raw or local_text_raw.startswith(("Ошибка:", "Формат файла")):
        return {
            "status": "error",
            "message": f"Не удалось извлечь текст из «{display_name}»: {local_text_raw}",
        }

    # ── 6. Извлечение текста вложения через temp-файл ─────────────────────────
    att_text_raw: str = ""
    try:
        with contextlib.ExitStack() as stack:
            tmp_file = stack.enter_context(
                tempfile.NamedTemporaryFile(delete=False, suffix=resolved_suffix)
            )
            tmp_path = tmp_file.name
            stack.callback(
                lambda p=tmp_path: os.unlink(p) if os.path.exists(p) else None
            )
            tmp_file.write(att_bytes)
            tmp_file.flush()
            att_text_raw = await FileProcessorService.extract_text_async(tmp_path)
    except Exception as exc:
        logger.error(
            "Text extraction from attachment '%s' failed: %s",
            resolved_name,
            exc,
            exc_info=True,
        )
        return {
            "status": "error",
            "message": f"Ошибка извлечения текста из «{resolved_name}»: {exc}",
        }

    if not att_text_raw or att_text_raw.startswith(("Ошибка:", "Формат файла")):
        return {
            "status": "error",
            "message": (
                f"Не удалось извлечь текст из вложения «{resolved_name}»: {att_text_raw}"
            ),
        }

    # ── 7. Нормализация → сравнение → diff ───────────────────────────────────
    local_text = _normalise(local_text_raw[:_MAX_TEXT_CHARS])
    att_text = _normalise(att_text_raw[:_MAX_TEXT_CHARS])

    are_identical = local_text == att_text
    similarity = round(
        difflib.SequenceMatcher(None, local_text, att_text, autojunk=False).ratio()
        * 100,
        1,
    )

    diff_result: List[Dict[str, str]] = []
    if not are_identical:
        diff_result = _compute_diff(
            local_text,
            att_text,
            local_label=f"Загруженный файл: {display_name}",
            att_label=f"Вложение СЭД: {resolved_name}",
        )

    local_stats = {"chars": len(local_text), "lines": local_text.count("\n") + 1}
    att_stats = {"chars": len(att_text), "lines": att_text.count("\n") + 1}

    summary = _build_summary(
        are_identical,
        similarity,
        display_name,
        resolved_name,
        local_stats,
        att_stats,
        diff_result,
    )

    logger.info(
        "doc_compare_with_local completed",
        extra={
            "are_identical": are_identical,
            "similarity": similarity,
            "diff_lines": len(diff_result),
        },
    )

    return {
        "status": "success",
        "are_identical": are_identical,
        "similarity_percent": similarity,
        "local_file": display_name,
        "attachment_name": resolved_name,
        "local_stats": local_stats,
        "attachment_stats": att_stats,
        "differences": diff_result,
        "summary": summary,
    }

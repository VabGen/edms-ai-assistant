from __future__ import annotations

import difflib
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.clients.attachment_client import EdmsAttachmentClient
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.services.file_processor import FileProcessorService

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS = 20000


class FileCompareInput(BaseModel):
    """Validated input for doc_compare_with_local.

    Attributes:
        token: User JWT bearer token.
        document_id: EDMS document UUID (to fetch the attachment).
        local_file_path: Absolute path to the uploaded local file.
        attachment_id: Specific attachment UUID; if omitted the first is used.
    """

    token: str = Field(..., description="Токен авторизации пользователя (JWT)")
    document_id: str = Field(..., description="UUID документа в СЭД")
    local_file_path: str = Field(
        ...,
        description=(
            "Абсолютный путь к локальному загруженному файлу. "
            "Берётся из <local_file_path> в system prompt."
        ),
    )
    attachment_id: Optional[str] = Field(
        None,
        description=(
            "UUID вложения из документа. Если не указан — берётся первое вложение."
        ),
    )


def _normalise(text: str) -> str:
    """Normalise whitespace for fair comparison.

    Args:
        text: Raw extracted text.

    Returns:
        Normalised string with consistent whitespace.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _diff_lines(text_a: str, text_b: str, label_a: str, label_b: str) -> List[str]:
    """Produce a unified diff between two texts.

    Args:
        text_a: First text (local file).
        text_b: Second text (attachment).
        label_a: Label for first text.
        label_b: Label for second text.

    Returns:
        List of diff lines.
    """
    lines_a = text_a.splitlines(keepends=True)
    lines_b = text_b.splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=label_a,
            tofile=label_b,
            lineterm="",
            n=2,
        )
    )
    return diff


def _find_meaningful_differences(diff_lines: List[str]) -> List[Dict[str, str]]:
    """Extract meaningful changed lines from a unified diff.

    Args:
        diff_lines: Output from _diff_lines.

    Returns:
        List of dicts with keys: type (added/removed), content.
    """
    changes: List[Dict[str, str]] = []
    for line in diff_lines:
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        stripped = line[1:].strip()
        if not stripped:
            continue
        if line.startswith("+"):
            changes.append({"type": "added_in_attachment", "content": stripped})
        elif line.startswith("-"):
            changes.append({"type": "removed_from_attachment", "content": stripped})
    return changes


@tool("doc_compare_with_local", args_schema=FileCompareInput)
async def doc_compare_with_local(
    token: str,
    document_id: str,
    local_file_path: str,
    attachment_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare a locally uploaded file with an EDMS document attachment.

    This tool performs a full text-level comparison and reports:
    - Whether files are identical or different.
    - Specific lines that differ (up to 50 differences).
    - Summary statistics (chars, lines, similarity score).

    Args:
        token: JWT bearer token.
        document_id: EDMS document UUID.
        local_file_path: Absolute path to local file.
        attachment_id: Specific attachment UUID (auto-selects first if omitted).

    Returns:
        Dict with keys:
        - status: «success» | «error»
        - are_identical: bool
        - similarity_percent: float (0–100)
        - local_stats: {chars, lines} for local file
        - attachment_stats: {chars, lines} for attachment
        - differences: list of changed lines (added/removed)
        - summary: Human-readable comparison summary
    """
    logger.info(
        "doc_compare_with_local called",
        extra={
            "document_id": document_id,
            "local_file_path": local_file_path,
            "attachment_id": attachment_id,
        },
    )

    if not os.path.exists(local_file_path):
        return {
            "status": "error",
            "message": f"Локальный файл не найден: '{local_file_path}'.",
        }

    local_text_raw: str = await FileProcessorService.extract_text_async(local_file_path)
    if not local_text_raw or local_text_raw.startswith("Ошибка:"):
        return {
            "status": "error",
            "message": f"Не удалось извлечь текст из локального файла: {local_text_raw}",
        }

    try:
        async with DocumentClient() as doc_client:
            raw_data = await doc_client.get_document_metadata(token, document_id)
            doc = DocumentDto.model_validate(raw_data)
            attachments = doc.attachmentDocument or []
    except Exception as exc:
        return {"status": "error", "message": f"Ошибка получения документа: {exc}"}

    if not attachments:
        return {"status": "error", "message": "В документе нет вложений для сравнения."}

    if attachment_id:
        target = next(
            (a for a in attachments if str(getattr(a, "id", "")) == attachment_id),
            None,
        )
        if not target:
            att_list = [
                f"{getattr(a, 'name', None) or getattr(a, 'fileName', '')} (ID: {getattr(a, 'id', '')})"
                for a in attachments
            ]
            return {
                "status": "error",
                "message": (
                    f"Вложение '{attachment_id}' не найдено. "
                    f"Доступные вложения: {', '.join(att_list)}"
                ),
            }
    else:
        local_name = os.path.basename(local_file_path).lower()
        target = None
        for att in attachments:
            att_name = (
                getattr(att, "name", "") or getattr(att, "fileName", "")
            ).lower()
            if att_name == local_name or os.path.splitext(att_name)[0] in local_name:
                target = att
                break
        if target is None:
            target = attachments[0]
            logger.info("No name match — comparing with first attachment")

    att_id = str(getattr(target, "id", ""))
    att_name = getattr(target, "name", None) or getattr(
        target, "fileName", "attachment"
    )
    att_suffix = os.path.splitext(att_name)[1].lower() or ".tmp"

    try:
        async with EdmsAttachmentClient() as att_client:
            att_bytes = await att_client.get_attachment_content(
                token, document_id, att_id
            )
    except Exception as exc:
        return {"status": "error", "message": f"Ошибка скачивания вложения: {exc}"}

    if not att_bytes:
        return {
            "status": "error",
            "message": f"Вложение '{att_name}' пустое или недоступно.",
        }

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=att_suffix) as tmp:
            tmp.write(att_bytes)
            tmp_path = tmp.name

        try:
            att_text_raw: str = await FileProcessorService.extract_text_async(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Ошибка извлечения текста вложения: {exc}",
        }

    if not att_text_raw or att_text_raw.startswith("Ошибка:"):
        return {
            "status": "error",
            "message": f"Не удалось извлечь текст из вложения '{att_name}': {att_text_raw}",
        }

    local_text = _normalise(local_text_raw[:_MAX_TEXT_CHARS])
    att_text = _normalise(att_text_raw[:_MAX_TEXT_CHARS])

    are_identical = local_text == att_text

    matcher = difflib.SequenceMatcher(None, local_text, att_text, autojunk=False)
    similarity = round(matcher.ratio() * 100, 1)

    diff_result: List[Dict[str, str]] = []
    if not are_identical:
        diff_lines = _diff_lines(
            local_text,
            att_text,
            label_a=f"Загруженный файл: {os.path.basename(local_file_path)}",
            label_b=f"Вложение СЭД: {att_name}",
        )
        diff_result = _find_meaningful_differences(diff_lines)[:60]

    local_stats = {"chars": len(local_text), "lines": local_text.count("\n") + 1}
    att_stats = {"chars": len(att_text), "lines": att_text.count("\n") + 1}

    if are_identical:
        summary = (
            f"Файлы идентичны по содержимому (схожесть: {similarity}%). "
            f"Локальный файл: {local_stats['chars']} символов, "
            f"вложение '{att_name}': {att_stats['chars']} символов."
        )
    else:
        added = sum(1 for d in diff_result if d["type"] == "added_in_attachment")
        removed = sum(1 for d in diff_result if d["type"] == "removed_from_attachment")
        summary = (
            f"Файлы различаются (схожесть: {similarity}%). "
            f"Строк только в вложении СЭД: {added}. "
            f"Строк только в локальном файле: {removed}. "
            f"Локальный файл: {local_stats['chars']} символов, "
            f"вложение '{att_name}': {att_stats['chars']} символов."
        )

    logger.info(
        "doc_compare_with_local completed",
        extra={
            "are_identical": are_identical,
            "similarity": similarity,
            "differences_found": len(diff_result),
        },
    )

    return {
        "status": "success",
        "are_identical": are_identical,
        "similarity_percent": similarity,
        "local_file": os.path.basename(local_file_path),
        "attachment_name": att_name,
        "local_stats": local_stats,
        "attachment_stats": att_stats,
        "differences": diff_result,
        "summary": summary,
    }

# edms_ai_assistant/tools/file_compare_tool.py
"""
EDMS AI Assistant — File Comparison Tool.

Сравнивает локально загруженный файл с вложением документа EDMS.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

from langchain_core.tools import InjectedToolArg, StructuredTool
from langgraph.errors import GraphInterrupt
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.agent.hitl_primitives import ToolAborted, ask_human
from edms_ai_assistant.agent.interrupt_contract import (
    CardSelectInterrupt,
    CardSelectResume,
    InterruptCard,
)
from edms_ai_assistant.agent.runnable_utils import (
    get_document_id_from_config,
    get_token_from_config,
)
from edms_ai_assistant.services.file_processor import FileProcessorService
from edms_ai_assistant.utils.regex_utils import UUID_RE
from langchain_core.runnables import RunnableConfig
if TYPE_CHECKING:
    from edms_ai_assistant.clients.attachment_client import AttachmentClient
    from edms_ai_assistant.clients.document_client import DocumentClient

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
    """Validated input for the doc_compare_attachment_with_local tool."""

    local_file_path: str = Field(
        ...,
        description=(
            "Абсолютный путь к загруженному файлу. "
            "Инжектируется автоматически из контекста агента."
        ),
    )
    attachment_id: str | None = Field(
        None,
        description=(
            "UUID вложения или его имя для сравнения. "
            "Если не указан — ищем по оригинальному имени локального файла. "
            "Если совпадение не найдено — возвращается список вложений для выбора."
        ),
    )
    original_filename: str | None = Field(
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
    # Поддержка как dict, так и Pydantic объектов с extra='allow'
    if isinstance(attachment, dict):
        return attachment.get("name") or attachment.get("fileName") or ""
    return (
        getattr(attachment, "name", None) or getattr(attachment, "fileName", None) or ""
    )


def _att_id(attachment: Any) -> str:
    if isinstance(attachment, dict):
        return str(attachment.get("id", ""))
    return str(getattr(attachment, "id", "") or "")


def _resolve_attachment(attachments: list[Any], hint: str) -> Any | None:
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

    logger.warning(
        "Attachment resolution failed: hint='%s', available=%s",
        hint,
        [_att_name(a) for a in attachments],
    )

    return None


def _normalise(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _compute_diff(
    local_text: str,
    att_text: str,
    local_label: str,
    att_label: str,
) -> list[dict[str, str]]:
    raw_diff = difflib.unified_diff(
        local_text.splitlines(keepends=True),
        att_text.splitlines(keepends=True),
        fromfile=local_label,
        tofile=att_label,
        lineterm="",
        n=2,
    )
    changes: list[dict[str, str]] = []
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
    local_stats: dict[str, int],
    att_stats: dict[str, int],
    diff_result: list[dict[str, str]],
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


# ─── Tool Factory ─────────────────────────────────────────────────────────────


def create_file_compare_tool(
    document_client: DocumentClient,
    attachment_client: AttachmentClient,
) -> StructuredTool:
    """Фабрика для создания инструмента сравнения файлов.

    Args:
        document_client: Клиент для работы с документами EDMS.
        attachment_client: Клиент для скачивания вложений EDMS.

    Returns:
        Настроенный StructuredTool.
    """

    async def doc_compare_attachment_with_local(
        local_file_path: str,
        attachment_id: str | None = None,
        original_filename: str | None = None,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """СРАВНИТЬ локальный файл с вложением документа в СЭД.

        ИСПОЛЬЗУЙ ЭТОТ ИНСТРУМЕНТ, КОГДА:
        • Пользователь загрузил файл и просит сравнить его с вложением
        • Нужно найти различия между файлом на компьютере и файлом в СЭД

        НЕ ИСПОЛЬЗУЙ этот инструмент для:
        • Сравнения двух документов СЭД -> используй `doc_compare`
        • Сравнения версий документа -> используй `doc_get_versions`

        ПАРАМЕТРЫ:
        • attachment_id: UUID вложения ИЛИ имя файла. Если не указан — ищем по имени загруженного файла.
        • local_file_path: БЕРЁТСЯ АВТОМАТИЧЕСКИ из контекста — НЕ указывай вручную!
        • document_id и token: БЕРУТСЯ АВТОМАТИЧЕСКИ из контекста — НЕ указывай вручную!

        ПОСЛЕ DISAMBIGUATION:
        Если пользователь выбрал UUID из списка вложений — передай этот UUID в `attachment_id`
        и вызови этот инструмент снова. НЕ вызывай `doc_compare`!

        Args:
            local_file_path: Путь к локальному файлу.
            attachment_id: UUID или имя вложения.
            original_filename: Оригинальное имя файла.
            config: LangGraph RunnableConfig (инжектируется автоматически).
        """
        try:
            token = get_token_from_config(config)
            document_id = get_document_id_from_config(config)
        except RuntimeError as exc:
            logger.error("Missing context in tool call: %s", exc)
            return {"status": "error", "message": str(exc)}

        local_path = Path(local_file_path)
        display_name: str = (
            original_filename.strip()
            if original_filename and original_filename.strip()
            else local_path.name
        )

        logger.info(
            "doc_compare_attachment_with_local called",
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
            doc_raw = await document_client.get_document_metadata(token, document_id)
            if not doc_raw:
                return {"status": "error", "message": "Документ не найден."}
            from edms_ai_assistant.domain.document import DocumentDto

            doc = DocumentDto.model_validate(doc_raw)
            attachments: list[Any] = doc.attachment_document or []
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

        # ── 3. Разрешение целевого вложения (с HITL Disambiguation) ──────────────
        target = None
        if attachment_id:
            target = _resolve_attachment(attachments, attachment_id)
        else:
            target = _resolve_attachment(attachments, display_name)
            if target is None and display_name != local_path.name:
                target = _resolve_attachment(attachments, local_path.name)

        if target is None:
            logger.info("Attachment resolution failed -> HITL disambiguation")

            cards = [
                InterruptCard(
                    id=_att_id(a),
                    label=_att_name(a) or "без имени",
                    description="Вложение документа",
                    badges=["Документ"],
                    metadata={"id": _att_id(a), "name": _att_name(a)},
                )
                for a in attachments
                if _att_id(a)
            ]

            hint = attachment_id or display_name
            prompt_msg = (
                f"Вложение «{hint}» не найдено в документе. "
                f"Уточните, какое вложение сравнить с «{display_name}»:"
                if attachment_id
                else f"Не удалось автоматически определить вложение для сравнения "
                f"с «{display_name}». Выберите нужное:"
            )

            try:
                resume = ask_human(
                    CardSelectInterrupt(
                        prompt=prompt_msg,
                        cards=cards,
                        multiple=False,
                    )
                )
                if not isinstance(resume, CardSelectResume):
                    raise ToolAborted("Expected CardSelectResume")

                selected_id = resume.selected_ids[0]
                target = next(
                    (a for a in attachments if _att_id(a) == selected_id), None
                )

            except ToolAborted:
                return {"status": "cancelled", "message": "Выбор вложения отменён."}

            except GraphInterrupt:
                raise

            except Exception as exc:
                logger.error("HITL disambiguation failed: %s", exc, exc_info=True)
                return {"status": "error", "message": f"Ошибка выбора вложения: {exc}"}

        resolved_id = str(target.id) if target.id else ""
        resolved_name = target.name or "attachment"
        resolved_suffix = Path(resolved_name).suffix.lower() or ".tmp"

        att_doc_id: str = str(target.document_id or document_id)
        if att_doc_id and att_doc_id != document_id:
            logger.debug(
                "Attachment belongs to version document_id=%s…, context=%s…",
                att_doc_id[:8],
                document_id[:8],
            )

        logger.info("Attachment resolved: '%s' (%s…)", resolved_name, resolved_id[:8])

        # ── 4. Скачивание вложения ────────────────────────────────────────────────
        try:
            att_bytes: bytes = await attachment_client.get_attachment_content(
                token, UUID(att_doc_id), UUID(resolved_id)
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
        local_text_raw: str = await FileProcessorService.extract_text_async(
            str(local_path)
        )
        if not local_text_raw or local_text_raw.startswith(("Ошибка:", "Формат файла")):
            return {
                "status": "error",
                "message": f"Не удалось извлечь текст из «{display_name}»: {local_text_raw}",
            }

        # ── 6. Извлечение текста вложения через temp-файл ─────────────────────────
        att_text_raw: str = ""

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=resolved_suffix) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(att_bytes)
                tmp_file.flush()

            try:
                att_text_raw = await FileProcessorService.extract_text_async(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except PermissionError:
                        logger.warning(
                            "Could not delete temp file %s, it will be cleaned up later",
                            tmp_path,
                        )

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

        # ── 7. Нормализация -> сравнение -> diff ───────────────────────────────────
        local_text = _normalise(local_text_raw[:_MAX_TEXT_CHARS])
        att_text = _normalise(att_text_raw[:_MAX_TEXT_CHARS])

        are_identical = local_text == att_text
        similarity = round(
            difflib.SequenceMatcher(None, local_text, att_text, autojunk=False).ratio()
            * 100,
            1,
        )

        diff_result: list[dict[str, str]] = []
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
            "doc_compare_attachment_with_local completed",
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

    return StructuredTool.from_function(
        coroutine=doc_compare_attachment_with_local,
        name="doc_compare_attachment_with_local",
        description="СРАВНИТЬ локальный файл с вложением документа в СЭД.",
        args_schema=FileCompareInput,
    )

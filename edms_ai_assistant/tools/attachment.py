# edms_ai_assistant/tools/attachment.py
"""
EDMS AI Assistant — Attachment Fetch Tool (DI Factory).

Извлекает содержимое вложений из документов EDMS.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.agent.hitl_primitives import ask_human
from edms_ai_assistant.agent.interrupt_contract import (
    CardSelectInterrupt,
    CardSelectResume,
    InterruptCard,
)
from edms_ai_assistant.agent.runnable_utils import (
    get_document_id_from_config,
    get_token_from_config,
)
from edms_ai_assistant.clients.attachment_client import AttachmentClient
from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.core.deps import AppDeps
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.services.file_processor import FileProcessorService
from edms_ai_assistant.utils.regex_utils import UUID_RE

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS: int = 15_000
_SUPPORTED_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".doc",
        ".txt",
        ".rtf",
        ".odt",
        ".md",
    }
)
_SUPPORTED_TABLE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".xlsx",
        ".xls",
        ".csv",
    }
)
_ALL_SUPPORTED: frozenset[str] = (
    _SUPPORTED_TEXT_EXTENSIONS | _SUPPORTED_TABLE_EXTENSIONS
)


# ─── Input schema ─────────────────────────────────────────────────────────────


class AttachmentFetchInput(BaseModel):
    """Validated input for the doc_get_file_content tool."""

    attachment_id: str | None = Field(
        None,
        description=(
            "UUID вложения или имя файла (например: «Шаблон обложки.docx»). "
            "Если не указан — берётся первое вложение документа."
        ),
    )
    analysis_mode: str = Field(
        "text",
        description=(
            "Режим анализа: "
            "'text' — текст (PDF/DOCX/TXT/RTF), "
            "'tables' — таблицы (Excel/CSV), "
            "'metadata' — только метаданные, "
            "'full' — текст + таблицы + метаданные."
        ),
    )
    summary_type: str | None = Field(
        None,
        description=(
            "Тип суммаризации для передачи следующему инструменту: "
            "extractive | abstractive | thesis."
        ),
    )

    @field_validator("analysis_mode")
    @classmethod
    def validate_analysis_mode(cls, v: str) -> str:
        allowed = {"text", "tables", "metadata", "full"}
        normalised = v.strip().lower()
        if normalised not in allowed:
            raise ValueError(
                f"Неизвестный режим анализа '{v}'. "
                f"Допустимые значения: {', '.join(sorted(allowed))}."
            )
        return normalised


# ─── Private helpers ──────────────────────────────────────────────────────────


def _get_attachment_name(attachment: Any) -> str:
    return (
        getattr(attachment, "name", None) or getattr(attachment, "fileName", None) or ""
    )


def _get_attachment_id(attachment: Any) -> str:
    return str(getattr(attachment, "id", "") or "")


def _resolve_attachment(attachments: list[Any], hint: str) -> Any | None:
    hint_stripped = hint.strip()

    if UUID_RE.match(hint_stripped):
        found = next(
            (a for a in attachments if _get_attachment_id(a) == hint_stripped),
            None,
        )
        if found is not None:
            logger.debug("Resolved attachment by UUID: %s...", hint_stripped[:8])
            return found

    hint_lower = hint_stripped.lower()
    hint_stem = Path(hint_lower).stem

    for att in attachments:
        if _get_attachment_name(att).lower() == hint_lower:
            logger.info("Resolved attachment by exact name: '%s'", hint_stripped)
            return att

    for att in attachments:
        att_stem = Path(_get_attachment_name(att).lower()).stem
        if att_stem == hint_stem and hint_stem:
            logger.info("Resolved attachment by stem: '%s'", hint_stripped)
            return att

    for att in attachments:
        att_stem = Path(_get_attachment_name(att).lower()).stem
        if hint_stem and att_stem and (hint_stem in att_stem or att_stem in hint_stem):
            logger.info(
                "Resolved attachment by partial stem: '%s' ~ '%s'",
                hint_stem,
                att_stem,
            )
            return att

    return None


def _build_attachment_meta(attachment: Any) -> dict[str, Any]:
    name = _get_attachment_name(attachment) or "unknown"
    size_bytes: int = getattr(attachment, "size", None) or 0
    upload_date = getattr(attachment, "uploadDate", None)
    signs = getattr(attachment, "signs", None) or []

    att_type_obj = getattr(attachment, "attachmentDocumentType", None)
    att_type: str | None = None
    if att_type_obj:
        att_type = getattr(att_type_obj, "name", None) or str(att_type_obj)

    formatted_date: str | None = None
    if upload_date:
        try:
            formatted_date = (
                upload_date.strftime("%d.%m.%Y %H:%M")
                if hasattr(upload_date, "strftime")
                else str(upload_date)[:16]
            )
        except Exception:
            formatted_date = str(upload_date)

    return {
        "название": name,
        "тип_вложения": att_type,
        "размер_кб": round(size_bytes / 1024, 2) if size_bytes else 0,
        "дата_загрузки": formatted_date,
        "есть_эцп": bool(signs),
        "id": _get_attachment_id(attachment) or None,
    }


# ─── Tool Factory ─────────────────────────────────────────────────────────────


def create_attachment_fetch_tool(deps: AppDeps) -> StructuredTool:
    """Фабрика инструмента извлечения содержимого вложений с DI."""

    doc_client: DocumentClient = deps.document_client
    attach_client: AttachmentClient = deps.attachment_client
    file_processor: FileProcessorService = deps.file_processor_service

    async def doc_get_file_content(
        attachment_id: str | None = None,
        analysis_mode: str = "text",
        summary_type: str | None = None,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Extract and analyse the content of a document attachment from EDMS.

        Accepts ``attachment_id`` as either a UUID or a filename string.
        Performs multi-strategy name resolution before falling back to
        the first attachment with an info-level log entry.

        ВАЖНО: Токен авторизации и ID документа передаются системой АВТОМАТИЧЕСКИ.
        Тебе НЕ НУЖНО запрашивать их у пользователя или передавать в аргументах.

        Analysis modes:
        - ``text``     : Plain text extraction (PDF, DOCX, TXT, RTF, ODT, MD).
        - ``tables``   : Structured table extraction (XLSX, XLS, CSV).
        - ``metadata`` : Attachment metadata only — no file download.
        - ``full``     : Text + tables + metadata in a single response.
        """
        try:
            token = get_token_from_config(config)
            document_id = get_document_id_from_config(config)
        except Exception as e:
            logger.error("Failed to get token/document_id from config: %s", e)
            return {"status": "error", "message": f"Ошибка авторизации или контекста документа: {e}"}

        logger.info(
            "doc_get_file_content called",
            extra={
                "document_id": document_id[:8] + "...",
                "attachment_id": attachment_id,
                "analysis_mode": analysis_mode,
                "summary_type": summary_type,
            },
        )

        # ── Получение метаданных документа ───────────────────────────────────────
        try:
            raw_data = await doc_client.get_document_metadata(token, document_id)
            doc = DocumentDto.model_validate(raw_data)
            attachments: list[Any] = doc.attachmentDocument or []
        except Exception as exc:
            logger.error("Failed to fetch document metadata: %s", exc, exc_info=True)
            return {
                "status": "error",
                "message": f"Ошибка получения метаданных документа: {exc}",
                "summary_type": summary_type,
            }

        if not attachments:
            return {
                "status": "info",
                "message": "В документе нет вложений.",
                "summary_type": summary_type,
            }

        # ── Разрешение целевого вложения ──────────────────────────────────────────
        target = None
        if attachment_id:
            target = _resolve_attachment(attachments, attachment_id)
            if target is None:
                cards = []
                for a in attachments:
                    meta = _build_attachment_meta(a)
                    cards.append(InterruptCard(
                        id=meta["id"],
                        label=meta["название"],
                        description=f"{meta['тип_вложения'] or 'Файл'} • {meta['размер_кб']} КБ",
                        primary_attrs={
                            "Дата": meta["дата_загрузки"] or "—",
                            "ЭЦП": "Да" if meta["есть_эцп"] else "Нет"
                        },
                        metadata={"url": f"/attachment/{meta['название']}"}
                    ))

                resume = ask_human(CardSelectInterrupt(
                    prompt=f"Вложение «{attachment_id}» не найдено. Выберите вложение для анализа:",
                    cards=cards
                ))
                if isinstance(resume, CardSelectResume):
                    resolved_id = resume.selected_ids[0]
                    target = next((a for a in attachments if _get_attachment_id(a) == resolved_id), None)
                else:
                    return {"status": "error", "message": "Выбор вложения отменён."}

            if target:
                resolved_id = _get_attachment_id(target)
                logger.info("Attachment resolved: '%s' → %s...", attachment_id, resolved_id[:8])
        else:
            if len(attachments) == 1:
                target = attachments[0]
                resolved_id = _get_attachment_id(target)
                logger.info("Single attachment auto-selected: %s...", resolved_id[:8])
            else:
                cards = []
                for a in attachments:
                    meta = _build_attachment_meta(a)
                    cards.append(InterruptCard(
                        id=meta["id"],
                        label=meta["название"],
                        description=f"{meta['тип_вложения'] or 'Файл'} • {meta['размер_кб']} КБ",
                        primary_attrs={
                            "Дата": meta["дата_загрузки"] or "—",
                            "ЭЦП": "Да" if meta["есть_эцп"] else "Нет"
                        },
                        metadata={"url": f"/attachment/{meta['название']}"}
                    ))

                resume = ask_human(CardSelectInterrupt(
                    prompt="В документе несколько вложений. Какое из них проанализировать?",
                    cards=cards
                ))
                if isinstance(resume, CardSelectResume):
                    resolved_id = resume.selected_ids[0]
                    target = next((a for a in attachments if _get_attachment_id(a) == resolved_id), None)
                else:
                    return {"status": "error", "message": "Выбор вложения отменён."}

        if not target:
             return {"status": "error", "message": "Не удалось определить вложение."}

        file_info = _build_attachment_meta(target)
        file_name: str = file_info["название"]
        suffix: str = Path(file_name).suffix.lower() if file_name else ".tmp"

        # ── Режим metadata: без скачивания ────────────────────────────────────────
        if analysis_mode == "metadata":
            return {
                "status": "success",
                "mode": "metadata",
                "file_info": file_info,
                "summary_type": summary_type,
            }

        # ── Скачивание вложения ───────────────────────────────────────────────────
        att_doc_id: str = str(getattr(target, "documentId", None) or document_id)
        try:
            content_bytes = await attach_client.get_attachment_content(
                token, att_doc_id, resolved_id
            )
        except Exception as exc:
            logger.error(
                "Failed to download attachment '%s': %s", file_name, exc, exc_info=True
            )
            return {
                "status": "error",
                "message": f"Ошибка скачивания «{file_name}»: {exc}",
                "file_info": file_info,
                "summary_type": summary_type,
            }

        if not content_bytes:
            return {
                "status": "error",
                "message": f"Файл «{file_name}» пустой или недоступен для скачивания.",
                "file_info": file_info,
                "summary_type": summary_type,
            }

        # ── Обработка через FileProcessorService ─────────────────────────────────
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content_bytes)
                tmp_path = tmp.name

            # ── full: текст + таблицы + метаданные ───────────────────────────────
            if analysis_mode == "full":
                structured = await file_processor.extract_structured_data(tmp_path)
                text_content: str = structured.get("text", "")
                return {
                    "status": "success",
                    "mode": "full",
                    "file_info": file_info,
                    "content": text_content[:_MAX_TEXT_CHARS],
                    "is_truncated": len(text_content) > _MAX_TEXT_CHARS,
                    "total_chars": len(text_content),
                    "metadata": structured.get("metadata"),
                    "stats": structured.get("stats"),
                    "tables": structured.get("tables"),
                    "summary_type": summary_type,
                }

            # ── tables: только таблицы (Excel/CSV) ───────────────────────────────
            if analysis_mode == "tables":
                if suffix not in _SUPPORTED_TABLE_EXTENSIONS:
                    return {
                        "status": "error",
                        "message": (
                            f"Режим 'tables' поддерживается только для Excel/CSV. "
                            f"Текущий формат: '{suffix}'. "
                            "Используй режим 'text' для текстовых документов."
                        ),
                        "file_info": file_info,
                        "summary_type": summary_type,
                    }
                structured = await file_processor.extract_structured_data(tmp_path)
                tables = structured.get("tables", [])
                return {
                    "status": "success",
                    "mode": "tables",
                    "file_info": file_info,
                    "tables": tables,
                    "tables_count": len(tables) if tables else 0,
                    "summary_type": summary_type,
                }

            # ── text: извлечение текста (default) ────────────────────────────────
            if suffix not in _ALL_SUPPORTED:
                return {
                    "status": "warning",
                    "message": (
                        f"Формат '{suffix}' не поддерживается для извлечения текста. "
                        f"Поддерживаемые: {', '.join(sorted(_ALL_SUPPORTED))}. "
                        "Метаданные файла возвращены."
                    ),
                    "file_info": file_info,
                    "summary_type": summary_type,
                }

            text_content = await file_processor.extract_text_async(tmp_path)

            if not text_content or text_content.startswith(("Ошибка:", "Формат файла")):
                return {
                    "status": "error",
                    "message": (
                        f"Не удалось извлечь текст из «{file_name}». "
                        "Возможно, файл является сканом или защищён паролем. "
                        f"Подробности: {text_content}"
                    ),
                    "file_info": file_info,
                    "summary_type": summary_type,
                }

            return {
                "status": "success",
                "mode": "text",
                "file_info": file_info,
                "content": text_content[:_MAX_TEXT_CHARS],
                "is_truncated": len(text_content) > _MAX_TEXT_CHARS,
                "total_chars": len(text_content),
                "summary_type": summary_type,
            }

        except Exception as exc:
            logger.error(
                "File processing error for '%s': %s", file_name, exc, exc_info=True
            )
            return {
                "status": "error",
                "message": (
                    f"Не удалось обработать «{file_name}»: {exc}. "
                    "Возможно, файл повреждён или защищён паролем."
                ),
                "file_info": file_info,
                "summary_type": summary_type,
            }
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError as cleanup_err:
                    logger.warning(
                        "Failed to remove temp file '%s': %s", tmp_path, cleanup_err
                    )

    return StructuredTool.from_function(
        func=doc_get_file_content,
        name="doc_get_file_content",
        description=(
            "Извлекает и анализирует содержимое вложения документа из EDMS.\n"
            "Принимает attachment_id как UUID или имя файла. "
            "Выполняет разрешение имени перед тем, как выбрать первое вложение.\n\n"
            "Режимы анализа:\n"
            "- text: Извлечение простого текста (PDF, DOCX, TXT, RTF, ODT, MD).\n"
            "- tables: Извлечение структурированных таблиц (XLSX, XLS, CSV).\n"
            "- metadata: Только метаданные вложения (без скачивания файла).\n"
            "- full: Текст + таблицы + метаданные в одном ответе.\n\n"
            "ВАЖНО: Токен авторизации и ID документа передаются системой АВТОМАТИЧЕСКИ. "
            "НЕ запрашивай их у пользователя."
        ),
        args_schema=AttachmentFetchInput,
    )
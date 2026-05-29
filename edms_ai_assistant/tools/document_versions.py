# edms_ai_assistant/tools/doc_versions_tool.py
"""
EDMS AI Assistant — Document Versions Tool.

Получает все версии документа и АВТОМАТИЧЕСКИ сравнивает каждую пару
соседних версий (v1↔v2, v2↔v3, ...) без вопросов к пользователю.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.tools import InjectedToolArg, StructuredTool
from pydantic import BaseModel

from edms_ai_assistant.agent.runnable_utils import (
    get_document_id_from_config,
    get_token_from_config,
)
from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from edms_ai_assistant.clients.document_client import DocumentClient

logger = logging.getLogger(__name__)

# ── Поля метаданных для сравнения версий ─────────────────────────────────────
_METADATA_FIELDS: tuple[tuple[str, str], ...] = (
    ("regNumber", "Рег. номер"),
    ("regDate", "Дата регистрации"),
    ("status", "Статус"),
    ("shortSummary", "Краткое содержание"),
    ("correspondentName", "Корреспондент"),
    ("outRegNumber", "Исходящий номер"),
    ("outRegDate", "Исходящая дата"),
    ("author", "Автор"),
)


def _att_name(attachment: Any) -> str:
    """Извлекает имя вложения с fallback на originalName."""
    if isinstance(attachment, dict):
        return (
            attachment.get("name")
            or attachment.get("originalName")
            or attachment.get("fileName")
            or ""
        )
    return (
        getattr(attachment, "name", None)
        or getattr(attachment, "original_name", None)  # Поддержка snake_case из DTO
        or getattr(attachment, "originalName", None)
        or ""
    )


def _compare_metadata(doc1: dict[str, Any], doc2: dict[str, Any]) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    for field_key, field_label in _METADATA_FIELDS:
        v1 = doc1.get(field_key)
        v2 = doc2.get(field_key)
        if v1 != v2:
            changes[field_label] = {
                "было": str(v1) if v1 is not None else "—",
                "стало": str(v2) if v2 is not None else "—",
            }
    return changes


def _compare_attachments(doc1: dict[str, Any], doc2: dict[str, Any]) -> dict[str, Any]:
    att1: list[Any] = doc1.get("attachmentDocument") or []
    att2: list[Any] = doc2.get("attachmentDocument") or []

    names1: set[str] = {_att_name(a) for a in att1 if _att_name(a)}
    names2: set[str] = {_att_name(a) for a in att2 if _att_name(a)}

    return {
        "добавлены_в_новой": sorted(names2 - names1) or [],
        "удалены_из_старой": sorted(names1 - names2) or [],
        "присутствуют_в_обеих": sorted(names1 & names2) or [],
        "кол-во_вложений_старая": len(att1),
        "кол-во_вложений_новая": len(att2),
    }


class DocumentVersionsInput(BaseModel):
    """Схема входных данных для получения и сравнения всех версий документа."""

    pass


# ─── Tool Factory ─────────────────────────────────────────────────────────────


def create_doc_get_versions_tool(document_client: DocumentClient) -> StructuredTool:
    """Фабрика для создания инструмента сравнения версий документа.

    Args:
        document_client: Клиент для работы с документами EDMS.

    Returns:
        Настроенный StructuredTool, готовый к регистрации в агенте.
    """

    async def doc_get_versions(
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Retrieve all document versions and compare each consecutive pair automatically.

        BEHAVIOUR:
        - Fetches ALL versions of the document (N versions).
        - Compares every consecutive pair: v1↔v2, v2↔v3, ..., v(N-1)↔vN.
        - Returns full aggregated comparison results for ALL pairs.
        - Agent MUST NOT ask the user which versions to compare — all pairs are compared.
        - If only 1 version exists — informs user, no comparison possible.
        - If exactly 2 versions — compares them (1 pair).
        - If N > 2 versions — compares all N-1 consecutive pairs and presents full history.

        ВАЖНО: Токен авторизации и ID документа передаются системой АВТОМАТИЧЕСКИ.
        Тебе НЕ НУЖНО запрашивать их у пользователя или передавать в аргументах.

        Args:
            config: LangGraph RunnableConfig (инжектируется автоматически).

        Returns:
            Dict with all versions metadata and full pair-wise comparison results.
        """
        try:
            token = get_token_from_config(config)
            document_id = get_document_id_from_config(config)
        except RuntimeError as exc:
            logger.error("Missing context in tool call: %s", exc)
            return {"status": "error", "message": str(exc)}

        try:
            versions = await document_client.get_document_versions(token, document_id)

            if not versions:
                return {
                    "status": "success",
                    "total_versions": 0,
                    "message": "У документа только одна версия — сравнивать не с чем.",
                }

            # Сортируем по номеру версии (атрибут DocumentVersionDto)
            sorted_versions = sorted(versions, key=lambda v: v.version or 0)
            total = len(sorted_versions)

            # ── Сбор метаданных всех версий ─────────────────────────────────
            versions_info: list[dict[str, Any]] = []
            version_ids: dict[str, str] = {}  # "1" -> doc_uuid

            for v in sorted_versions:
                vnum = v.version or (len(versions_info) + 1)
                doc_uuid = v.document_id
                created_dt = (
                    getattr(v.document, "create_date", None) if v.document else None
                )
                if doc_uuid:
                    version_ids[str(vnum)] = str(doc_uuid)
                versions_info.append(
                    {
                        "version_number": vnum,
                        "created_date": str(created_dt or ""),
                    }
                )

            if total == 1:
                return {
                    "status": "success",
                    "total_versions": 1,
                    "versions": versions_info,
                    "message": "Документ имеет только одну версию — история изменений недоступна.",
                }

            # ── Последовательное сравнение всех соседних пар ────────────────
            comparisons: list[dict[str, Any]] = []
            errors: list[str] = []

            version_nums = sorted(version_ids.keys(), key=lambda x: int(x))

            for i in range(len(version_nums) - 1):
                from_vnum = version_nums[i]
                to_vnum = version_nums[i + 1]
                from_id = version_ids[from_vnum]
                to_id = version_ids[to_vnum]

                try:
                    doc_from_dto = await document_client.get_document_metadata(
                        token, from_id
                    )
                    doc_to_dto = await document_client.get_document_metadata(
                        token, to_id
                    )

                    if not doc_from_dto or not doc_to_dto:
                        errors.append(
                            f"Версия {from_vnum} или {to_vnum}: метаданные недоступны"
                        )
                        continue

                    # Конвертируем DTO в dict с camelCase ключами для переиспользования
                    # логики сравнения, которая опирается на ключи API СЭД
                    dict_from = doc_from_dto.model_dump(
                        by_alias=True, exclude_none=True
                    )
                    dict_to = doc_to_dto.model_dump(by_alias=True, exclude_none=True)

                    meta_diff = _compare_metadata(dict_from, dict_to)
                    att_diff = _compare_attachments(dict_from, dict_to)

                    comparisons.append(
                        {
                            "pair": f"Версия {from_vnum} -> Версия {to_vnum}",
                            "from_version": int(from_vnum),
                            "to_version": int(to_vnum),
                            "metadata_changes": meta_diff,
                            "metadata_changed": bool(meta_diff),
                            "attachment_changes": att_diff,
                            "attachments_changed": (
                                bool(att_diff.get("добавлены_в_новой"))
                                or bool(att_diff.get("удалены_из_старой"))
                            ),
                        }
                    )

                except Exception as pair_exc:
                    err = f"Ошибка сравнения версий {from_vnum}↔{to_vnum}: {pair_exc}"
                    logger.error(err)
                    errors.append(err)

            # ── Агрегированный вывод ─────────────────────────────────────────
            has_any_changes = any(
                c.get("metadata_changed") or c.get("attachments_changed")
                for c in comparisons
            )

            logger.info(
                "doc_get_versions completed: total=%d pairs=%d", total, len(comparisons)
            )

            return {
                "status": "success",
                "total_versions": total,
                "versions": versions_info,
                "version_ids": version_ids,
                "comparisons": comparisons,
                "has_any_changes": has_any_changes,
                "errors": errors if errors else None,
                "message": (
                    f"Документ имеет {total} версии. "
                    f"Выполнено {len(comparisons)} сравнений: "
                    + ", ".join(c["pair"] for c in comparisons)
                    + (
                        ". Изменения обнаружены."
                        if has_any_changes
                        else ". Версии идентичны."
                    )
                ),
                "comparison_complete": True,
                "instruction": (
                    "Все сравнения уже выполнены. Сформулируй полный ответ пользователю "
                    "на основе поля 'comparisons'. НЕ вызывай doc_compare — данные уже есть."
                ),
            }

        except Exception as e:
            logger.error("[DOC-VERSIONS-TOOL] Error: %s", e, exc_info=True)
            return {"status": "error", "message": f"Ошибка получения версий: {e!s}"}

    return StructuredTool.from_function(
        coroutine=doc_get_versions,
        name="doc_get_versions",
        description="Retrieve all document versions and compare each consecutive pair automatically.",
        args_schema=DocumentVersionsInput,
    )

# edms_ai_assistant/tools/doc_search.py
"""
EDMS AI Assistant — Document Search Tool.

Слой: Infrastructure / Tool.
Поиск документов в EDMS по текстовому запросу, номеру,
дате регистрации и категории документа.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.clients.document_client import DocumentClient

logger = logging.getLogger(__name__)

# Максимальное количество документов в одном ответе агенту
_MAX_RESULTS = 10


class DocSearchInput(BaseModel):
    """Validated input schema for document search."""

    token: str = Field(..., description="JWT токен авторизации пользователя")
    search: Optional[str] = Field(
        None,
        max_length=500,
        description=(
            "Текстовый поиск по краткому содержанию, регистрационному "
            "номеру или автору документа"
        ),
    )
    reg_number: Optional[str] = Field(
        None,
        max_length=100,
        description="Точный или частичный регистрационный номер документа",
    )
    doc_category: Optional[str] = Field(
        None,
        description=(
            "Категория документа: INTERN, INCOMING, OUTGOING, "
            "APPEAL, CONTRACT, MEETING"
        ),
    )
    date_from: Optional[str] = Field(
        None,
        description="Начало диапазона дат регистрации (YYYY-MM-DD)",
        pattern=r"^\d{4}-\d{2}-\d{2}$|^$",
    )
    date_to: Optional[str] = Field(
        None,
        description="Конец диапазона дат регистрации (YYYY-MM-DD)",
        pattern=r"^\d{4}-\d{2}-\d{2}$|^$",
    )

    @field_validator("search", "reg_number")
    @classmethod
    def strip_strings(cls, v: Optional[str]) -> Optional[str]:
        """Strips surrounding whitespace from string fields."""
        return v.strip() if v else None


@tool("doc_search_tool", args_schema=DocSearchInput)
async def doc_search_tool(
    token: str,
    search: Optional[str] = None,
    reg_number: Optional[str] = None,
    doc_category: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ищет документы в EDMS по текстовому запросу, номеру, дате или категории.

    Используй когда пользователь просит:
    - «Найди документы про договор с ООО Альфа»
    - «Покажи входящие документы за март»
    - «Есть ли документ с номером ВХ-2026-001?»
    - «Найди все обращения за последний месяц»

    Возвращает список документов с id, регистрационным номером, датой,
    кратким содержанием и категорией (до 10 документов).
    """
    if not any([search, reg_number, doc_category, date_from, date_to]):
        return {
            "status": "error",
            "message": (
                "Укажите хотя бы один параметр поиска: текст, "
                "номер документа, категорию или диапазон дат."
            ),
        }

    filters: Dict[str, Any] = {"size": _MAX_RESULTS, "page": 0}

    if search:
        filters["search"] = search
    if reg_number:
        filters["regNumber"] = reg_number
    if doc_category:
        filters["docCategoryConstant"] = doc_category.upper()
    if date_from:
        filters["regDateFrom"] = f"{date_from}T00:00:00Z"
    if date_to:
        filters["regDateTo"] = f"{date_to}T23:59:59Z"

    logger.info(
        "Document search requested",
        extra={"filters": {k: v for k, v in filters.items() if k != "token"}},
    )

    try:
        async with DocumentClient() as client:
            raw_docs = await client.search_documents(token, filters)

        if not raw_docs:
            return {
                "status": "success",
                "message": "По вашему запросу документы не найдены.",
                "documents": [],
                "total": 0,
            }

        documents: List[Dict[str, Any]] = []
        for d in raw_docs[:_MAX_RESULTS]:
            documents.append(
                {
                    "id": str(d.get("id", "")),
                    "reg_number": d.get("regNumber") or d.get("reservedRegNumber", "—"),
                    "reg_date": str(d.get("regDate", ""))[:10] or "—",
                    "category": str(d.get("docCategoryConstant", "—")),
                    "short_summary": (d.get("shortSummary") or "")[:200],
                    "author": _format_author(d.get("author")),
                    "status": str(d.get("status", "—")),
                }
            )

        logger.info(
            "Document search completed",
            extra={"found": len(documents), "query": search or reg_number},
        )

        return {
            "status": "success",
            "total": len(documents),
            "documents": documents,
            "message": f"Найдено {len(documents)} документ(ов)",
        }

    except Exception as exc:
        logger.error("Document search failed", exc_info=True)
        return {
            "status": "error",
            "message": f"Ошибка поиска документов: {exc}",
        }


def _format_author(author: Optional[Dict[str, Any]]) -> str:
    """Formats an author dict into a human-readable name string."""
    if not author:
        return "—"
    parts = [
        author.get("lastName", ""),
        author.get("firstName", ""),
        author.get("middleName", ""),
    ]
    return " ".join(p for p in parts if p).strip() or "—"

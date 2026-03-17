# edms_ai_assistant/tools/doc_resolution.py
"""
EDMS AI Assistant — Document Resolution Tool.

Слой: Infrastructure / Tool.
Позволяет читать существующие резолюции документа и добавлять
новые резолюции через EDMS API.

Резолюция (resolution) — это официальное решение/поручение руководителя,
внесённое непосредственно в карточку документа.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.clients.base_client import EdmsHttpClient

logger = logging.getLogger(__name__)

# Максимальная длина текста резолюции
_MAX_RESOLUTION_TEXT = 2000


class _ResolutionClient(EdmsHttpClient):
    """Minimal EDMS client for resolution endpoints."""

    async def get_resolutions(
        self, token: str, document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Fetches all resolutions for a document.

        GET api/document/{document_id}/resolution
        """
        result = await self._make_request(
            "GET", f"api/document/{document_id}/resolution", token=token
        )
        return result if isinstance(result, list) else []

    async def create_resolution(
        self,
        token: str,
        document_id: str,
        text: str,
        executor_ids: List[str],
        deadline: Optional[str],
    ) -> Dict[str, Any]:
        """
        Creates a resolution on a document.

        POST api/document/{document_id}/resolution
        """
        payload: Dict[str, Any] = {
            "resolutionText": text.strip(),
            "executorIds": executor_ids,
        }
        if deadline:
            payload["planedDateEnd"] = deadline

        result = await self._make_request(
            "POST",
            f"api/document/{document_id}/resolution",
            token=token,
            json=payload,
            is_json_response=False,
        )
        return result if isinstance(result, dict) else {"raw": result}


# ──────────────────────────────────────────────────────────────────────────────
# Схемы ввода
# ──────────────────────────────────────────────────────────────────────────────


class GetResolutionsInput(BaseModel):
    """Input schema for fetching document resolutions."""

    token: str = Field(..., description="JWT токен авторизации пользователя")
    document_id: str = Field(..., description="UUID документа")


class CreateResolutionInput(BaseModel):
    """Input schema for creating a document resolution."""

    token: str = Field(..., description="JWT токен авторизации пользователя")
    document_id: str = Field(..., description="UUID документа")
    resolution_text: str = Field(
        ...,
        min_length=5,
        max_length=_MAX_RESOLUTION_TEXT,
        description="Текст резолюции (обязательно)",
    )
    executor_ids: Optional[List[str]] = Field(
        None,
        description=(
            "Список UUID сотрудников-исполнителей. Если не указан — "
            "резолюция без назначенных исполнителей."
        ),
    )
    deadline: Optional[str] = Field(
        None,
        description=(
            "Плановая дата исполнения резолюции в ISO 8601 "
            "(например: '2026-04-01T23:59:59Z')"
        ),
    )

    @field_validator("resolution_text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        """Strips whitespace from resolution text."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Текст резолюции не может быть пустым")
        return stripped

    @field_validator("executor_ids")
    @classmethod
    def validate_executor_ids(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Filters out empty strings from executor id list."""
        if not v:
            return None
        cleaned = [uid.strip() for uid in v if uid and uid.strip()]
        return cleaned or None


# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────


@tool("doc_get_resolutions", args_schema=GetResolutionsInput)
async def doc_get_resolutions(
    token: str,
    document_id: str,
) -> Dict[str, Any]:
    """
    Возвращает список всех резолюций на документ.

    Используй когда пользователь спрашивает:
    - «Какие резолюции есть по этому документу?»
    - «Покажи резолюции руководителя»
    - «Кто написал резолюцию?»

    Возвращает список резолюций с текстом, автором, датой и исполнителями.
    """
    logger.info(
        "Fetching resolutions",
        extra={"document_id": document_id},
    )

    try:
        async with _ResolutionClient() as client:
            resolutions = await client.get_resolutions(token, document_id)

        if not resolutions:
            return {
                "status": "success",
                "message": "У документа нет резолюций.",
                "resolutions": [],
                "total": 0,
            }

        formatted: List[Dict[str, Any]] = []
        for r in resolutions:
            author = r.get("author") or {}
            author_name = " ".join(
                filter(
                    None,
                    [
                        author.get("lastName", ""),
                        author.get("firstName", ""),
                        author.get("middleName", ""),
                    ],
                )
            )

            executors = [
                " ".join(
                    filter(
                        None,
                        [
                            (e.get("executor") or e).get("lastName", ""),
                            (e.get("executor") or e).get("firstName", ""),
                        ],
                    )
                )
                for e in (r.get("executors") or [])
            ]

            formatted.append(
                {
                    "id": str(r.get("id", "")),
                    "text": r.get("resolutionText") or r.get("text", ""),
                    "author": author_name.strip() or "—",
                    "create_date": str(r.get("createDate", ""))[:10],
                    "deadline": str(r.get("planedDateEnd", ""))[:10] or None,
                    "executors": [e for e in executors if e.strip()],
                    "status": str(r.get("status") or "—"),
                }
            )

        logger.info(
            "Resolutions fetched",
            extra={"document_id": document_id, "count": len(formatted)},
        )

        return {
            "status": "success",
            "total": len(formatted),
            "resolutions": formatted,
            "message": f"Найдено {len(formatted)} резолюций",
        }

    except Exception as exc:
        logger.error("Failed to fetch resolutions", exc_info=True)
        return {
            "status": "error",
            "message": f"Ошибка получения резолюций: {exc}",
        }


@tool("doc_create_resolution", args_schema=CreateResolutionInput)
async def doc_create_resolution(
    token: str,
    document_id: str,
    resolution_text: str,
    executor_ids: Optional[List[str]] = None,
    deadline: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Создаёт новую резолюцию на документ.

    Используй когда пользователь просит:
    - «Напиши резолюцию: "Исполнить в срок до 15 апреля"»
    - «Добавь резолюцию с назначением Иванова»
    - «Зарезервируй резолюцию руководителя»

    Для назначения конкретных исполнителей используй employee_search_tool
    чтобы получить их UUID перед вызовом этого инструмента.
    """
    logger.info(
        "Creating resolution",
        extra={
            "document_id": document_id,
            "text_len": len(resolution_text),
            "has_executors": bool(executor_ids),
            "has_deadline": bool(deadline),
        },
    )

    try:
        async with _ResolutionClient() as client:
            result = await client.create_resolution(
                token=token,
                document_id=document_id,
                text=resolution_text,
                executor_ids=executor_ids or [],
                deadline=deadline,
            )

        logger.info(
            "Resolution created successfully",
            extra={"document_id": document_id},
        )

        return {
            "status": "success",
            "message": "Резолюция успешно добавлена к документу.",
            "resolution_text": resolution_text,
            "result": result,
        }

    except Exception as exc:
        logger.error("Failed to create resolution", exc_info=True)
        return {
            "status": "error",
            "message": f"Ошибка создания резолюции: {exc}",
        }

# edms_ai_assistant/services/introduction_service.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edms_ai_assistant.services.resolution_service import ResolutionService
    from edms_ai_assistant.clients.transport import IAsyncTransport

logger = logging.getLogger(__name__)


class PostIntroductionRequest(BaseModel):
    executorListIds: list[UUID] = Field(..., description="UUID сотрудников для добавления в список ознакомления")
    comment: str = Field(default="", description="Комментарий к ознакомлению")
    model_config = ConfigDict(json_encoders={UUID: str}, use_enum_values=True)


@dataclass(frozen=True)
class IntroductionResult:
    success: bool
    added_count: int = 0
    error_message: str | None = None


class IntroductionService:
    """Сервисный слой для управления списками ознакомления."""

    def __init__(self, resolution_service: ResolutionService, transport: IAsyncTransport):
        self._resolution = resolution_service
        self._transport = transport

    async def resolve_employees(
            self, token: str, last_names: list[str], department_names: list[str],
            group_names: list[str], personal_group_names: list[str] | None = None,
            include_subordinates: bool = False,
    ) -> dict:
        """Резолвит сотрудников по множественным критериям."""
        result = await self._resolution.resolve_bulk(
            token=token,
            department_names=department_names,
            group_names=group_names,
            personal_group_names=personal_group_names,
            include_subordinates=include_subordinates,
        )

        emp_ids, not_found_names, ambiguous_data = await self._resolution.resolve_employees(token, last_names)

        result.employee_ids.update(emp_ids)
        result.not_found.extend(not_found_names)

        return {
            "employee_ids": result.employee_ids,
            "not_found": result.not_found,
            "ambiguous": [am.__dict__ for am in ambiguous_data],
        }

    async def create_introduction(
            self, token: str, document_id: str, employee_ids: list[UUID], comment: str | None = None,
    ) -> IntroductionResult:
        """Создает список ознакомления через API EDMS."""
        if not employee_ids:
            return IntroductionResult(success=False, added_count=0,
                                      error_message="Не указаны сотрудники для добавления")

        normalized_comment = self._normalize_comment(comment)
        request = PostIntroductionRequest(executorListIds=employee_ids, comment=normalized_comment)

        try:
            endpoint = f"api/document/{document_id}/introduction"
            payload = request.model_dump(mode="json")

            await self._transport.request("POST", endpoint, token=token, json=payload)
            logger.info("Introduction created successfully",
                        extra={"document_id": document_id, "added_count": len(employee_ids)})
            return IntroductionResult(success=True, added_count=len(employee_ids))

        except Exception as e:
            logger.error("Failed to create introduction: %s", e, exc_info=True, extra={"document_id": document_id})
            return IntroductionResult(success=False, added_count=0, error_message=f"Ошибка API: {e!s}")

    @staticmethod
    def _normalize_comment(comment: str | None) -> str:
        if not comment:
            return ""
        comment = comment.strip()
        template_phrases = ["не указан комментарий к ознакомлению", "не указан комментарий",
                            "комментарий к ознакомлению"]
        if comment.lower() in template_phrases:
            return ""
        if len(comment) > 1:
            return comment[0].upper() + comment[1:]
        return comment.upper()

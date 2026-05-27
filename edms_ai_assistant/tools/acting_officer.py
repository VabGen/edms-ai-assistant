# edms_ai_assistant/tools/acting_officer.py
"""
EDMS AI Assistant — Acting Officer Tool.
Инструменты для управления ИО и секретарями.
"""

from __future__ import annotations

import logging
from typing import Any, Annotated, TYPE_CHECKING
from uuid import UUID

from langchain_core.tools import StructuredTool, InjectedToolArg
from pydantic import BaseModel, Field

from edms_ai_assistant.agent.runnable_utils import get_token_from_config

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from edms_ai_assistant.core.deps import AppDeps

logger = logging.getLogger(__name__)


class ActingOfficerListInput(BaseModel):
    employee_name: str | None = Field(None, description="Фамилия сотрудника, чьих ИО нужно найти (по умолчанию - текущий пользователь).")


def create_acting_officer_tools(deps: AppDeps) -> list[StructuredTool]:
    """Фабрика инструментов для работы с ИО."""

    async def acting_officer_list_tool(
        employee_name: str | None = None,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Возвращает список ИО и секретарей для сотрудника."""
        token = get_token_from_config(config)

        try:
            if employee_name:
                emp = await deps.employee_client.find_by_last_name_fts(token, employee_name)
                target_id = emp.id
            else:
                user = await deps.employee_client.get_current_user(token)
                target_id = user.id

            if not target_id:
                return {"status": "error", "message": "Сотрудник не найден."}

            ios = await deps.employee_acting_client.get_acting_for_target(token, target_id)
            return {
                "status": "success",
                "acting_officers": [io.model_dump(by_alias=True) for io in ios]
            }
        except Exception as e:
            return {"status": "error", "message": f"Ошибка при получении списка ИО: {e}"}

    return [
        StructuredTool.from_function(
            coroutine=acting_officer_list_tool,
            name="acting_officer_list",
            description="Показывает список исполняющих обязанности (ИО) и секретарей для указанного сотрудника.",
            args_schema=ActingOfficerListInput,
        ),
    ]

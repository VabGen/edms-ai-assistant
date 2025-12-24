import logging
from typing import Dict, Any, Optional, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.clients.employee_client import EmployeeClient

from edms_ai_assistant.generated.resources_openapi import (
    EmployeeFilter,
    Include1,
    EmployeeDto
)

logger = logging.getLogger(__name__)


class EmployeeSearchInput(BaseModel):
    """Схема входных данных для инструмента."""
    token: str = Field(..., description="Токен авторизации пользователя (JWT)")
    employee_id: Optional[str] = Field(
        None,
        description="UUID конкретного сотрудника. Используется для получения детальной карточки после выбора из списка."
    )
    lastName: Optional[str] = Field(None, description="Фамилия")
    firstName: Optional[str] = Field(None, description="Имя")
    middleName: Optional[str] = Field(None, description="Отчество")
    fullPostName: Optional[str] = Field(None, description="Полное название должности")


@tool("employee_search_tool", args_schema=EmployeeSearchInput)
async def employee_search_tool(
        token: str,
        employee_id: Optional[str] = None,
        **filter_params
) -> Dict[str, Any]:
    """
    Универсальный инструмент для работы с персоналом EDMS.
    Сначала ищет по критериям (фамилия и т.д.), возвращает список.
    Затем, если указан employee_id, возвращает полные данные сотрудника.
    """
    try:
        async with EmployeeClient() as client:

            if employee_id:
                raw_data = await client.get_employee(token, employee_id)
                if not raw_data:
                    return {"status": "error", "message": f"Сотрудник с ID {employee_id} не найден."}

                employee_dto = EmployeeDto.model_validate(raw_data)
                return {
                    "status": "found",
                    "data": employee_dto.model_dump(exclude_none=True, exclude_unset=True)
                }

            payload = {k: v for k, v in filter_params.items() if v}

            if "includes" not in payload:
                payload["includes"] = [Include1.POST, Include1.DEPARTMENT]

            if not any(k for k in payload if k != "includes"):
                return {"status": "error", "message": "Укажите фамилию или другие данные для поиска."}

            results = await client.search_employees(token, payload)

            if not results:
                return {"status": "not_found", "message": "Сотрудники не найдены."}

            if len(results) == 1:
                res_item = results[0]
                employee_dto = EmployeeDto.model_validate(res_item)
                return {
                    "status": "found",
                    "data": employee_dto.model_dump(exclude_none=True, exclude_unset=True)
                }

            choices = []
            for r in results:
                item = r if isinstance(r, dict) else r.model_dump()

                choices.append({
                    "id": str(item.get("id", "")),
                    "full_name": f"{item.get('lastName', '')} {item.get('firstName', '')} {item.get('middleName', '')}".strip(),
                    "post": item.get("post", {}).get("postName") if item.get("post") else "Не указана",
                    "department": item.get("department", {}).get("name") if item.get("department") else "Не указан"
                })

            return {
                "status": "requires_action",
                "action_type": "select_employee",
                "count": len(results),
                "choices": choices,
                "display_hint": "LIST_AS_CARDS"
            }

    except Exception as e:
        logger.error(f"Ошибка в employee_search_tool: {e}", exc_info=True)
        return {"status": "error", "message": f"Ошибка сервиса персонала: {str(e)}"}
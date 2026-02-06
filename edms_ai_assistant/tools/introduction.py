# edms_ai_assistant/tools/introduction.py
import logging
from typing import Dict, Any, Optional, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.services.introduction_service import (
    IntroductionService,
    IntroductionResult,
)

logger = logging.getLogger(__name__)


class IntroductionInput(BaseModel):

    token: str = Field(..., description="Токен авторизации пользователя (JWT)")
    document_id: str = Field(..., description="UUID документа для создания ознакомления")

    last_names: Optional[List[str]] = Field(
        None,
        description="Фамилии сотрудников для добавления в ознакомление (например: ['Иванов', 'Петров'])",
    )
    department_names: Optional[List[str]] = Field(
        None,
        description="Названия подразделений/отделов для массового добавления",
    )
    group_names: Optional[List[str]] = Field(
        None,
        description="Названия групп для массового добавления",
    )
    comment: Optional[str] = Field(
        None, description="Комментарий к ознакомлению (необязательно)"
    )

    selected_employee_ids: Optional[List[str]] = Field(
        None,
        description="UUID сотрудников, выбранных пользователем для разрешения неоднозначности"
    )


@tool("introduction_create_tool", args_schema=IntroductionInput)
async def introduction_create_tool(
        token: str,
        document_id: str,
        last_names: Optional[List[str]] = None,
        department_names: Optional[List[str]] = None,
        group_names: Optional[List[str]] = None,
        comment: Optional[str] = None,
        selected_employee_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Создает список ознакомления с документом.

    - Если фамилия неоднозначна (несколько Ивановых), возвращает статус "requires_disambiguation"
    - Пользователь должен выбрать конкретного сотрудника из списка
    - После выбора, инструмент вызывается повторно с selected_employee_ids

    Примеры:
    1. Простой случай: last_names=["Уникальная_Фамилия"] → автоматическое добавление
    2. Неоднозначность: last_names=["Иванов", "Петров"] → требуется уточнение
    3. После выбора: selected_employee_ids=["uuid1", "uuid2"] → создание ознакомления
    """
    logger.info(
        f"[INTRODUCTION-TOOL] Creating introduction for document {document_id}. "
        f"LastNames: {last_names}, Departments: {department_names}, Groups: {group_names}, "
        f"SelectedIDs: {selected_employee_ids}"
    )

    if last_names:
        last_names_set = set(name.lower() for name in last_names)
        if department_names:
            department_names = [
                d for d in department_names if d.lower() not in last_names_set
            ]
        if group_names:
            group_names = [g for g in group_names if g.lower() not in last_names_set]

    if not any([last_names, department_names, group_names, selected_employee_ids]):
        return {
            "status": "error",
            "message": "Необходимо указать хотя бы один параметр: фамилии, департаменты, группы или выбранных сотрудников.",
        }

    try:
        async with IntroductionService() as service:

            if selected_employee_ids:
                from uuid import UUID

                employee_uuids = [UUID(emp_id) for emp_id in selected_employee_ids]

                success = await service.create_introduction(
                    token=token,
                    document_id=document_id,
                    employee_ids=employee_uuids,
                    comment=comment,
                )

                if success:
                    return {
                        "status": "success",
                        "message": f"Успешно добавлено {len(employee_uuids)} сотрудников в список ознакомления.",
                        "added_count": len(employee_uuids),
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Не удалось создать ознакомление. Проверьте права доступа или корректность данных.",
                    }

            employee_ids, not_found, ambiguous_results = await service.collect_employees(
                token=token,
                last_names=last_names,
                department_names=department_names,
                group_names=group_names,
            )

            if ambiguous_results:
                return {
                    "status": "requires_disambiguation",
                    "message": "Найдено несколько сотрудников с указанными фамилиями. Выберите нужных из списка:",
                    "ambiguous_matches": ambiguous_results,
                    "instruction": (
                        "Пожалуйста, выберите конкретных сотрудников из списка. "
                        "Затем вызовите инструмент повторно с параметром selected_employee_ids."
                    )
                }

            if not employee_ids:
                return {
                    "status": "error",
                    "message": "Не найдено ни одного сотрудника для добавления в ознакомление.",
                    "not_found": not_found,
                }

            success = await service.create_introduction(
                token=token,
                document_id=document_id,
                employee_ids=list(employee_ids),
                comment=comment,
            )

            if success:
                result = {
                    "status": "success",
                    "message": f"Успешно добавлено {len(employee_ids)} сотрудников в список ознакомления.",
                    "added_count": len(employee_ids),
                }

                if not_found:
                    result["partial_success"] = True
                    result["not_found"] = not_found
                    result["message"] += f" Не найдено: {', '.join(not_found)}."

                return result
            else:
                return {
                    "status": "error",
                    "message": "Не удалось создать ознакомление. Проверьте права доступа или корректность данных.",
                }

    except Exception as e:
        logger.error(f"[INTRODUCTION-TOOL] Error: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Произошла ошибка при создании ознакомления: {str(e)}",
        }
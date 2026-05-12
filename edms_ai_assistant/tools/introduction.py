"""
EDMS AI Assistant - Introduction Tool.

Инструмент для создания списков ознакомления с документами.
"""

import logging
from uuid import UUID

from typing import Any, Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.services.introduction_service import IntroductionService

logger = logging.getLogger(__name__)


class IntroductionInput(BaseModel):
    """Валидированная схема входных данных для создания ознакомления."""
    last_names: list[str] | None = Field(
        None,
        description="Фамилии сотрудников для поиска (например: ['Иванов', 'Петров'])",
        max_length=50,
    )
    department_names: list[str] | None = Field(
        None,
        description="Названия подразделений для массового добавления",
        max_length=20,
    )
    group_names: list[str] | None = Field(
        None,
        description="Названия групп для массового добавления",
        max_length=20,
    )
    personal_group_names: list[str] | None = Field(
        None,
        description=(
            "Названия ЛИЧНЫХ групп пользователя. В отличие от обычных групп, "
            "личная группа принадлежит конкретному пользователю и содержит "
            "сотрудников, которых он сам добавил. "
            "Пример: ['Моя команда', 'Контактная группа']"
        ),
        max_length=20,
    )
    include_subordinates: bool | None = Field(
        None,
        description=(
            "True — включить подчинённых текущего пользователя. "
            "Подчинённые — сотрудники подразделения, которым руководит пользователь."
        ),
    )
    comment: str | None = Field(
        None,
        description="Комментарий к ознакомлению",
        max_length=500,
    )
    selected_employee_ids: list[str] | None = Field(
        None,
        description=(
            "UUID выбранных сотрудников для разрешения disambiguation. "
            "Используется после выбора пользователем из списка неоднозначных совпадений."
        ),
        max_length=100,
    )

    @field_validator(
        "last_names", "department_names", "group_names", "personal_group_names"
    )
    @classmethod
    def validate_string_lists(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        return [s.strip() for s in v if s and s.strip()]

    @field_validator("selected_employee_ids")
    @classmethod
    def validate_employee_ids(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        validated = []
        for emp_id in v:
            try:
                UUID(emp_id)
                validated.append(emp_id)
            except ValueError:
                logger.warning(f"Invalid UUID in selected_employee_ids: {emp_id}")
        return validated if validated else None


@tool("introduction_create_tool", args_schema=IntroductionInput)
async def introduction_create_tool(
        last_names: list[str] | None = None,
        department_names: list[str] | None = None,
        group_names: list[str] | None = None,
        personal_group_names: list[str] | None = None,
        include_subordinates: bool | None = None,
        comment: str | None = None,
        selected_employee_ids: list[str] | None = None,
        document_id: Annotated[str, InjectedToolArg] = "",
        token: Annotated[str, InjectedToolArg] = "",
) -> dict[str, Any]:
    """
    Создает список ознакомления с документом.

    Типы исполнителей:
    1. Индивидуальные: по фамилии/ФИО (last_names)
    2. Подразделения: все сотрудники отдела (department_names)
    3. Группы: все сотрудники группы (group_names)
    4. Личные группы: сотрудники из личной группы пользователя (personal_group_names)
    5. Подчинённые: сотрудники подчинённых подразделений (include_subordinates=True)

    Можно комбинировать:
    "Добавь ознакомление для Петрова, отдела ИТ и моей команды"

    Args:
        last_names: Фамилии сотрудников.
        department_names: Подразделения.
        group_names: Группы.
        personal_group_names: Личные группы.
        include_subordinates: Включить подчинённых.
        comment: Комментарий.
        selected_employee_ids: UUID выбранных сотрудников (для disambiguation).
        document_id: UUID документа (инжектируется автоматически).
        token: JWT токен авторизации (инжектируется автоматически).
    """
    logger.info(
        "Creating introduction",
        extra={
            "document_id": document_id,
            "last_names": last_names,
            "departments": department_names,
            "groups": group_names,
            "personal_groups": personal_group_names,
            "subordinates": include_subordinates,
            "has_selected_ids": bool(selected_employee_ids),
        },
    )

    try:
        async with IntroductionService() as service:
            if selected_employee_ids:
                return await _handle_direct_addition(
                    service=service,
                    token=token,
                    document_id=document_id,
                    employee_ids=selected_employee_ids,
                    comment=comment,
                )

            return await _handle_search_and_create(
                service=service,
                token=token,
                document_id=document_id,
                last_names=last_names,
                department_names=department_names,
                group_names=group_names,
                personal_group_names=personal_group_names,
                include_subordinates=bool(include_subordinates),
                comment=comment,
            )

    except Exception as e:
        logger.error(
            f"Introduction creation failed: {e}",
            exc_info=True,
            extra={"document_id": document_id},
        )
        return {
            "status": "error",
            "message": f"❌ Произошла ошибка при создании ознакомления: {e!s}",
        }


async def _handle_direct_addition(
    service: IntroductionService,
    token: str,
    document_id: str,
    employee_ids: list[str],
    comment: str | None,
) -> dict[str, Any]:
    """Обработка прямого добавления сотрудников по UUID."""
    logger.info(f"Direct addition of {len(employee_ids)} employees")

    if not employee_ids:
        return {
            "status": "error",
            "message": "Не указаны ID сотрудников для добавления.",
        }

    result = await service.create_introduction(
        token=token,
        document_id=document_id,
        employee_ids=[UUID(emp_id) for emp_id in employee_ids],
        comment=comment,
    )

    if result.success:
        return {
            "status": "success",
            "message": (
                f"✅ Успешно добавлено {result.added_count} сотрудников "
                f"в список ознакомления."
            ),
            "added_count": result.added_count,
        }

    return {
        "status": "error",
        "message": (
            result.error_message
            or "❌ Не удалось создать ознакомление. "
            "Проверьте права доступа или корректность данных."
        ),
    }


async def _handle_search_and_create(
    service: IntroductionService,
    token: str,
    document_id: str,
    last_names: list[str] | None,
    department_names: list[str] | None,
    group_names: list[str] | None,
    personal_group_names: list[str] | None,
    include_subordinates: bool,
    comment: str | None,
) -> dict[str, Any]:
    """Обработка поиска сотрудников с последовательным disambiguation."""
    resolution_result = await service.resolve_employees(
        token=token,
        last_names=last_names or [],
        department_names=department_names or [],
        group_names=group_names or [],
        personal_group_names=personal_group_names or [],
        include_subordinates=include_subordinates,
    )

    employee_ids = resolution_result.employee_ids
    not_found = resolution_result.not_found
    ambiguous_results = resolution_result.ambiguous

    if ambiguous_results:
        logger.info(f"Found {len(ambiguous_results)} ambiguous search terms")
        return _build_sequential_disambiguation_response(
            ambiguous_results=ambiguous_results,
            original_last_names=last_names or [],
            not_found=not_found,
        )

    if not employee_ids:
        not_found_str = (
            ", ".join(not_found) if not_found else "Критерии поиска не заданы"
        )
        return {
            "status": "error",
            "message": f"❌ Не найдено ни одного сотрудника. Не найдены: {not_found_str}",
            "not_found": not_found,
        }

    logger.info(f"Creating introduction with {len(employee_ids)} employees")

    result = await service.create_introduction(
        token=token,
        document_id=document_id,
        employee_ids=list(employee_ids),
        comment=comment,
    )

    if result.success:
        response = {
            "status": "success",
            "message": (
                f"✅ Успешно добавлено {result.added_count} сотрудников "
                f"в список ознакомления."
            ),
            "added_count": result.added_count,
        }

        if not_found:
            response["partial_success"] = True
            response["not_found"] = not_found
            response["message"] += f" ⚠️ Не найдено: {', '.join(not_found)}."

        return response

    return {
        "status": "error",
        "message": result.error_message or "❌ Не удалось создать ознакомление.",
    }


def _build_sequential_disambiguation_response(
    ambiguous_results: list[dict[str, Any]],
    original_last_names: list[str],
    not_found: list[str],
) -> dict[str, Any]:
    """Формирует ответ для последовательного disambiguation."""
    groups: list[dict[str, Any]] = []
    for amb in ambiguous_results:
        search_term = amb.get("search_query", "Неизвестно")
        matches = amb.get("matches", [])
        groups.append({"search_term": search_term, "matches": matches})

    if not groups:
        return {
            "status": "error",
            "message": "Неожиданная ошибка: нет групп для disambiguation.",
        }

    current = groups[0]
    remaining = groups[1:]

    formatted_choices = [
        {
            "id": m.get("id", ""),
            "full_name": m.get("full_name", "Не указано"),
            "post": m.get("post", "Не указана"),
            "department": m.get("department", "Не указан"),
            "search_term": current["search_term"],
        }
        for m in current["matches"]
    ]

    msg = f"Уточните сотрудника для «{current['search_term']}» ({len(current['matches'])} совпадений)."
    if remaining:
        remaining_names = [g["search_term"] for g in remaining]
        msg += f" Осталось уточнить: {', '.join(remaining_names)}."

    return {
        "status": "requires_disambiguation",
        "action_type": "select_employee",
        "message": msg,
        "ambiguous_matches": formatted_choices,
        "current_group": current["search_term"],
        "remaining_groups": [g["search_term"] for g in remaining],
        "already_selected_ids": [],
        "original_last_names": original_last_names,
        "disambiguation_groups": [
            {"search_term": g["search_term"], "count": len(g["matches"])}
            for g in groups
        ],
        "not_found": not_found,
    }

# edms_ai_assistant/tools/task.py
"""Task Creation Tool with Sequential Disambiguation."""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from edms_ai_assistant.domain.task_models import TaskType
from edms_ai_assistant.services.task_service import TaskService

logger = logging.getLogger(__name__)


class TaskCreateInput(BaseModel):
    """Схема входных данных инструмента создания поручения."""

    token: str = Field(..., description="Токен авторизации пользователя (JWT)")
    document_id: str = Field(..., description="UUID документа для создания поручения")
    task_text: str = Field(..., description="Текст поручения (обязательно)")

    # ── Исполнители: индивидуальные ────────────────────────────────────
    executor_last_names: list[str] | None = Field(
        None,
        description=(
            "Фамилии или ФИО исполнителей. "
            "Примеры: ['Иванов'], ['Петров Леонид'], ['Иванов', 'Петров']"
        ),
    )
    responsible_last_name: str | None = Field(
        None, description="Фамилия ответственного исполнителя"
    )

    # ── Исполнители: массовые ──────────────────────────────────────────
    department_names: list[str] | None = Field(
        None,
        description=(
            "Названия подразделений. Все сотрудники подразделений станут "
            "исполнителями. Пример: ['Отдел ИТ', 'Бухгалтерия']"
        ),
        max_length=20,
    )
    group_names: list[str] | None = Field(
        None,
        description=(
            "Названия групп. Все сотрудники групп станут исполнителями. "
            "Пример: ['Бухгалтеры', 'Операторы ГРС']"
        ),
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
            "True — включить подчинённых текущего пользователя как исполнителей. "
            "Подчинённые — сотрудники подразделения, которым руководит пользователь."
        ),
    )

    # ── Прочее ─────────────────────────────────────────────────────────
    planed_date_end: str | None = Field(
        None, description="Плановая дата окончания в ISO 8601"
    )
    task_type: TaskType | None = Field(
        TaskType.GENERAL, description="Тип поручения (по умолчанию: GENERAL)"
    )
    selected_employee_ids: list[str] | None = Field(
        None,
        description=(
            "UUID выбранных сотрудников из предыдущих раундов disambiguation. "
            "Заполняется автоматически системой. НЕ заполняйте вручную."
        ),
        max_length=100,
    )


@tool("task_create_tool", args_schema=TaskCreateInput)
async def task_create_tool(
    token: str,
    document_id: str,
    task_text: str,
    executor_last_names: list[str] | None = None,
    responsible_last_name: str | None = None,
    department_names: list[str] | None = None,
    group_names: list[str] | None = None,
    personal_group_names: list[str] | None = None,
    include_subordinates: bool | None = None,
    planed_date_end: str | None = None,
    task_type: TaskType | None = TaskType.GENERAL,
    selected_employee_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Создает поручение с поддержкой различных типов исполнителей.

    Типы исполнителей:
    1. Индивидуальные: по фамилии/ФИО (executor_last_names)
    2. Подразделения: все сотрудники отдела (department_names)
    3. Группы: все сотрудники группы (group_names)
    4. Личные группы: сотрудники из личной группы пользователя (personal_group_names)
    5. Подчинённые: сотрудники подчинённых подразделений (include_subordinates=True)

    Можно комбинировать:
    "Создай поручение для Петрова, отдела ИТ, группы Бухгалтеры и моей команды"
    """
    if not task_text or not task_text.strip():
        return {"status": "error", "message": "Текст поручения не может быть пустым."}

    deadline = None
    if planed_date_end:
        try:
            deadline = datetime.fromisoformat(planed_date_end.replace("Z", "+00:00"))
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=UTC)
        except ValueError as e:
            return {"status": "error", "message": f"Неверный формат даты: {e}"}

    effective_task_type = task_type or TaskType.GENERAL
    preselected_ids: list[str] = list(selected_employee_ids or [])

    try:
        async with TaskService() as service:

            # ================================================================
            # Шаг 1: Резолвинг массовых исполнителей
            # ================================================================
            bulk_ids: list[UUID] = []
            bulk_not_found: list[str] = []

            has_bulk = (
                department_names
                or group_names
                or personal_group_names
                or include_subordinates
            )
            if has_bulk:
                bulk_result = await service.resolve_bulk_executors(
                    token=token,
                    department_names=department_names,
                    group_names=group_names,
                    personal_group_names=personal_group_names,
                    include_subordinates=bool(include_subordinates),
                )
                bulk_ids = list(bulk_result.employee_ids)
                bulk_not_found = bulk_result.not_found

            # ================================================================
            # Шаг 2: Резолвинг индивидуальных исполнителей
            # ================================================================
            all_uuids: list[UUID] = [UUID(eid) for eid in preselected_ids]
            all_uuids.extend(bulk_ids)

            if executor_last_names:
                executors, not_found, ambiguous = await service.collect_executors(
                    token, executor_last_names, responsible_last_name
                )

                if ambiguous:
                    return _build_sequential_disambiguation_response(
                        ambiguous_results=ambiguous,
                        executor_last_names=executor_last_names,
                        already_selected_ids=[str(u) for u in all_uuids],
                        already_resolved_uuids=[str(u) for u in all_uuids],
                        task_text=task_text,
                        not_found=not_found + bulk_not_found,
                    )

                if executors:
                    all_uuids.extend(e.employeeId for e in executors)
                bulk_not_found.extend(not_found)

            # Убираем дубликаты
            seen: set[UUID] = set()
            unique_uuids: list[UUID] = []
            for uid in all_uuids:
                if uid not in seen:
                    seen.add(uid)
                    unique_uuids.append(uid)

            if not unique_uuids:
                return {
                    "status": "error",
                    "message": "Не найдены исполнители."
                    + (
                        f" Не найдены: {', '.join(bulk_not_found)}"
                        if bulk_not_found
                        else ""
                    ),
                }

            # ================================================================
            # Шаг 3: Создание поручения
            # ================================================================
            result = await service.create_task_by_employee_ids(
                token=token,
                document_id=document_id,
                task_text=task_text,
                employee_ids=unique_uuids,
                planed_date_end=deadline,
                task_type=effective_task_type,
            )

            if result.success:
                response = {
                    "status": "success",
                    "message": (
                        f"✅ Поручение успешно создано. "
                        f"Исполнителей: {result.created_count}"
                    ),
                    "created_count": result.created_count,
                }
                if bulk_not_found:
                    response["partial_success"] = True
                    response["not_found"] = bulk_not_found
                return response

            return {
                "status": "error",
                "message": result.error_message,
                "not_found_employees": result.not_found_employees,
            }

    except Exception as e:
        logger.error("[TASK-TOOL] Error: %s", e, exc_info=True)
        return {"status": "error", "message": f"Произошла ошибка: {e!s}"}


# ---------------------------------------------------------------------------
# Sequential Disambiguation Response Builder
# ---------------------------------------------------------------------------


def _build_sequential_disambiguation_response(
    ambiguous_results: list[dict[str, Any]],
    executor_last_names: list[str],
    already_selected_ids: list[str],
    already_resolved_uuids: list[str],
    task_text: str,
    not_found: list[str],
) -> dict[str, Any]:
    """Формирует ответ для последовательного disambiguation."""
    groups: list[dict[str, Any]] = []
    for amb in ambiguous_results:
        groups.append(
            {
                "search_term": amb.get("search_query", "Неизвестно"),
                "matches": amb.get("matches", []),
            }
        )

    if not groups:
        return {"status": "error", "message": "Нет групп для disambiguation."}

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
        msg += f" Осталось уточнить: {', '.join(g['search_term'] for g in remaining)}."
    if already_selected_ids:
        msg += f" Уже выбрано: {len(already_selected_ids)} сотрудник(ов)."

    return {
        "status": "requires_disambiguation",
        "action_type": "select_employee",
        "message": msg,
        "ambiguous_matches": formatted_choices,
        "current_group": current["search_term"],
        "remaining_groups": [g["search_term"] for g in remaining],
        "already_selected_ids": already_selected_ids,
        "already_resolved_uuids": already_resolved_uuids,
        "original_executor_last_names": executor_last_names,
        "original_task_text": task_text,
        "disambiguation_groups": [
            {"search_term": g["search_term"], "count": len(g["matches"])}
            for g in groups
        ],
        "not_found": not_found,
    }

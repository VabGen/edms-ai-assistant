# edms_ai_assistant/tools/task.py
"""Task Creation Tool with Native HITL Disambiguation (DI Factory)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

from langchain_core.tools import InjectedToolArg, StructuredTool
from langgraph.errors import GraphInterrupt
from pydantic import BaseModel, Field

from edms_ai_assistant.agent.hitl_primitives import ToolAborted, ask_human
from edms_ai_assistant.agent.interrupt_contract import (
    CardSelectInterrupt,
    CardSelectResume,
    InterruptCard,
)
from edms_ai_assistant.agent.runnable_utils import (
    get_document_id_from_config,
    get_token_from_config,
)
from edms_ai_assistant.domain.task_models import TaskType

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from edms_ai_assistant.core.deps import AppDeps

logger = logging.getLogger(__name__)


class TaskCreateInput(BaseModel):
    """Схема входных данных инструмента создания поручения."""

    task_text: str = Field(..., description="Текст поручения (обязательно)")

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
    planed_date_end: str | None = Field(
        None, description="Плановая дата окончания в ISO 8601"
    )
    task_type: TaskType | None = Field(
        None, description="Тип поручения (по умолчанию: GENERAL)"
    )
    selected_employee_ids: list[str] | None = Field(
        None,
        description=(
            "UUID выбранных сотрудников. Используйте, если UUID уже известен, "
            "иначе система сама предложит выбор через карточки."
        ),
        max_length=100,
    )


def create_task_tool(deps: AppDeps) -> StructuredTool:
    """Фабрика инструмента создания поручений с DI."""

    task_service = deps.task_service

    async def task_create_tool(
        task_text: str,
        executor_last_names: list[str] | None = None,
        responsible_last_name: str | None = None,
        department_names: list[str] | None = None,
        group_names: list[str] | None = None,
        personal_group_names: list[str] | None = None,
        include_subordinates: bool | None = None,
        planed_date_end: str | None = None,
        task_type: TaskType | None = None,
        selected_employee_ids: list[str] | None = None,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Создает поручение с поддержкой различных типов исполнителей.

        Типы исполнителей:
        1. Индивидуальные: по фамилии/ФИО (executor_last_names)
        2. Подразделения: все сотрудники отдела (department_names)
        3. Группы: все сотрудники группы (group_names)
        4. Личные группы: сотрудники из личной группы пользователя (personal_group_names)
        5. Подчинённые: сотрудники подчинённых подразделений (include_subordinates=True)

        Можно комбинировать:
        "Создай поручение для Петрова, отдела ИТ, группы Бухгалтеры и моей команды"

        ВАЖНО: Токен авторизации и ID документа передаются системой АВТОМАТИЧЕСКИ.
        Если найдено несколько сотрудников с одинаковыми фамилиями, система
        АВТОМАТИЧЕСКИ покажет карточки для выбора — не нужно спрашивать пользователя текстом.
        """
        try:
            token = get_token_from_config(config)
            document_id = get_document_id_from_config(config)
        except RuntimeError as exc:
            logger.error("Missing context in tool call: %s", exc)
            return {"status": "error", "message": str(exc)}

        if not task_text or not task_text.strip():
            return {
                "status": "error",
                "message": "Текст поручения не может быть пустым.",
            }

        deadline = None
        if planed_date_end:
            try:
                deadline = datetime.fromisoformat(
                    planed_date_end.replace("Z", "+00:00")
                )
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=UTC)
            except ValueError as e:
                return {"status": "error", "message": f"Неверный формат даты: {e}"}

        effective_task_type = task_type if task_type is not None else TaskType.GENERAL
        preselected_ids: list[str] = list(selected_employee_ids or [])

        try:
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
                bulk_result = await task_service.resolve_bulk_executors(
                    token=token,
                    department_names=department_names,
                    group_names=group_names,
                    personal_group_names=personal_group_names,
                    include_subordinates=bool(include_subordinates),
                )
                bulk_ids = list(bulk_result["employee_ids"])
                bulk_not_found = bulk_result["not_found"]

            # ================================================================
            # Шаг 2: Резолвинг индивидуальных исполнителей
            # ================================================================
            all_uuids: list[UUID] = [UUID(eid) for eid in preselected_ids]
            all_uuids.extend(bulk_ids)

            if executor_last_names:
                executors, not_found, ambiguous = await task_service.collect_executors(
                    token, executor_last_names, responsible_last_name
                )

                if executors:
                    all_uuids.extend(e.employee_id for e in executors)
                bulk_not_found.extend(not_found)

                if ambiguous:
                    for amb in ambiguous:
                        search_term = amb.get("search_query", "Неизвестно")
                        matches = amb.get("matches", [])
                        if not matches:
                            continue

                        cards = [
                            InterruptCard(
                                id=m.get("id", ""),
                                label=m.get("full_name", "Не указано"),
                                description=m.get("post", "") or "Сотрудник",
                                badges=["Сотрудник"],
                                primary_attrs={
                                    "Подразделение": m.get("department", "") or "—",
                                },
                            )
                            for m in matches
                        ]

                        prompt_msg = f"Уточните исполнителя для «{search_term}» ({len(matches)} совпадений)."

                        try:
                            resume = ask_human(
                                CardSelectInterrupt(
                                    prompt=prompt_msg,
                                    cards=cards,
                                    multiple=False,
                                )
                            )
                            if not isinstance(resume, CardSelectResume):
                                raise ToolAborted("Expected CardSelectResume")
                            selected_id = resume.selected_ids[0]
                            all_uuids.append(UUID(selected_id))
                        except ToolAborted:
                            return {
                                "status": "cancelled",
                                "message": "Выбор исполнителя отменён.",
                            }
                        except GraphInterrupt:
                            raise
                        except Exception as exc:
                            logger.error(
                                "HITL disambiguation failed: %s", exc, exc_info=True
                            )
                            return {
                                "status": "error",
                                "message": f"Ошибка выбора исполнителя: {exc}",
                            }

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
            result = await task_service.create_task_by_employee_ids(
                token=token,
                document_id=document_id,
                task_text=task_text,
                employee_ids=unique_uuids,
                planed_date_end=deadline,
                task_type=effective_task_type,
            )

            if result.success:
                response: dict[str, Any] = {
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

        except GraphInterrupt:
            raise
        except Exception as e:
            logger.error("[TASK-TOOL] Error: %s", e, exc_info=True)
            return {"status": "error", "message": f"Произошла ошибка: {e!s}"}

    return StructuredTool.from_function(
        coroutine=task_create_tool,
        name="task_create_tool",
        description=(
            "Создает поручение с поддержкой различных типов исполнителей.\n"
            "Типы исполнителей:\n"
            "1. Индивидуальные: по фамилии/ФИО (executor_last_names)\n"
            "2. Подразделения: все сотрудники отдела (department_names)\n"
            "3. Группы: все сотрудники группы (group_names)\n"
            "4. Личные группы: сотрудники из личной группы пользователя (personal_group_names)\n"
            "5. Подчинённые: сотрудники подчинённых подразделений (include_subordinates=True)\n\n"
            "ВАЖНО: Токен авторизации и ID документа передаются системой АВТОМАТИЧЕСКИ."
        ),
        args_schema=TaskCreateInput,
    )

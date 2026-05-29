# edms_ai_assistant/tools/introduction.py
"""Introduction Creation Tool with Native HITL Disambiguation (DI Factory)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

from langchain_core.tools import InjectedToolArg, StructuredTool
from langgraph.errors import GraphInterrupt
from pydantic import BaseModel, Field, field_validator

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
from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from edms_ai_assistant.core.deps import AppDeps
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
            "UUID выбранных сотрудников. Используйте, если UUID уже известен, "
            "иначе система сама предложит выбор через карточки."
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
                logger.warning("Invalid UUID in selected_employee_ids: %s", emp_id)
        return validated if validated else None


# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════


async def _handle_direct_addition(
    service: IntroductionService,
    token: str,
    document_id: str,
    employee_ids: list[str],
    comment: str | None,
) -> dict[str, Any]:
    """Обработка прямого добавления сотрудников по UUID."""
    logger.info("Direct addition of %d employees", len(employee_ids))

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
    """Обработка поиска сотрудников с native HITL disambiguation."""
    resolution_result = await service.resolve_employees(
        token=token,
        last_names=last_names or [],
        department_names=department_names or [],
        group_names=group_names or [],
        personal_group_names=personal_group_names or [],
        include_subordinates=include_subordinates,
    )

    employee_ids = list(resolution_result.employee_ids)
    not_found = resolution_result.not_found
    ambiguous_results = resolution_result.ambiguous

    if ambiguous_results:
        logger.info("Found %d ambiguous search terms", len(ambiguous_results))

        for amb in ambiguous_results:
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

            prompt_msg = (
                f"Уточните сотрудника для «{search_term}» ({len(matches)} совпадений)."
            )

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
                employee_ids.append(UUID(selected_id))
            except ToolAborted:
                return {"status": "cancelled", "message": "Выбор сотрудника отменён."}
            except GraphInterrupt:
                raise
            except Exception as exc:
                logger.error("HITL disambiguation failed: %s", exc, exc_info=True)
                return {
                    "status": "error",
                    "message": f"Ошибка выбора сотрудника: {exc}",
                }

    if not employee_ids:
        not_found_str = (
            ", ".join(not_found) if not_found else "Критерии поиска не заданы"
        )
        return {
            "status": "error",
            "message": f"❌ Не найдено ни одного сотрудника. Не найдены: {not_found_str}",
            "not_found": not_found,
        }

    logger.info("Creating introduction with %d employees", len(employee_ids))

    result = await service.create_introduction(
        token=token,
        document_id=document_id,
        employee_ids=employee_ids,
        comment=comment,
    )

    if result.success:
        response: dict[str, Any] = {
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


# ══════════════════════════════════════════════════════════════════════════════
# Tool Factory
# ══════════════════════════════════════════════════════════════════════════════


def create_introduction_tool(deps: AppDeps) -> StructuredTool:
    """Фабрика инструмента ознакомления с DI."""

    introduction_service = deps.introduction_service

    async def introduction_create_tool(
        last_names: list[str] | None = None,
        department_names: list[str] | None = None,
        group_names: list[str] | None = None,
        personal_group_names: list[str] | None = None,
        include_subordinates: bool | None = None,
        comment: str | None = None,
        selected_employee_ids: list[str] | None = None,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Создает список ознакомления с документом.

        Типы исполнителей:
        1. Индивидуальные: по фамилии/ФИО (last_names)
        2. Подразделения: все сотрудники отдела (department_names)
        3. Группы: все сотрудники группы (group_names)
        4. Личные группы: сотрудники из личной группы пользователя (personal_group_names)
        5. Подчинённые: сотрудники подчинённых подразделений (include_subordinates=True)

        Можно комбинировать:
        "Добавь ознакомление для Петрова, отдела ИТ и моей команды"

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
            try:
                if selected_employee_ids:
                    return await _handle_direct_addition(
                        service=introduction_service,
                        token=token,
                        document_id=document_id,
                        employee_ids=selected_employee_ids,
                        comment=comment,
                    )

                return await _handle_search_and_create(
                    service=introduction_service,
                    token=token,
                    document_id=document_id,
                    last_names=last_names,
                    department_names=department_names,
                    group_names=group_names,
                    personal_group_names=personal_group_names,
                    include_subordinates=bool(include_subordinates),
                    comment=comment,
                )

            except GraphInterrupt:
                raise
            except Exception as e:
                logger.error(
                    "Introduction creation failed: %s",
                    e,
                    exc_info=True,
                    extra={"document_id": document_id},
                )
                return {
                    "status": "error",
                    "message": f"❌ Произошла ошибка при создании ознакомления: {e!s}",
                }
        except GraphInterrupt:
            raise
        except Exception as e:
            logger.critical("Introduction tool fatal error: %s", e, exc_info=True)
            return {
                "status": "error",
                "message": f"❌ Критическая ошибка в инструменте ознакомления: {e!s}",
            }

    return StructuredTool.from_function(
        coroutine=introduction_create_tool,
        name="introduction_create_tool",
        description=(
            "Создает список ознакомления с документом.\n"
            "Типы исполнителей:\n"
            "1. Индивидуальные: по фамилии/ФИО (last_names)\n"
            "2. Подразделения: все сотрудники отдела (department_names)\n"
            "3. Группы: все сотрудники группы (group_names)\n"
            "4. Личные группы: сотрудники из личной группы пользователя (personal_group_names)\n"
            "5. Подчинённые: сотрудники подчинённых подразделений (include_subordinates=True)\n\n"
            "ВАЖНО: Токен авторизации и ID документа передаются системой АВТОМАТИЧЕСКИ."
        ),
        args_schema=IntroductionInput,
    )

# edms_ai_assistant/tools/access_grief.py
"""
EDMS AI Assistant — Access Grief Search Tool (DI Factory).

Поиск грифов доступа и сотрудников с конкретными грифами.

Сценарии:
  - «Покажи сотрудников с грифом Секретно»  -> grief_name='Секретно'
  - «У кого гриф ДСП?»                      -> grief_name='ДСП'
  - «Какие грифы у сотрудника {uuid}»       -> employee_id=uuid
  - «Список всех грифов»                     -> (без параметров, list_all=True)

Маппинг:
  grief_name    -> GET /api/access-grief?name=...  -> UUID -> GET /api/access-grief/{id}/employees
  grief_id      -> GET /api/access-grief/{id}/employees
  employee_id   -> GET /api/employee/{id}/griefs
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, TYPE_CHECKING

import httpx
from langchain_core.tools import InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field, model_validator

from edms_ai_assistant.agent.runnable_utils import get_token_from_config

if TYPE_CHECKING:
    from edms_ai_assistant.clients.access_grief_client import AccessGriefClient
    from edms_ai_assistant.clients.employee_client import EmployeeClient
    from edms_ai_assistant.domain.employee import AccessGriefDto, EmployeeAccessGriefDto
    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

_MAX_EMPLOYEES: int = 50


# ══════════════════════════════════════════════════════════════════════════════
# Input Schema
# ══════════════════════════════════════════════════════════════════════════════


class AccessGriefSearchInput(BaseModel):
    """Схема ввода для поиска по грифам доступа."""
    grief_name: str | None = Field(
        None,
        max_length=300,
        description=(
            "Название грифа доступа. Примеры: 'Секретно', 'ДСП', "
            "'Совершенно секретно', 'Для служебного пользования'. "
            "UUID грифа резолвится автоматически через справочник. "
            "НЕ передавайте UUID в это поле."
        ),
    )

    grief_id: str | None = Field(
        None,
        description=(
            "UUID грифа доступа. Используйте ТОЛЬКО когда UUID уже "
            "получен из предыдущего API-ответа. "
            "Если известно только название — используйте grief_name."
        ),
    )

    employee_id: str | None = Field(
        None,
        description=(
            "UUID сотрудника — для получения списка грифов доступа "
            "конкретного сотрудника. "
            "Пример: «Какие грифы у сотрудника Иванов?» "
            "-> сначала найдите Иванова через employee_search_tool, "
            "затем вызовите этот инструмент с employee_id."
        ),
    )

    list_all: bool | None = Field(
        None,
        description=(
            "True — вернуть список всех грифов доступа в системе. "
            "Используйте когда пользователь спрашивает «какие грифы есть»."
        ),
    )

    @model_validator(mode="after")
    def at_least_one_param(self) -> AccessGriefSearchInput:
        fields = [self.grief_name, self.grief_id, self.employee_id, self.list_all]
        if not any(v is not None for v in fields):
            raise ValueError(
                "Укажите хотя бы один параметр: grief_name, grief_id, "
                "employee_id или list_all=True."
            )
        return self


# ══════════════════════════════════════════════════════════════════════════════
# Formatters
# ══════════════════════════════════════════════════════════════════════════════


def _format_grief_info(grief: AccessGriefDto, grief_id: str) -> dict[str, Any]:
    return {
        "id": str(grief.id or grief_id),
        "name": grief.name or "—",
    }


def _format_grief_brief(grief: AccessGriefDto) -> dict[str, Any]:
    return {
        "id": str(grief.id or ""),
        "name": grief.name or "—",
    }


def _format_employee_grief(grief: EmployeeAccessGriefDto) -> dict[str, Any]:
    """Форматирует EmployeeAccessGriefDto."""
    name = "—"
    if grief.access_grief:
        name = grief.access_grief.name or "—"
    return {
        "id": str(grief.id or ""),
        "name": name,
    }


def _format_grief_employee(raw: EmployeeAccessGriefDto) -> dict[str, Any]:
    """Форматирует EmployeeAccessGriefDto для списка сотрудников с грифом."""
    # Note: EmployeeAccessGriefDto might need to be checked if it actually contains an employee field.
    # Looking at domain/employee.py, EmployeeAccessGriefDto(AccessGriefDto) doesn't have employee.
    # However, the backend might be sending it. Let's assume it behaves as DTO if possible.
    # If the backend sends something else, we'd need to adjust the DTO.
    # Based on _format_grief_employee logic, it seems it expects an object with 'employee' attr.
    
    emp = getattr(raw, "employee", None)
    if not emp:
        return {"id": "—", "full_name": "—"}

    post = getattr(emp, "post", None)
    department = getattr(emp, "department", None)

    parts = [
        getattr(emp, "last_name", "") or "",
        getattr(emp, "first_name", "") or "",
        getattr(emp, "middle_name", "") or "",
    ]
    full_name = " ".join(p for p in parts if p).strip() or "—"

    return {
        "id": str(getattr(emp, "id", "")),
        "full_name": full_name,
        "post": getattr(post, "post_name", None) or "—",
        "department": getattr(department, "name", None) or "—",
        "active": getattr(emp, "active", None),
        "fired": getattr(emp, "fired", None),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════════════


async def _resolve_grief_name(
        grief_client: AccessGriefClient, token: str, name: str
) -> str | None:
    """Ищет UUID грифа по названию через GET /api/access-grief?name=..."""
    try:
        griefs = await grief_client.search_griefs(token, name=name.strip())

        if griefs:
            name_lower = name.strip().lower()
            for g in griefs:
                g_name = (g.name or "").lower()
                if g_name == name_lower:
                    return str(g.id)
            return str(griefs[0].id)

        return None
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 403:
            logger.warning("User lacks ReadAccessGrief permission")
            raise PermissionError(
                "У вас нет прав для просмотра грифов доступа (403 Forbidden)."
            ) from exc
        logger.error("Failed to resolve grief name '%s'", name, exc_info=True)
        return None
    except Exception:
        logger.error("Failed to resolve grief name '%s'", name, exc_info=True)
        return None


async def _get_grief_employees(
        grief_client: AccessGriefClient,
        token: str,
        grief_id: str,
        grief_name: str | None = None,
) -> dict[str, Any]:
    """Получает сотрудников с указанным грифом."""
    try:
        pageable = {
            "page": 0,
            "size": _MAX_EMPLOYEES,
            "sort": "employee.lastName,ASC",
        }

        grief_info = await grief_client.get_grief(token, grief_id)
        raw_employees_page = await grief_client.get_grief_employees(
            token, grief_id, pageable=pageable
        )

        employees = [_format_grief_employee(e) for e in raw_employees_page.content]
        grief_label = grief_name or (
            grief_info.name if grief_info else grief_id[:8]
        )

        if not employees:
            return {
                "status": "not_found",
                "message": f"Нет сотрудников с грифом «{grief_label}».",
                "grief": (
                    _format_grief_info(grief_info, grief_id) if grief_info else None
                ),
                "employees": [],
                "total": 0,
            }

        return {
            "status": "found",
            "grief": (
                _format_grief_info(grief_info, grief_id)
                if grief_info
                else {"id": grief_id}
            ),
            "total": len(employees),
            "employees": employees,
        }
    except PermissionError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        logger.error("Failed to get grief employees", exc_info=True)
        return {
            "status": "error",
            "message": f"Ошибка получения сотрудников с грифом: {exc}",
        }


async def _get_employee_griefs(
        employee_client: EmployeeClient, token: str, employee_id: str
) -> dict[str, Any]:
    """Получает грифы доступа конкретного сотрудника."""
    try:
        griefs_raw = await employee_client.get_employee_griefs(token, employee_id)

        if not griefs_raw:
            return {
                "status": "not_found",
                "message": f"У сотрудника {employee_id[:8]}... нет грифов доступа.",
                "access_griefs": [],
            }

        griefs = [_format_employee_grief(g) for g in griefs_raw]
        return {
            "status": "found",
            "employee_id": employee_id,
            "access_griefs": griefs,
        }
    except Exception as exc:
        logger.error(
            "Failed to get employee griefs",
            exc_info=True,
            extra={"employee_id": employee_id},
        )
        return {
            "status": "error",
            "message": f"Ошибка получения грифов сотрудника: {exc}",
        }


async def _list_all_griefs(
        grief_client: AccessGriefClient, token: str
) -> dict[str, Any]:
    """Возвращает список всех грифов доступа в системе."""
    try:
        griefs = await grief_client.search_griefs(
            token, pageable={"page": 0, "size": 100, "sort": "name,ASC"}
        )

        if not griefs:
            return {
                "status": "not_found",
                "message": "Грифы доступа не найдены.",
                "griefs": [],
            }

        return {
            "status": "found",
            "total": len(griefs),
            "griefs": [_format_grief_brief(g) for g in griefs],
        }
    except Exception as exc:
        logger.error("Failed to list griefs", exc_info=True)
        return {
            "status": "error",
            "message": f"Ошибка получения списка грифов: {exc}",
        }


# ══════════════════════════════════════════════════════════════════════════════
# Factory — DI via closure
# ══════════════════════════════════════════════════════════════════════════════


def create_access_grief_tool(
        grief_client: AccessGriefClient,
        employee_client: EmployeeClient,
) -> StructuredTool:
    """Фабрика инструмента поиска грифов с внедрением зависимостей.

    Args:
        grief_client: Клиент для работы с API грифов доступа.
        employee_client: Клиент для работы с API сотрудников.

    Returns:
        StructuredTool, готовый к регистрации в агенте.
    """

    async def access_grief_tool(
            grief_name: str | None = None,
            grief_id: str | None = None,
            employee_id: str | None = None,
            list_all: bool | None = None,
            config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Searches for access griefs and employees with specific griefs.

        Use when the user asks:
        - «Покажи сотрудников с грифом Секретно»   -> grief_name='Секретно'
        - «У кого гриф ДСП?»                       -> grief_name='ДСП'
        - «Какие грифы у сотрудника {uuid}»        -> employee_id=uuid
        - «Список всех грифов доступа»             -> list_all=True
        - «Сотрудники с грифом {uuid}»             -> grief_id=uuid

        ВАЖНО: Токен авторизации передается системой АВТОМАТИЧЕСКИ.
        Тебе НЕ НУЖНО запрашивать его у пользователя.
        """
        try:
            token = get_token_from_config(config)
        except Exception as e:
            logger.error("Failed to get token from config: %s", e)
            return {
                "status": "error",
                "message": f"Ошибка авторизации: токен не найден. {e}",
            }

        # ── Грифы конкретного сотрудника ──────────────────────────────────
        if employee_id:
            return await _get_employee_griefs(employee_client, token, employee_id)

        # ── Список всех грифов ────────────────────────────────────────────
        if list_all is True:
            return await _list_all_griefs(grief_client, token)

        # ── Сотрудники с конкретным грифом ────────────────────────────────
        resolved_grief_id = grief_id

        if grief_name and not grief_id:
            try:
                resolved_grief_id = await _resolve_grief_name(
                    grief_client, token, grief_name
                )
            except PermissionError as exc:
                return {"status": "error", "message": str(exc)}

            if not resolved_grief_id:
                return {
                    "status": "not_found",
                    "message": (
                        f"Гриф доступа «{grief_name}» не найден. "
                        "Проверьте название. Доступные грифы можно посмотреть "
                        "через list_all=True."
                    ),
                }

        if resolved_grief_id:
            return await _get_grief_employees(
                grief_client, token, resolved_grief_id, grief_name
            )

        return {
            "status": "error",
            "message": "Не указан параметр для поиска.",
        }

    return StructuredTool.from_function(
        func=access_grief_tool,
        name="access_grief_tool",
        description=(
            "Searches for access griefs and employees with specific griefs.\n"
            "Use when the user asks:\n"
            "- «Покажи сотрудников с грифом Секретно»   -> grief_name='Секретно'\n"
            "- «У кого гриф ДСП?»                       -> grief_name='ДСП'\n"
            "- «Какие грифы у сотрудника {uuid}»        -> employee_id=uuid\n"
            "- «Список всех грифов доступа»             -> list_all=True\n"
            "- «Сотрудники с грифом {uuid}»             -> grief_id=uuid\n\n"
            "ВАЖНО: Токен авторизации передается системой АВТОМАТИЧЕСКИ. "
            "Тебе НЕ НУЖНО запрашивать его у пользователя."
        ),
        args_schema=AccessGriefSearchInput,
    )

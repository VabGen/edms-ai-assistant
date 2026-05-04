"""
EDMS AI Assistant — Access Grief Search Tool.

Поиск грифов доступа и сотрудников с конкретными грифами.

Сценарии:
  - «Покажи сотрудников с грифом Секретно»  → grief_name='Секретно'
  - «У кого гриф ДСП?»                      → grief_name='ДСП'
  - «Какие грифы у сотрудника {uuid}»       → employee_id=uuid
  - «Список всех грифов»                     → (без параметров, list_all=True)

Маппинг:
  grief_name    → GET /api/access-grief?name=...  → UUID → GET /api/access-grief/{id}/employees
  grief_id      → GET /api/access-grief/{id}/employees
  employee_id   → GET /api/employee/{id}/griefs
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

from edms_ai_assistant.clients.access_grief_client import AccessGriefClient
from edms_ai_assistant.clients.employee_client import EmployeeClient

logger = logging.getLogger(__name__)

_MAX_EMPLOYEES: int = 50


# ══════════════════════════════════════════════════════════════════════════════
# Input Schema
# ══════════════════════════════════════════════════════════════════════════════


class AccessGriefSearchInput(BaseModel):
    """Схема ввода для поиска по грифам доступа."""

    token: str = Field(..., description="JWT токен авторизации пользователя")

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
            "→ сначала найдите Иванова через employee_search_tool, "
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
# Tool
# ══════════════════════════════════════════════════════════════════════════════


@tool("access_grief_tool", args_schema=AccessGriefSearchInput)
async def access_grief_tool(
    token: str,
    grief_name: str | None = None,
    grief_id: str | None = None,
    employee_id: str | None = None,
    list_all: bool | None = None,
) -> dict[str, Any]:
    """Searches for access griefs and employees with specific griefs.

    Use when the user asks:
    - «Покажи сотрудников с грифом Секретно»   → grief_name='Секретно'
    - «У кого гриф ДСП?»                       → grief_name='ДСП'
    - «Какие грифы у сотрудника {uuid}»        → employee_id=uuid
    - «Список всех грифов доступа»             → list_all=True
    - «Сотрудники с грифом {uuid}»             → grief_id=uuid

    Returns:
    - status='found' with employees[] — сотрудники с указанным грифом
    - status='found' with access_griefs[] — грифы сотрудника
    - status='found' with griefs[] — список всех грифов
    - status='not_found' — ничего не найдено
    """
    # ── Грифы конкретного сотрудника ──────────────────────────────────────
    if employee_id:
        return await _get_employee_griefs(token, employee_id)

    # ── Список всех грифов ────────────────────────────────────────────────
    if list_all is True:
        return await _list_all_griefs(token)

    # ── Сотрудники с конкретным грифом ────────────────────────────────────
    resolved_grief_id = grief_id

    if grief_name and not grief_id:
        resolved_grief_id = await _resolve_grief_name(token, grief_name)
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
        return await _get_grief_employees(token, resolved_grief_id, grief_name)

    return {
        "status": "error",
        "message": "Не указан параметр для поиска.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════════════


async def _resolve_grief_name(token: str, name: str) -> str | None:
    """Ищет UUID грифа по названию через GET /api/access-grief?name=..."""
    try:
        async with AccessGriefClient() as client:
            griefs = await client.search_griefs(token, name=name.strip())

        if griefs:
            # Точное совпадение приоритетнее
            name_lower = name.strip().lower()
            for g in griefs:
                g_name = (g.get("name") or "").lower()
                if g_name == name_lower:
                    return str(g["id"])
            return str(griefs[0]["id"])

        return None
    except Exception:
        logger.error("Failed to resolve grief name '%s'", name, exc_info=True)
        return None


async def _get_grief_employees(
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

        async with AccessGriefClient() as client:
            # Получаем информацию о грифе
            grief_info = await client.get_grief(token, grief_id)
            # Получаем сотрудников
            raw_employees = await client.get_grief_employees(
                token, grief_id, pageable=pageable
            )

        employees = [_format_grief_employee(e) for e in raw_employees]

        grief_label = grief_name or (
            grief_info.get("name") if grief_info else grief_id[:8]
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
    except Exception as exc:
        logger.error("Failed to get grief employees", exc_info=True)
        return {
            "status": "error",
            "message": f"Ошибка получения сотрудников с грифом: {exc}",
        }


async def _get_employee_griefs(token: str, employee_id: str) -> dict[str, Any]:
    """Получает грифы доступа конкретного сотрудника."""
    try:
        async with EmployeeClient() as client:
            griefs_raw = await client.get_employee_griefs(token, employee_id)

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


async def _list_all_griefs(token: str) -> dict[str, Any]:
    """Возвращает список всех грифов доступа в системе."""
    try:
        async with AccessGriefClient() as client:
            griefs = await client.search_griefs(
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


# ── Форматирование ────────────────────────────────────────────────────────


def _format_grief_info(grief: dict[str, Any], grief_id: str) -> dict[str, Any]:
    return {
        "id": str(grief.get("id", grief_id)),
        "name": grief.get("name") or "—",
    }


def _format_grief_brief(grief: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(grief.get("id", "")),
        "name": grief.get("name") or "—",
    }


def _format_employee_grief(grief: dict[str, Any]) -> dict[str, Any]:
    """Форматирует EmployeeAccessGriefDto."""
    grief_obj = grief.get("grief") or grief
    return {
        "id": str(grief_obj.get("id", "")),
        "name": grief_obj.get("name") or "—",
    }


def _format_grief_employee(raw: dict[str, Any]) -> dict[str, Any]:
    """Форматирует EmployeeAccessGriefDto для списка сотрудников с грифом."""
    emp = raw.get("employee") or {}
    post = emp.get("post") or {}
    department = emp.get("department") or {}

    parts = [
        emp.get("lastName") or "",
        emp.get("firstName") or "",
        emp.get("middleName") or "",
    ]
    full_name = " ".join(p for p in parts if p).strip() or "—"

    return {
        "id": str(emp.get("id", "")),
        "full_name": full_name,
        "post": post.get("postName") or "—",
        "department": department.get("name") or "—",
        "active": emp.get("active"),
        "fired": emp.get("fired"),
    }

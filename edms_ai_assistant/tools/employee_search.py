# edms_ai_assistant/tools/employee_search.py
"""
EDMS AI Assistant — Employee Search Tool.

Ключевые принципы:
  1. FTS-first УДАЛЁН — всегда search_employees_post для предсказуемости.
  2. Scoring-based smart-match: каждый результат получает оценку по ВСЕМ
     предоставленным критериям (ФИО + должность). Если лучший результат
     с большим отрывом — возвращаем карточку. Иначе — список.
  3. message НЕ содержит технических инструкций — только текст для пользователя.
     Инструкция для агента — ТОЛЬКО в docstring инструмента.
  4. Формат requires_action + choices — фронтенд рендерит кликабельные карточки.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator

from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.generated.resources_openapi import EmployeeDto
from edms_ai_assistant.services.nlp_service import EDMSNaturalLanguageService
from edms_ai_assistant.services.search_utils import (
    DEFAULT_PAGEABLE,
    build_employee_filter,
    get_merged_name_parts,
    merge_name_parts,
)

logger = logging.getLogger(__name__)

_MAX_PAGE_SIZE: int = 100
_SCORE_GAP_THRESHOLD: int = 5
_VALID_INCLUDES: set[str] = {"POST", "DEPARTMENT"}


# ══════════════════════════════════════════════════════════════════════════════
# Input Schema
# ══════════════════════════════════════════════════════════════════════════════


class EmployeeSearchInput(BaseModel):
    """Полная схема ввода для поиска сотрудников."""

    token: str = Field(..., description="JWT токен авторизации пользователя")

    query: str | None = Field(
        None,
        description=(
            "Универсальная строка поиска. Если вы не уверены, как разбить ФИО на поля, "
            "передайте всё сюда. Будет автоматически разобрано: "
            "'Петров' → last_name='Петров', "
            "'Петров Леонид' → last_name='Петров', first_name='Леонид', "
            "'Иванов Алексей Петрович' → last_name='Иванов', first_name='Алексей', "
            "middle_name='Петрович'. "
            "Если передан query, необязательно передавать last_name/first_name/middle_name."
        ),
    )

    employee_id: str | None = Field(
        None,
        description=(
            "UUID сотрудника для полной карточки. "
            "ОБЯЗАТЕЛЬНО при выборе из списка — берётся из поля id карточки. "
            "НЕ повторяйте поиск по фамилии — это вызовет зацикливание."
        ),
    )

    last_name: str | None = Field(
        None,
        max_length=150,
        description="Фамилия сотрудника. Примеры: 'Иванов', 'Bahdanovich'.",
    )
    first_name: str | None = Field(
        None,
        max_length=100,
        description="Имя сотрудника. Примеры: 'Алексей', 'Tatsiana'.",
    )
    middle_name: str | None = Field(
        None,
        max_length=150,
        description="Отчество сотрудника. Пример: 'Петрович'.",
    )
    full_post_name: str | None = Field(
        None,
        max_length=300,
        description="Название должности. Примеры: 'главный бухгалтер'.",
    )
    post_id: int | None = Field(None, description="ID должности.")

    active_only: bool | None = Field(None, description="True — только активные.")
    fired_only: bool | None = Field(None, description="True — только уволенные.")

    department_names: list[str] | None = Field(None, description="Названия отделов.")
    department_ids: list[str] | None = Field(None, description="UUID отделов.")
    child_departments: bool | None = Field(
        None, description="True — включить дочерние."
    )

    leader_department_name: str | None = Field(None, max_length=300)
    leader_department_id: str | None = Field(
        None, description="UUID отдела-руководство."
    )
    include_child_leaders: bool | None = Field(None)

    leader_department_all_name: str | None = Field(None, max_length=300)
    leader_department_all_id: str | None = Field(None)
    only_leaders: bool | None = Field(None, description="True — только руководители.")

    org_id: str | None = Field(None, description="Идентификатор филиала.")
    employee_ids: list[str] | None = Field(None, description="Список UUID сотрудников.")

    exclude_ids: list[str] | None = Field(None, description="UUID для исключения.")
    exclude_role_id: str | None = Field(None, description="UUID роли для исключения.")
    exclude_group_id: str | None = Field(
        None, description="UUID группы для исключения."
    )
    exclude_personal_group_id: str | None = Field(
        None, description="UUID персональной группы."
    )
    exclude_grief_id: str | None = Field(None, description="UUID грифа для исключения.")

    includes: list[str] | None = Field(
        None, description="POST, DEPARTMENT. По умолчанию оба."
    )
    fetch_all: bool | None = Field(None, description="True — все записи. ⚠ ОПАСНО.")

    page: int | None = Field(None, ge=0, description="Номер страницы.")
    page_size: int | None = Field(None, ge=1, le=100, description="Размер страницы.")

    # ══════════════════════════════════════════════════════════════════════
    # Валидаторы
    # ══════════════════════════════════════════════════════════════════════

    @model_validator(mode="before")
    @classmethod
    def parse_query_to_structured_fields(cls, data: Any) -> Any:
        """Разбирает query в last_name/first_name/middle_name.

        Использует shared parse_name_query + merge_name_parts для
        корректной обработки приоритетов (явные поля > query).
        """
        if not isinstance(data, dict):
            return data

        q = data.get("query")
        if not q or not isinstance(q, str) or not q.strip():
            data["query"] = None
            return data

        existing_last = data.get("last_name")
        if existing_last and isinstance(existing_last, str) and existing_last.strip():
            data["query"] = None
            return data

        if data.get("employee_id"):
            data["query"] = None
            return data

        merged = merge_name_parts(
            name_query=q,
            last_name=data.get("last_name"),
            first_name=data.get("first_name"),
            middle_name=data.get("middle_name"),
        )

        if merged.last_name:
            data["last_name"] = merged.last_name
        if merged.first_name:
            data["first_name"] = merged.first_name
        if merged.middle_name:
            data["middle_name"] = merged.middle_name

        logger.info(
            "Auto-parsed query='%s' → %s",
            q.strip(),
            merged.to_display(),
        )

        data["query"] = None
        return data

    @field_validator(
        "last_name",
        "first_name",
        "middle_name",
        "full_post_name",
        "leader_department_name",
        "leader_department_all_name",
        mode="before",
    )
    @classmethod
    def strip_and_none(cls, v: str | None) -> str | None:
        if v is None:
            return None
        stripped = v.strip()
        return stripped if stripped else None

    @field_validator("includes", mode="before")
    @classmethod
    def validate_includes(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        upper = [item.strip().upper() for item in v if item.strip()]
        invalid = set(upper) - _VALID_INCLUDES
        if invalid:
            raise ValueError(f"Недопустимые includes: {invalid}")
        return upper if upper else None

    @model_validator(mode="after")
    def at_least_one_param(self) -> EmployeeSearchInput:
        fields = [
            self.employee_id,
            self.last_name,
            self.first_name,
            self.middle_name,
            self.full_post_name,
            self.post_id,
            self.active_only,
            self.fired_only,
            self.department_names,
            self.department_ids,
            self.leader_department_name,
            self.leader_department_id,
            self.leader_department_all_name,
            self.leader_department_all_id,
            self.org_id,
            self.employee_ids,
            self.exclude_ids,
            self.exclude_role_id,
            self.exclude_group_id,
            self.exclude_personal_group_id,
            self.exclude_grief_id,
        ]
        if not any(v is not None for v in fields):
            raise ValueError("Укажите хотя бы один параметр поиска.")
        return self


# ══════════════════════════════════════════════════════════════════════════════
# Tool
# ══════════════════════════════════════════════════════════════════════════════


@tool("employee_search_tool", args_schema=EmployeeSearchInput)
async def employee_search_tool(
    token: str,
    query: str | None = None,
    employee_id: str | None = None,
    last_name: str | None = None,
    first_name: str | None = None,
    middle_name: str | None = None,
    full_post_name: str | None = None,
    post_id: int | None = None,
    active_only: bool | None = None,
    fired_only: bool | None = None,
    department_names: list[str] | None = None,
    department_ids: list[str] | None = None,
    child_departments: bool | None = None,
    leader_department_name: str | None = None,
    leader_department_id: str | None = None,
    include_child_leaders: bool | None = None,
    leader_department_all_name: str | None = None,
    leader_department_all_id: str | None = None,
    only_leaders: bool | None = None,
    org_id: str | None = None,
    employee_ids: list[str] | None = None,
    exclude_ids: list[str] | None = None,
    exclude_role_id: str | None = None,
    exclude_group_id: str | None = None,
    exclude_personal_group_id: str | None = None,
    exclude_grief_id: str | None = None,
    includes: list[str] | None = None,
    fetch_all: bool | None = None,
    page: int | None = None,
    page_size: int | None = None,
) -> dict[str, Any]:
    """Searches for employees in the EDMS directory by ANY criteria."""

    nlp = EDMSNaturalLanguageService()

    if employee_id:
        return await _get_employee_card(token, employee_id, nlp)

    merged = get_merged_name_parts(
        name_query=None,  # уже разобран валидатором
        last_name=last_name,
        first_name=first_name,
        middle_name=middle_name,
    )

    employee_filter = build_employee_filter(
        last_name=last_name,
        first_name=first_name,
        middle_name=middle_name,
        full_post_name=full_post_name,
        post_id=post_id,
        active=active_only,
        fired=fired_only,
        includes=includes,
    )

    if merged.first_name and "lastName" in employee_filter:
        logger.info(
            "Search: API filter lastName='%s' only (OR-logic workaround), "
            "scoring will also match firstName='%s'",
            employee_filter.get("lastName"),
            merged.first_name,
        )

    # ── Резолв подразделений ──────────────────────────────────────────────
    resolved_dept_ids: list[str] = list(department_ids or [])
    unresolved_all: list[str] = []

    if department_names:
        newly_resolved, unresolved = await _resolve_department_names(
            token, department_names
        )
        resolved_dept_ids.extend(newly_resolved)
        unresolved_all.extend(unresolved)

    if resolved_dept_ids:
        employee_filter["departmentId"] = resolved_dept_ids
    if child_departments is True:
        employee_filter["childDepartments"] = True

    resolved_leader_id = leader_department_id
    if leader_department_name and not leader_department_id:
        resolved, unresolved = await _resolve_department_names(
            token, [leader_department_name]
        )
        if resolved:
            resolved_leader_id = resolved[0]
        else:
            unresolved_all.extend(unresolved)

    if resolved_leader_id:
        employee_filter["employeeLeaderDepartmentId"] = resolved_leader_id
    if include_child_leaders is True:
        employee_filter["includeChildLeadersEmployeeLeaderDepartmentId"] = True

    resolved_leader_all_id = leader_department_all_id
    if leader_department_all_name and not leader_department_all_id:
        resolved, unresolved = await _resolve_department_names(
            token, [leader_department_all_name]
        )
        if resolved:
            resolved_leader_all_id = resolved[0]
        else:
            unresolved_all.extend(unresolved)

    if resolved_leader_all_id:
        employee_filter["employeeLeaderDepartmentAllId"] = resolved_leader_all_id
    if only_leaders is True:
        employee_filter["onlyLeadersEmployeeLeaderDepartmentAll"] = True

    # ── Остальные параметры ───────────────────────────────────────────────
    if org_id:
        employee_filter["orgId"] = org_id
    if employee_ids:
        employee_filter["ids"] = employee_ids
    if exclude_ids:
        employee_filter["excludeIds"] = exclude_ids
    if exclude_role_id:
        employee_filter["excludeRoleId"] = exclude_role_id
    if exclude_group_id:
        employee_filter["excludeGroupId"] = exclude_group_id
    if exclude_personal_group_id:
        employee_filter["excludePersonalGroupId"] = exclude_personal_group_id
    if exclude_grief_id:
        employee_filter["excludeGriefId"] = exclude_grief_id
    if fetch_all is True:
        employee_filter["all"] = True

    # ── Проверка неразрешённых отделов ────────────────────────────────────
    if unresolved_all:
        logger.warning("Departments not resolved", extra={"unresolved": unresolved_all})
        has_other = any(
            [
                last_name,
                first_name,
                middle_name,
                full_post_name,
                post_id,
                employee_ids,
                resolved_leader_id,
                resolved_leader_all_id,
                org_id,
            ]
        )
        if not resolved_dept_ids and not has_other:
            return {
                "status": "not_found",
                "message": f"Отдел(ы) не найдены: {', '.join(unresolved_all)}.",
            }

    effective_page = page or 0
    effective_size = min(page_size or DEFAULT_PAGEABLE["size"], _MAX_PAGE_SIZE)
    pageable = {
        "page": effective_page,
        "size": effective_size,
        "sort": DEFAULT_PAGEABLE["sort"],
    }

    logger.info(
        "Employee search requested",
        extra={
            "filter_keys": list(employee_filter.keys()),
            "last_name": last_name,
            "first_name": first_name,
            "full_post_name": full_post_name,
        },
    )

    # ── Вызов API ─────────────────────────────────────────────────────────
    try:
        async with EmployeeClient() as client:
            results = await client.search_employees_post(
                token=token,
                employee_filter=employee_filter,
                pageable=pageable,
            )

        if not results:
            return {
                "status": "not_found",
                "message": "Сотрудники по данным критериям не найдены.",
            }

        if len(results) == 1:
            emp_card = await _build_enriched_card(token, results[0], nlp)
            return {"status": "found", "total": 1, "employee_card": emp_card}

        best = _find_best_match(
            results,
            last_name=merged.last_name,
            first_name=merged.first_name,
            middle_name=merged.middle_name,
            full_post_name=full_post_name,
        )
        if best:
            emp_card = await _build_enriched_card(token, best, nlp)
            logger.info(
                "Best match found via scoring",
                extra={"id": str(best.get("id", ""))[:8]},
            )
            return {"status": "found", "total": 1, "employee_card": emp_card}

        display_results = results
        if full_post_name:
            display_results = _filter_by_post(results, full_post_name)
            if len(display_results) == 1:
                emp_card = await _build_enriched_card(token, display_results[0], nlp)
                return {"status": "found", "total": 1, "employee_card": emp_card}

        choices = [
            _serialize_employee_brief(r) for r in display_results[:effective_size]
        ]

        logger.info("Multiple employees found", extra={"count": len(choices)})

        return {
            "status": "requires_action",
            "action_type": "select_employee",
            "message": f"Найдено {len(choices)} сотрудников. Выберите нужного.",
            "total": len(choices),
            "choices": choices,
        }

    except Exception as exc:
        logger.error("Employee search failed", exc_info=True)
        return {"status": "error", "message": f"Ошибка поиска: {exc}"}


# ══════════════════════════════════════════════════════════════════════════════
# Scoring Engine
# ══════════════════════════════════════════════════════════════════════════════


def _score_result(
    result: dict[str, Any],
    last_name: str | None,
    first_name: str | None,
    middle_name: str | None,
    full_post_name: str | None,
) -> int:
    score = 0
    if last_name:
        val = (result.get("lastName") or "").lower()
        term = last_name.lower()
        if val == term:
            score += 10
        elif val.startswith(term):
            score += 5
    if first_name:
        val = (result.get("firstName") or "").lower()
        term = first_name.lower()
        if val == term:
            score += 10
        elif val.startswith(term):
            score += 5
    if middle_name:
        val = (result.get("middleName") or "").lower()
        term = middle_name.lower()
        if val == term:
            score += 10
        elif val.startswith(term):
            score += 5
    if full_post_name:
        post_name = ((result.get("post") or {}).get("postName") or "").lower()
        term = full_post_name.lower()
        if post_name == term:
            score += 10
        elif post_name.startswith(term):
            score += 7
        elif term in post_name:
            score += 3
    return score


def _find_best_match(
    results: list[dict[str, Any]],
    last_name: str | None,
    first_name: str | None,
    middle_name: str | None,
    full_post_name: str | None,
) -> dict[str, Any] | None:
    has_criteria = any([last_name, first_name, middle_name, full_post_name])
    if not has_criteria:
        return None
    scored = [
        (r, _score_result(r, last_name, first_name, middle_name, full_post_name))
        for r in results
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return None
    top_result, top_score = scored[0]
    if top_score < 5:
        return None
    if len(scored) == 1:
        return top_result
    _, second_score = scored[1]
    if top_score - second_score >= _SCORE_GAP_THRESHOLD:
        logger.info(
            "Best match selected: score=%d vs %d (gap=%d)",
            top_score,
            second_score,
            top_score - second_score,
        )
        return top_result
    return None


def _filter_by_post(
    results: list[dict[str, Any]], full_post_name: str
) -> list[dict[str, Any]]:
    term = full_post_name.lower()
    filtered = []
    for r in results:
        post_name = ((r.get("post") or {}).get("postName") or "").lower()
        if post_name == term or post_name.startswith(term) or term in post_name:
            filtered.append(r)
    return filtered if filtered else results


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


async def _build_enriched_card(
    token: str, raw: dict[str, Any], nlp: EDMSNaturalLanguageService
) -> dict[str, Any]:
    emp_id = str(raw.get("id", ""))
    try:
        emp = EmployeeDto.model_validate(raw)
        card = nlp.process_employee_info(emp)
    except Exception:
        logger.warning("NLP formatting failed", exc_info=True)
        card = _serialize_employee_full(raw)
    try:
        async with EmployeeClient() as client:
            roles_raw = await client.get_employee_roles(token, emp_id)
        card["roles"] = [
            {"id": str(r.get("id", "")), "name": r.get("name") or "—"}
            for r in roles_raw
        ]
    except Exception:
        card["roles"] = []
    try:
        async with EmployeeClient() as client:
            griefs_raw = await client.get_employee_griefs(token, emp_id)
        card["access_griefs"] = [
            {
                "id": str((g.get("grief") or g).get("id", "")),
                "name": (g.get("grief") or g).get("name") or "—",
            }
            for g in griefs_raw
        ]
    except Exception:
        card["access_griefs"] = []
    return card


async def _resolve_department_names(
    token: str, names: list[str]
) -> tuple[list[str], list[str]]:
    from edms_ai_assistant.clients.department_client import DepartmentClient

    resolved: list[str] = []
    unresolved: list[str] = []
    async with DepartmentClient() as dept_client:
        for name in names:
            ns = name.strip()
            if not ns:
                continue
            try:
                dept = await dept_client.find_by_name(token, ns)
                if dept and dept.get("id"):
                    resolved.append(str(dept["id"]))
                else:
                    unresolved.append(ns)
            except Exception:
                logger.warning("Failed to resolve department '%s'", ns, exc_info=True)
                unresolved.append(ns)
    return resolved, unresolved


async def _get_employee_card(
    token: str, employee_id: str, nlp: EDMSNaturalLanguageService
) -> dict[str, Any]:
    try:
        async with EmployeeClient() as client:
            raw = await client.get_employee(token, employee_id)
        if not raw:
            return {"status": "not_found", "message": "Сотрудник не найден."}
        card = await _build_enriched_card(token, raw, nlp)
        return {"status": "found", "total": 1, "employee_card": card}
    except Exception as exc:
        logger.error("Failed to fetch employee", exc_info=True)
        return {"status": "error", "message": f"Ошибка получения сотрудника: {exc}"}


def _serialize_employee_brief(raw: dict[str, Any]) -> dict[str, Any]:
    post = raw.get("post") or {}
    department = raw.get("department") or {}
    parts = [
        raw.get("lastName") or "",
        raw.get("firstName") or "",
        raw.get("middleName") or "",
    ]
    full_name = " ".join(p for p in parts if p).strip() or "—"
    return {
        "id": str(raw.get("id", "")),
        "full_name": full_name,
        "post": post.get("postName") or "—",
        "department": department.get("name") or "—",
        "active": raw.get("active"),
        "fired": raw.get("fired"),
    }


def _serialize_employee_full(raw: dict[str, Any]) -> dict[str, Any]:
    post = raw.get("post") or {}
    department = raw.get("department") or {}
    parts = [
        raw.get("lastName") or "",
        raw.get("firstName") or "",
        raw.get("middleName") or "",
    ]
    full_name = " ".join(p for p in parts if p).strip() or "—"
    return {
        "id": str(raw.get("id", "")),
        "full_name": full_name,
        "post": post.get("postName") or "—",
        "department": department.get("name") or "—",
        "active": raw.get("active"),
        "fired": raw.get("fired"),
        "email": raw.get("email") or "—",
        "phone": raw.get("phone") or "—",
        "room": raw.get("room") or "—",
    }

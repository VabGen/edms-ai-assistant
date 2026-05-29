# edms_ai_assistant/tools/employee_search.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.tools import InjectedToolArg, StructuredTool
from langgraph.errors import GraphInterrupt
from pydantic import BaseModel, Field, field_validator, model_validator

from edms_ai_assistant.agent.hitl_primitives import ToolAborted, ask_human
from edms_ai_assistant.agent.interrupt_contract import (
    CardSelectInterrupt,
    CardSelectResume,
    InterruptCard,
)
from edms_ai_assistant.agent.runnable_utils import get_token_from_config
from edms_ai_assistant.services.search_utils import (
    DEFAULT_PAGEABLE,
    build_employee_filter,
    find_best_employee_match,
    get_merged_name_parts,
    merge_name_parts,
)
from langchain_core.runnables import RunnableConfig
if TYPE_CHECKING:
    from edms_ai_assistant.clients.department_client import DepartmentClient
    from edms_ai_assistant.domain.employee import EmployeeDto
    from edms_ai_assistant.clients.employee_client import EmployeeClient
    from edms_ai_assistant.core.deps import AppDeps
    from edms_ai_assistant.services.nlp_service import EDMSNaturalLanguageService

logger = logging.getLogger(__name__)

_MAX_PAGE_SIZE: int = 100
_SCORE_GAP_THRESHOLD: int = 5
_VALID_INCLUDES: set[str] = {"POST", "DEPARTMENT"}


class EmployeeSearchInput(BaseModel):
    """Полная схема ввода для поиска сотрудников."""

    query: str | None = Field(None, description="Универсальная строка поиска.")
    employee_id: str | None = Field(None, description="UUID сотрудника.")
    last_name: str | None = Field(
        None, max_length=150, description="Фамилия сотрудника."
    )
    first_name: str | None = Field(None, max_length=100, description="Имя сотрудника.")
    middle_name: str | None = Field(
        None, max_length=150, description="Отчество сотрудника."
    )
    full_post_name: str | None = Field(
        None, max_length=300, description="Название должности."
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
    includes: list[str] | None = Field(None, description="POST, DEPARTMENT.")
    fetch_all: bool | None = Field(None, description="True — все записи.")
    page: int | None = Field(None, ge=0, description="Номер страницы.")
    page_size: int | None = Field(None, ge=1, le=100, description="Размер страницы.")

    @model_validator(mode="before")
    @classmethod
    def parse_query_to_structured_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        q = data.get("query")
        if not q or not isinstance(q, str) or not q.strip():
            data["query"] = None
            return data
        if isinstance(data.get("last_name"), str) and data["last_name"].strip():
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
        logger.info("Auto-parsed query='%s' -> %s", q.strip(), merged.to_display())
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
# Pure Helper Functions (No DI needed)
# ══════════════════════════════════════════════════════════════════════════════


def _filter_by_post(
    results: list[EmployeeDto], full_post_name: str
) -> list[EmployeeDto]:
    term = full_post_name.lower()
    filtered = []
    for r in results:
        post_name = (r.post.post_name if r.post else "").lower()
        if term in post_name:
            filtered.append(r)
    return filtered if filtered else results


def _serialize_employee_dto(emp: EmployeeDto) -> dict[str, Any]:
    parts = [emp.last_name or "", emp.first_name or "", emp.middle_name or ""]
    full_name = " ".join(p for p in parts if p).strip() or "—"
    return {
        "id": str(emp.id or ""),
        "full_name": full_name,
        "post": emp.post.post_name if emp.post else "—",
        "department": emp.department.name if emp.department else "—",
        "active": emp.active,
        "email": emp.email or "—",
        "phone": emp.phone or "—",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Async Helper Functions (Receive clients as args)
# ══════════════════════════════════════════════════════════════════════════════


async def _resolve_department_names(
    token: str, names: list[str], dept_client: DepartmentClient
) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    unresolved: list[str] = []
    for name in names:
        ns = name.strip()
        if not ns:
            continue
        try:
            dept = await dept_client.find_by_name(token, ns)
            if dept and dept.id:
                resolved.append(str(dept.id))
            else:
                unresolved.append(ns)
        except Exception:
            logger.warning("Failed to resolve department '%s'", ns, exc_info=True)
            unresolved.append(ns)
    return resolved, unresolved


async def _resolve_single_department(
    token: str,
    dept_name: str | None,
    existing_id: str | None,
    dept_client: DepartmentClient,
    unresolved_all: list[str],
) -> str | None:
    """Разрешает имя отдела в UUID. Если existing_id уже задан, пропускает."""
    if not dept_name or existing_id:
        return existing_id

    resolved, unresolved = await _resolve_department_names(
        token, [dept_name], dept_client
    )
    unresolved_all.extend(unresolved)
    return resolved[0] if resolved else None


async def _build_enriched_card(
    token: str,
    emp: EmployeeDto,
    nlp_service: EDMSNaturalLanguageService,
    employee_client: EmployeeClient,
) -> dict[str, Any]:
    emp_id = str(emp.id or "")
    try:
        card = nlp_service.process_employee_info(emp)
    except Exception:
        logger.warning("NLP formatting failed", exc_info=True)
        card = _serialize_employee_dto(emp)

    try:
        roles_raw = await employee_client.get_employee_roles(token, emp_id)
        card["roles"] = [{"id": r.id, "name": r.name or "—"} for r in (roles_raw or [])]
    except Exception:
        card["roles"] = []

    try:
        griefs_raw = await employee_client.get_employee_griefs(token, emp_id)
        access_griefs = []
        for g in griefs_raw or []:
            access_griefs.append(
                {"id": str(g.id or ""), "name": g.access_grief.name or "—"}
            )
        card["access_griefs"] = access_griefs
    except Exception:
        card["access_griefs"] = []

    return card


async def _get_employee_card(
    token: str,
    employee_id: str,
    nlp_service: EDMSNaturalLanguageService,
    employee_client: EmployeeClient,
) -> dict[str, Any]:
    try:
        emp = await employee_client.get_employee(token, employee_id)
        if not emp:
            return {"status": "not_found", "message": "Сотрудник не найден."}
        card = await _build_enriched_card(token, emp, nlp_service, employee_client)
        return {"status": "found", "total": 1, "employee_card": card}
    except Exception as exc:
        logger.error("Failed to fetch employee", exc_info=True)
        return {"status": "error", "message": f"Ошибка получения сотрудника: {exc}"}


async def _resolve_via_ask_human(
    token: str,
    results: list[EmployeeDto],
    choices: list[dict[str, Any]],
    merged_last_name: str | None,
    nlp_service: EDMSNaturalLanguageService,
    employee_client: EmployeeClient,
) -> dict[str, Any]:
    """Disambiguate over native ``ask_human`` and return the picked card."""
    index: dict[str, EmployeeDto] = {str(r.id or ""): r for r in results}

    resume = ask_human(
        CardSelectInterrupt(
            prompt=(
                f"Уточните «{merged_last_name}»"
                if merged_last_name
                else "Уточните сотрудника"
            ),
            cards=[
                InterruptCard(
                    id=brief["id"],
                    label=brief["full_name"],
                    description=brief.get("post") or "Сотрудник",
                    badges=["Сотрудник"] + (["Активен"] if brief.get("active") else []),
                    primary_attrs={
                        "Подразделение": brief.get("department") or "—",
                        "Email": brief.get("email") or "—",
                    },
                    metadata={
                        "active": brief.get("active"),
                        "fired": brief.get("fired"),
                    },
                )
                for brief in choices
            ],
            multiple=False,
        )
    )

    if not isinstance(resume, CardSelectResume):
        raise ToolAborted(
            f"Contract mismatch: expected CardSelectResume, "
            f"got {type(resume).__name__}"
        )

    selected_id = resume.selected_ids[0]
    selected_raw = index.get(selected_id)
    if selected_raw is None:
        logger.warning(
            "Resume value referenced unknown employee id=%s — fetching by id",
            selected_id,
        )
        return await _get_employee_card(
            token, selected_id, nlp_service, employee_client
        )

    card = await _build_enriched_card(token, selected_raw, nlp_service, employee_client)
    return {"status": "found", "total": 1, "employee_card": card}


# ══════════════════════════════════════════════════════════════════════════════
# Tool Factory
# ══════════════════════════════════════════════════════════════════════════════


def create_employee_search_tool(deps: AppDeps) -> StructuredTool:
    """Фабрика инструмента поиска сотрудников с DI."""

    employee_client = deps.employee_client
    nlp_service = deps.nlp_service
    department_client = deps.department_client

    async def employee_search_tool(
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
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Searches for employees in the EDMS directory by ANY criteria.

        ВАЖНО: Токен авторизации передается системой АВТОМАТИЧЕСКИ.
        НЕ запрашивай его у пользователя.
        """
        try:
            token = get_token_from_config(config)
        except Exception as e:
            logger.error(
                "Failed to get token from config: %s | config keys: %s",
                e,
                (
                    list((config or {}).get("configurable", {}).keys())
                    if config
                    else "None"
                ),
            )
            return {
                "status": "error",
                "message": f"Ошибка авторизации: токен не найден в конфигурации запроса. {e}",
            }

        _ = query  # Supress unused warning, processed by Pydantic model_validator

        if employee_id:
            return await _get_employee_card(
                token, employee_id, nlp_service, employee_client
            )

        merged = get_merged_name_parts(
            name_query=None,
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
                "Search: API filter lastName='%s' only, scoring will also match firstName='%s'",
                employee_filter.get("lastName"),
                merged.first_name,
            )

        resolved_dept_ids: list[str] = list(department_ids or [])
        unresolved_all: list[str] = []

        if department_names:
            newly_resolved, unresolved = await _resolve_department_names(
                token, department_names, department_client
            )
            resolved_dept_ids.extend(newly_resolved)
            unresolved_all.extend(unresolved)

        if resolved_dept_ids:
            employee_filter["departmentId"] = resolved_dept_ids
        if child_departments:
            employee_filter["childDepartments"] = True

        resolved_leader_id = await _resolve_single_department(
            token,
            leader_department_name,
            leader_department_id,
            department_client,
            unresolved_all,
        )
        if resolved_leader_id:
            employee_filter["employeeLeaderDepartmentId"] = resolved_leader_id
        if include_child_leaders:
            employee_filter["includeChildLeadersEmployeeLeaderDepartmentId"] = True

        resolved_leader_all_id = await _resolve_single_department(
            token,
            leader_department_all_name,
            leader_department_all_id,
            department_client,
            unresolved_all,
        )
        if resolved_leader_all_id:
            employee_filter["employeeLeaderDepartmentAllId"] = resolved_leader_all_id
        if only_leaders:
            employee_filter["onlyLeadersEmployeeLeaderDepartmentAll"] = True

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
        if fetch_all:
            employee_filter["all"] = True

        if unresolved_all:
            logger.warning(
                "Departments not resolved", extra={"unresolved": unresolved_all}
            )
            has_other = any(
                (
                    last_name,
                    first_name,
                    middle_name,
                    full_post_name,
                    post_id,
                    employee_ids,
                    resolved_leader_id,
                    resolved_leader_all_id,
                    org_id,
                )
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
            extra={"filter_keys": list(employee_filter.keys()), "last_name": last_name},
        )

        try:
            try:
                results = await employee_client.search_employees_post(
                    token=token, employee_filter=employee_filter, pageable=pageable
                )

                if not results:
                    return {
                        "status": "not_found",
                        "message": "Сотрудники по данным критериям не найдены.",
                    }
                if len(results) == 1:
                    return {
                        "status": "found",
                        "total": 1,
                        "employee_card": await _build_enriched_card(
                            token, results[0], nlp_service, employee_client
                        ),
                    }

                best = find_best_employee_match(
                    results,
                    last_name=merged.last_name,
                    first_name=merged.first_name,
                    middle_name=merged.middle_name,
                    full_post_name=full_post_name,
                )
                if best:
                    logger.info(
                        "Best match found via scoring", extra={"id": str(best.id or "")[:8]}
                    )
                    return {
                        "status": "found",
                        "total": 1,
                        "employee_card": await _build_enriched_card(
                            token, best, nlp_service, employee_client
                        ),
                    }

                display_results = results
                if full_post_name:
                    display_results = _filter_by_post(results, full_post_name)
                    if len(display_results) == 1:
                        return {
                            "status": "found",
                            "total": 1,
                            "employee_card": await _build_enriched_card(
                                token, display_results[0], nlp_service, employee_client
                            ),
                        }

                choices = [
                    _serialize_employee_dto(r) for r in display_results[:effective_size]
                ]
                logger.info("Multiple employees found", extra={"count": len(choices)})
                return await _resolve_via_ask_human(
                    token=token,
                    results=display_results,
                    choices=choices,
                    merged_last_name=merged.last_name,
                    nlp_service=nlp_service,
                    employee_client=employee_client,
                )

            except ToolAborted as aborted:
                logger.info("Employee disambiguation aborted: %s", aborted.reason)
                return {
                    "status": "cancelled",
                    "message": "Выбор сотрудника отменён пользователем.",
                }
            except GraphInterrupt:
                raise
            except Exception as exc:
                logger.error("Employee search failed", exc_info=True)
                return {"status": "error", "message": f"Ошибка поиска: {exc}"}
        except GraphInterrupt:
            raise
        except Exception as exc:
            logger.critical("Employee search tool fatal error: %s", exc, exc_info=True)
            return {"status": "error", "message": f"❌ Критическая ошибка при поиске сотрудника: {exc!s}"}

    return StructuredTool.from_function(
        coroutine=employee_search_tool,
        name="employee_search_tool",
        description="Searches for employees in the EDMS directory by ANY criteria. Токен авторизации передается системой АВТОМАТИЧЕСКИ.",
        args_schema=EmployeeSearchInput,
    )

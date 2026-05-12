# edms_ai_assistant/tools/doc_control.py
"""
EDMS AI Assistant — Document Control Tool.

Управление контролем документа: поставить, отредактировать, снять, удалить, получить.

API endpoints (из Java DocumentController + ControlTypeController):

  Справочник типов контроля:
    GET  /api/control-type                     → SliceDto<ControlTypeDto>
        (params: page, size, sort + BasicSearchRequest)

  Контроль документа:
    POST   /api/document/{docId}/control       → ControlDto  (создать)
    PUT    /api/document/{docId}/control       → ControlDto  (редактировать)
    PUT    /api/document/control               → 204         (снять, body: {id: UUID})
    DELETE /api/document/{docId}/control       → 204         (удалить)
    GET    /api/document/{documentId}/control  → ControlDto  (получить; пустой если нет)

Обязательные поля для постановки на контроль (CreateControl):
  controlTypeId       — UUID типа контроля
  controlDateStart    — дата постановки (Instant)
  controlPlanDateEnd  — плановая дата окончания (Instant)
  controlTermDays     — срок в днях (int)
  controlEmployeeId   — UUID контролёра  ← ОБЯЗАТЕЛЬНОЕ (подтверждено ошибкой 400)

Опциональные:
  comment             — комментарий

Права: DOCUMENT_READ (get), DOCUMENT_CONTROL_CREATE (set/edit/remove/delete).
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field, field_validator, model_validator

from edms_ai_assistant.clients.base_client import EdmsHttpClient
from edms_ai_assistant.clients.employee_client import EmployeeClient

logger = logging.getLogger(__name__)

_DEFAULT_CONTROL_DAYS: int = 14


# ─── Date format helpers (Java Instant with ms + Z suffix) ────────────────────


def _to_iso_start(date_str: str) -> str:
    """Convert YYYY-MM-DD to start-of-day Instant: 2026-04-22T00:00:00.000Z"""
    dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    return dt.strftime("%Y-%m-%dT00:00:00.000Z")


def _to_iso_end(date_str: str) -> str:
    """Convert YYYY-MM-DD to end-of-day Instant: 2026-05-06T23:59:59.999Z"""
    dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    return dt.strftime("%Y-%m-%dT23:59:59.999Z")


def _parse_date_only(raw: str) -> str:
    """Extract YYYY-MM-DD from any ISO date or datetime string."""
    raw = raw.strip()
    if "T" in raw:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return raw[:10]


def _has_control_data(data: dict[str, Any]) -> bool:
    """Check if ControlDto represents an actual control (not empty stub).

    Java API returns ``new ControlDto()`` with all-null fields when no control
    exists.  A dict like ``{"id": None, "controlTypeId": None, ...}`` is truthy
    in Python, so we must explicitly check for non-null key fields.
    """
    return bool(data.get("id") or data.get("controlTypeId"))


# ─── HTTP client ──────────────────────────────────────────────────────────────


class _ControlClient(EdmsHttpClient):
    """Thin HTTP client for document control endpoints."""

    # ── Control types ─────────────────────────────────────────────────────────

    async def get_control_types(self, token: str) -> list[dict[str, Any]]:
        """GET /api/control-type — paginated list of control types.

        API returns ``SliceDto<ControlTypeDto>`` with a ``content`` array.
        We request page 0 with a large size to get all types at once.
        """
        result = await self._make_request(
            "GET",
            "api/control-type",
            token=token,
            params={"page": "0", "size": "100"},
        )
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("content") or []
        return []

    # ── Document control CRUD ─────────────────────────────────────────────────

    async def get_control(self, token: str, document_id: str) -> dict[str, Any] | None:
        """GET /api/document/{documentId}/control

        API always returns 200: ``ControlDto`` (populated) or empty
        ``new ControlDto()`` with null fields when no control is set.
        Returns ``None`` when the DTO has no meaningful data.
        """
        result = await self._make_request(
            "GET",
            f"api/document/{document_id}/control",
            token=token,
        )
        if isinstance(result, dict) and _has_control_data(result):
            return result
        return None

    async def set_control(
        self, token: str, document_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /api/document/{docId}/control — create control."""
        result = await self._make_request(
            "POST",
            f"api/document/{document_id}/control",
            token=token,
            json=payload,
        )
        return result if isinstance(result, dict) else {}

    async def edit_control(
        self, token: str, document_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """PUT /api/document/{docId}/control — edit existing control."""
        result = await self._make_request(
            "PUT",
            f"api/document/{document_id}/control",
            token=token,
            json=payload,
        )
        return result if isinstance(result, dict) else {}

    async def remove_control(self, token: str, document_id: str) -> None:
        """PUT /api/document/control — снять с контроля (body: {id: UUID})."""
        await self._make_request(
            "PUT",
            "api/document/control",
            token=token,
            json={"id": document_id},
            is_json_response=False,
        )

    async def delete_control(self, token: str, document_id: str) -> None:
        """DELETE /api/document/{docId}/control — удалить контроль."""
        await self._make_request(
            "DELETE",
            f"api/document/{document_id}/control",
            token=token,
            is_json_response=False,
        )


# ─── Input schema ─────────────────────────────────────────────────────────────


class DocControlInput(BaseModel):
    """Validated input for doc_control tool.

    Обязательные поля для action='set':
      - control_employee_id — контролёр (ФИО или UUID)
      - control_type_name   — тип контроля (авто-выбор если единственный)

    Для action='edit' указываются только изменяемые поля.
    """
    action: str = Field(
        "set",
        description=(
            "Действие: set — поставить на контроль, "
            "edit — отредактировать контроль, "
            "remove — снять с контроля, "
            "delete — удалить контроль, "
            "get — получить данные контроля."
        ),
    )
    date_control_end: str | None = Field(
        None,
        description=(
            "Плановая дата окончания контроля (YYYY-MM-DD или ISO 8601). "
            "Если не указана — +14 дней от сегодня."
        ),
    )
    control_term_days: int | None = Field(
        None,
        ge=1,
        le=3650,
        description="Срок контроля в днях. Приоритет над date_control_end при вычислении.",
    )
    control_type_name: str | None = Field(
        None,
        description=(
            "Название типа контроля. Если не указан и тип единственный — "
            "выбирается автоматически. Если типов несколько — будет выдан список для выбора."
        ),
    )
    control_employee_id: str | None = Field(
        None,
        description=(
            "Контролёр — ФИО (например 'Иванов') или UUID сотрудника. "
            "ОБЯЗАТЕЛЬНО для action=set. Для action=edit — только если нужно изменить."
        ),
    )
    comment: str | None = Field(
        None,
        max_length=500,
        description="Комментарий к контролю (опционально).",
    )

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        allowed = {"set", "edit", "remove", "delete", "get"}
        n = v.strip().lower()
        if n not in allowed:
            raise ValueError(
                f"Недопустимое действие '{v}'. Допустимые: {', '.join(sorted(allowed))}"
            )
        return n

    @model_validator(mode="after")
    def fill_term_days(self) -> "DocControlInput":
        """Ensure control_term_days is always set for action=set.

        Priority:
        1. control_term_days given explicitly → use as-is.
        2. date_control_end given → compute days delta from today.
        3. Neither → use _DEFAULT_CONTROL_DAYS and compute date_control_end.

        For action='edit' we do NOT auto-fill: the user supplies only
        the fields they want to change.
        """
        if self.action != "set" or self.control_term_days is not None:
            return self

        today = datetime.now(UTC).date()

        if self.date_control_end:
            raw = self.date_control_end.strip()
            try:
                end_date = datetime.strptime(_parse_date_only(raw), "%Y-%m-%d").date()
                self.control_term_days = max((end_date - today).days, 1)
            except (ValueError, OverflowError) as exc:
                logger.warning(
                    "Cannot parse date_control_end '%s': %s — using default %d days",
                    raw,
                    exc,
                    _DEFAULT_CONTROL_DAYS,
                )
                self.control_term_days = _DEFAULT_CONTROL_DAYS
        else:
            self.control_term_days = _DEFAULT_CONTROL_DAYS
            self.date_control_end = (
                today + timedelta(days=_DEFAULT_CONTROL_DAYS)
            ).strftime("%Y-%m-%d")

        return self


# ─── Pre-validation helpers ───────────────────────────────────────────────────


def _format_control_types(types: list[dict[str, Any]]) -> str:
    """Format control types list for display to user."""
    lines = []
    for i, ct in enumerate(types, 1):
        name = ct.get("name", "Без названия")
        ct_id = ct.get("id", "?")
        lines.append(f"  {i}. «{name}» (id: {ct_id})")
    return "\n".join(lines)


def _need_input(
    message: str,
    missing_fields: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a standard 'need_input' response so the agent asks the user.

    Args:
        message: Human-readable explanation of what's missing.
        missing_fields: Optional list of field descriptors with available options.

    Returns:
        Dict with status='need_input' and structured data for the agent.
    """
    result: dict[str, Any] = {
        "status": "need_input",
        "message": message,
    }
    if missing_fields:
        result["missing_fields"] = missing_fields
    return result


async def _resolve_control_type(
    client: _ControlClient,
    token: str,
    control_type_name: str | None,
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    """Resolve control type with pre-validation.

    Returns:
        (type_id, type_name, need_input_response)
        If need_input_response is not None — caller must return it immediately.
    """
    ctrl_types = await client.get_control_types(token)

    if not ctrl_types:
        return (
            None,
            None,
            {
                "status": "error",
                "message": "Справочник типов контроля пуст. Обратитесь к администратору EDMS.",
            },
        )

    # ── User specified a type name → try to match ─────────────────────────
    if control_type_name:
        name_lower = control_type_name.lower()
        for ct in ctrl_types:
            ct_name = str(ct.get("name", "")).lower()
            if name_lower in ct_name or ct_name in name_lower:
                return str(ct["id"]), str(ct.get("name", "")), None

        # Not found — show available types
        return (
            None,
            None,
            _need_input(
                message=(
                    f"Тип контроля «{control_type_name}» не найден. "
                    f"Доступные типы контроля:\n" + _format_control_types(ctrl_types)
                ),
                missing_fields=[
                    {
                        "field": "control_type_name",
                        "description": "Название типа контроля",
                        "available_options": [
                            {"id": str(ct["id"]), "name": str(ct.get("name", ""))}
                            for ct in ctrl_types
                        ],
                    }
                ],
            ),
        )

    # ── No type specified ──────────────────────────────────────────────────
    if len(ctrl_types) == 1:
        ct = ctrl_types[0]
        ct_id, ct_name = str(ct["id"]), str(ct.get("name", ""))
        logger.info(
            "Control type auto-selected (single): id=%s... name=%s", ct_id[:8], ct_name
        )
        return ct_id, ct_name, None

    # Multiple types — need user choice
    return (
        None,
        None,
        _need_input(
            message=(
                "Укажите тип контроля. Доступные типы:\n"
                + _format_control_types(ctrl_types)
            ),
            missing_fields=[
                {
                    "field": "control_type_name",
                    "description": "Название типа контроля",
                    "available_options": [
                        {"id": str(ct["id"]), "name": str(ct.get("name", ""))}
                        for ct in ctrl_types
                    ],
                }
            ],
        ),
    )


async def _resolve_employee(
    token: str,
    control_employee_id: str | None,
    *,
    required: bool = True,
    current_employee_id: str | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve controller employee with pre-validation.

    Args:
        token: JWT bearer token.
        control_employee_id: Provided value (UUID string or last name).
        required: If True and no value → return need_input.
        current_employee_id: Fallback for edit (existing controller).

    Returns:
        (employee_uuid, need_input_response)
        If need_input_response is not None — caller must return it immediately.
    """
    # ── Value provided ─────────────────────────────────────────────────────
    if control_employee_id:
        if _is_uuid(control_employee_id):
            return control_employee_id, None

        # Try to resolve by name
        found = await _find_employee(token, control_employee_id)
        if found:
            return found, None

        return None, _need_input(
            message=(
                f"Сотрудник «{control_employee_id}» не найден. "
                "Уточните ФИО контролёра или укажите UUID сотрудника."
            ),
            missing_fields=[
                {
                    "field": "control_employee_id",
                    "description": "ФИО или UUID сотрудника-контролёра",
                }
            ],
        )

    # ── No value provided ──────────────────────────────────────────────────
    if current_employee_id:
        # Edit mode — keep existing controller
        return current_employee_id, None

    if required:
        return None, _need_input(
            message=(
                "Для постановки на контроль необходимо указать контролёра — "
                "сотрудника, ответственного за контроль документа. "
                "Укажите ФИО (например «Иванов») или UUID сотрудника "
                "в параметре control_employee_id."
            ),
            missing_fields=[
                {
                    "field": "control_employee_id",
                    "description": "ФИО или UUID сотрудника-контролёра",
                }
            ],
        )

    return None, None


# ─── Tool ─────────────────────────────────────────────────────────────────────


@tool("doc_control", args_schema=DocControlInput)
async def doc_control(
    action: str = "set",
    date_control_end: str | None = None,
    control_term_days: int | None = None,
    control_type_name: str | None = None,
    control_employee_id: str | None = None,
    comment: str | None = None,
    document_id: Annotated[str, InjectedToolArg] = "",
    token: Annotated[str, InjectedToolArg] = "",
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """Управляет контролем документа.

    Действия:
    - set    — поставить на контроль
    - edit   — отредактировать существующий контроль
    - remove — снять с контроля
    - delete — удалить запись контроля
    - get    — получить текущий контроль

    ОБЯЗАТЕЛЬНЫЕ параметры для set:
    - control_employee_id — ФИО или UUID контролёра
    - control_type_name   — тип контроля (авто-выбор если единственный)

    Если обязательный параметр не указан, инструмент вернёт status='need_input'
    со списком доступных вариантов — уточните у пользователя и вызовите повторно.

    При ошибке 403 проверьте права DOCUMENT_CONTROL_CREATE.

    Args:
        action: set | edit | remove | delete | get.
        date_control_end: Дата окончания контроля (YYYY-MM-DD).
        control_term_days: Срок в днях (приоритет над датой).
        control_type_name: Название типа контроля.
        control_employee_id: ФИО или UUID контролёра.
        comment: Комментарий.
        document_id: UUID документа (инжектируется автоматически).
        token: JWT токен авторизации (инжектируется автоматически).
        config: Конфиг Runnable (инжектируется автоматически).

    Returns:
        Dict со статусом, сообщением и данными контроля.
    """
    logger.info("doc_control: action=%s doc=%s...", action, document_id[:8] if document_id else "N/A")

    try:
        async with _ControlClient() as client:

            # ── GET ────────────────────────────────────────────────────────────
            if action == "get":
                data = await client.get_control(token, document_id)
                if data:
                    return {
                        "status": "success",
                        "message": "Контроль получен.",
                        "control": data,
                    }
                return {
                    "status": "success",
                    "message": "Документ не стоит на контроле.",
                    "control": None,
                }

            # ── REMOVE ─────────────────────────────────────────────────────────
            if action == "remove":
                await client.remove_control(token, document_id)
                return {
                    "status": "success",
                    "message": "✅ Документ снят с контроля.",
                    "requires_reload": True,
                }

            # ── DELETE ─────────────────────────────────────────────────────────
            if action == "delete":
                await client.delete_control(token, document_id)
                return {
                    "status": "success",
                    "message": "✅ Контроль удалён.",
                    "requires_reload": True,
                }

            # ── EDIT ───────────────────────────────────────────────────────────
            if action == "edit":
                current = await client.get_control(token, document_id)
                if not current:
                    return {
                        "status": "error",
                        "message": (
                            "Документ не стоит на контроле — нечего редактировать. "
                            "Используйте action=set для постановки."
                        ),
                    }

                payload: dict[str, Any] = {}

                if current.get("id"):
                    payload["id"] = current["id"]

                # ── Тип контроля ───────────────────────────────────────────────
                if control_type_name:
                    ct_id, ct_name, need_input = await _resolve_control_type(
                        client, token, control_type_name
                    )
                    if need_input:
                        return need_input
                    payload["controlTypeId"] = ct_id
                else:
                    if current.get("controlTypeId"):
                        payload["controlTypeId"] = current["controlTypeId"]

                # ── Контролёр ──────────────────────────────────────────────────
                emp_id, need_input = await _resolve_employee(
                    token,
                    control_employee_id,
                    required=False,
                    current_employee_id=current.get("controlEmployeeId"),
                )
                if need_input:
                    return need_input
                if emp_id:
                    payload["controlEmployeeId"] = emp_id

                # ── Дата / срок ────────────────────────────────────────────────
                today = datetime.now(UTC).date()

                if date_control_end:
                    end_date_only = _parse_date_only(date_control_end)
                    payload["controlPlanDateEnd"] = _to_iso_end(end_date_only)
                    try:
                        end_d = datetime.strptime(end_date_only, "%Y-%m-%d").date()
                        payload["controlTermDays"] = max((end_d - today).days, 1)
                    except ValueError:
                        payload["controlTermDays"] = current.get(
                            "controlTermDays", _DEFAULT_CONTROL_DAYS
                        )
                elif control_term_days and control_term_days >= 1:
                    payload["controlTermDays"] = control_term_days
                    end_date_only = (
                        today + timedelta(days=control_term_days)
                    ).strftime("%Y-%m-%d")
                    payload["controlPlanDateEnd"] = _to_iso_end(end_date_only)
                else:
                    if current.get("controlPlanDateEnd"):
                        payload["controlPlanDateEnd"] = current["controlPlanDateEnd"]
                    if current.get("controlTermDays"):
                        payload["controlTermDays"] = current["controlTermDays"]

                # ── Дата постановки ────────────────────────────────────────────
                if current.get("controlDateStart"):
                    payload["controlDateStart"] = current["controlDateStart"]
                else:
                    payload["controlDateStart"] = _to_iso_start(
                        today.strftime("%Y-%m-%d")
                    )

                # ── Комментарий ────────────────────────────────────────────────
                if comment is not None:
                    payload["comment"] = comment.strip()
                elif current.get("comment") is not None:
                    payload["comment"] = current["comment"]

                logger.info(
                    "doc_control: action=edit doc=%s... payload keys=%s",
                    document_id[:8],
                    list(payload.keys()),
                )

                result = await client.edit_control(token, document_id, payload)

                return {
                    "status": "success",
                    "message": "✅ Контроль отредактирован.",
                    "control": result,
                    "requires_reload": True,
                }

            # ── SET ────────────────────────────────────────────────────────────

            # Step 1: Resolve control type
            ctrl_type_id, ctrl_type_name_resolved, need_input = (
                await _resolve_control_type(client, token, control_type_name)
            )
            if need_input:
                return need_input

            # Step 2: Resolve controller — ОБЯЗАТЕЛЬНО
            resolved_employee_id, need_input = await _resolve_employee(
                token,
                control_employee_id,
                required=True,
            )
            if need_input:
                return need_input

            # Step 3: Compute dates
            today = datetime.now(UTC).date()

            if date_control_end:
                end_date_only = _parse_date_only(date_control_end)
            else:
                days = control_term_days or _DEFAULT_CONTROL_DAYS
                end_date_only = (today + timedelta(days=days)).strftime("%Y-%m-%d")

            if control_term_days and control_term_days >= 1:
                term_days = control_term_days
            else:
                try:
                    end_d = datetime.strptime(end_date_only, "%Y-%m-%d").date()
                    term_days = max((end_d - today).days, 1)
                except ValueError:
                    term_days = _DEFAULT_CONTROL_DAYS

            # Step 4: Build payload — controlEmployeeId ОБЯЗАТЕЛЬНО
            payload: dict[str, Any] = {
                "controlTypeId": ctrl_type_id,
                "controlDateStart": _to_iso_start(today.strftime("%Y-%m-%d")),
                "controlPlanDateEnd": _to_iso_end(end_date_only),
                "controlTermDays": term_days,
                "controlEmployeeId": resolved_employee_id,
            }
            if comment:
                payload["comment"] = comment.strip()

            logger.info(
                "doc_control: action=set doc=%s... type=%s days=%d "
                "controller=%s... start=%s end=%s",
                document_id[:8],
                ctrl_type_name_resolved,
                term_days,
                resolved_employee_id[:8],
                payload["controlDateStart"],
                payload["controlPlanDateEnd"],
            )

            result = await client.set_control(token, document_id, payload)

            return {
                "status": "success",
                "message": (
                    f"✅ Документ поставлен на контроль. "
                    f"Тип: «{ctrl_type_name_resolved}». "
                    f"Срок: {end_date_only} ({term_days} дн.)."
                ),
                "control": result,
                "requires_reload": True,
            }

    except Exception as exc:
        logger.error("doc_control error: %s", exc, exc_info=True)
        error_str = str(exc)

        if "403" in error_str or "NO_ACCESS" in error_str:
            hint = (
                "У пользователя нет прав DOCUMENT_CONTROL_CREATE. "
                "Обратитесь к администратору EDMS для назначения прав."
            )
        elif "controlTermDays" in error_str:
            hint = "API требует поле controlTermDays. Укажите дату или количество дней."
        elif "controlEmployeeId" in error_str:
            hint = (
                "API требует поле controlEmployeeId (контролёр). "
                "Укажите ФИО или UUID сотрудника в параметре control_employee_id."
            )
        elif "400" in error_str:
            hint = f"Ошибка валидации от API: {error_str[:300]}"
        else:
            hint = error_str[:300]

        return {"status": "error", "message": f"❌ Ошибка контроля: {hint}"}


# ─── Private helpers ──────────────────────────────────────────────────────────


def _is_uuid(value: str) -> bool:
    """Return True if value is a valid UUID string."""
    return bool(
        re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            value.strip(),
            re.IGNORECASE,
        )
    )


async def _find_employee(token: str, last_name: str) -> str | None:
    """Resolve employee UUID by last name via FTS (top-1 result).

    Returns None if not found or ambiguous — let agent handle disambiguation.
    """
    try:
        async with EmployeeClient() as emp_client:
            emp = await emp_client.find_by_last_name_fts(token, last_name)
            if emp and emp.get("id"):
                return str(emp["id"])
    except Exception as exc:
        logger.debug("Employee FTS failed for '%s': %s", last_name, exc)
    return None

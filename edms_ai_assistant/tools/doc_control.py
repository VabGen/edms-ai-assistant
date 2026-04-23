# edms_ai_assistant/tools/doc_control.py
"""
EDMS AI Assistant — Document Control Tool.

Управление контролем документа: поставить, снять, удалить, получить.

Поля API (подтверждены логами):
  controlTypeId       — UUID типа контроля (обязательное)
  controlDateStart    — дата постановки (Instant, обязательное)
  controlPlanDateEnd  — плановая дата окончания (Instant, обязательное)
  controlTermDays     — срок в днях (int, обязательное — was missing in v1)
  controlEmployeeId   — UUID контролёра (опциональное)
  comment             — комментарий (опциональное)

Права: пользователь должен иметь DOCUMENT_CONTROL_CREATE.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator

from edms_ai_assistant.clients.base_client import EdmsHttpClient
from edms_ai_assistant.clients.employee_client import EmployeeClient

logger = logging.getLogger(__name__)

_DEFAULT_CONTROL_DAYS: int = 14  # default when no date/days provided

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
    """Extract YYYY-MM-DD from any ISO date or datetime string.

    Args:
        raw: Date string in YYYY-MM-DD or full ISO 8601 format.

    Returns:
        10-char YYYY-MM-DD string.
    """
    raw = raw.strip()
    if "T" in raw:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return raw[:10]


# ─── HTTP client ──────────────────────────────────────────────────────────────


class _ControlClient(EdmsHttpClient):
    """Thin HTTP client for document control endpoints."""

    async def get_control_types(self, token: str) -> list[dict[str, Any]]:
        """GET /api/control-type?listAttribute=true"""
        result = await self._make_request(
            "GET",
            "api/control-type",
            token=token,
            params={"listAttribute": "true"},
        )
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("content") or []
        return []

    async def resolve_control_type(
        self, token: str, name: str | None = None
    ) -> tuple[str | None, str | None]:
        """Resolve control type to (id, name).

        Searches by partial name match; falls back to first available type.

        Args:
            token: JWT bearer token.
            name: Desired type name (partial, case-insensitive). None → first.

        Returns:
            Tuple (ctrl_type_id, ctrl_type_name) or (None, None) if empty.
        """
        types = await self.get_control_types(token)
        if not types:
            return None, None

        if name:
            name_lower = name.lower()
            for ct in types:
                ct_name = str(ct.get("name", "")).lower()
                if name_lower in ct_name or ct_name in name_lower:
                    return str(ct["id"]), str(ct.get("name", ""))

        first = types[0]
        cid, cname = str(first["id"]), str(first.get("name", ""))
        logger.info("Control type fallback (first): id=%s... name=%s", cid[:8], cname)
        return cid, cname

    async def set_control(
        self, token: str, document_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /api/document/{documentId}/control"""
        result = await self._make_request(
            "POST",
            f"api/document/{document_id}/control",
            token=token,
            json=payload,
        )
        return result if isinstance(result, dict) else {}

    async def remove_control(self, token: str, document_id: str) -> None:
        """PUT /api/document/control (снять с контроля)"""
        await self._make_request(
            "PUT",
            "api/document/control",
            token=token,
            json={"id": document_id},
            is_json_response=False,
        )

    async def delete_control(self, token: str, document_id: str) -> None:
        """DELETE /api/document/{documentId}/control"""
        await self._make_request(
            "DELETE",
            f"api/document/{document_id}/control",
            token=token,
            is_json_response=False,
        )

    async def get_control(self, token: str, document_id: str) -> dict[str, Any] | None:
        """GET /api/document/{documentId}/control"""
        result = await self._make_request(
            "GET",
            f"api/document/{document_id}/control",
            token=token,
        )
        return result if isinstance(result, dict) and result else None


# ─── Input schema ─────────────────────────────────────────────────────────────


class DocControlInput(BaseModel):
    """Validated input for doc_control tool.

    All date/days fields are optional — the validator fills defaults
    so that controlTermDays is never None when action='set'.
    """

    token: str = Field(..., description="JWT токен авторизации")
    document_id: str = Field(..., description="UUID документа")
    action: str = Field(
        "set",
        description=(
            "Действие: set — поставить на контроль, "
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
        description=(
            "Срок контроля в днях. " "Приоритет над date_control_end при вычислении."
        ),
    )
    control_type_name: str | None = Field(
        None,
        description="Название типа контроля. По умолчанию — первый доступный.",
    )
    control_employee_id: str | None = Field(
        None,
        description="UUID сотрудника-контролёра (опционально).",
    )
    comment: str | None = Field(
        None,
        max_length=500,
        description="Комментарий к контролю (опционально).",
    )

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        allowed = {"set", "remove", "delete", "get"}
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


# ─── Tool ─────────────────────────────────────────────────────────────────────


@tool("doc_control", args_schema=DocControlInput)
async def doc_control(
    token: str,
    document_id: str,
    action: str = "set",
    date_control_end: str | None = None,
    control_term_days: int | None = None,
    control_type_name: str | None = None,
    control_employee_id: str | None = None,
    comment: str | None = None,
) -> dict[str, Any]:
    """Управляет постановкой документа на контроль.

    Действия:
    - set    — поставить на контроль
    - remove — снять с контроля
    - delete — удалить запись контроля
    - get    — получить текущий контроль

    ВАЖНО:
    - controlTermDays вычисляется автоматически из date_control_end.
      Если дата не указана — срок +14 дней от сегодня.
    - При ошибке 403 проверьте права пользователя в EDMS:
      требуется разрешение DOCUMENT_CONTROL_CREATE.

    Args:
        token: JWT токен авторизации.
        document_id: UUID документа.
        action: set | remove | delete | get.
        date_control_end: Дата окончания контроля (YYYY-MM-DD).
        control_term_days: Срок в днях (приоритет над датой).
        control_type_name: Название типа контроля.
        control_employee_id: UUID контролёра.
        comment: Комментарий.

    Returns:
        Dict со статусом, сообщением и данными контроля.
    """
    logger.info("doc_control: action=%s doc=%s...", action, document_id[:8])

    try:
        async with _ControlClient() as client:

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

            if action == "remove":
                await client.remove_control(token, document_id)
                return {
                    "status": "success",
                    "message": "✅ Документ снят с контроля.",
                    "requires_reload": True,
                }

            if action == "delete":
                await client.delete_control(token, document_id)
                return {
                    "status": "success",
                    "message": "✅ Контроль удалён.",
                    "requires_reload": True,
                }

            # ── SET ────────────────────────────────────────────────────────────

            ctrl_type_id, ctrl_type_name_resolved = await client.resolve_control_type(
                token, control_type_name
            )
            if not ctrl_type_id:
                return {
                    "status": "error",
                    "message": "Не удалось найти тип контроля в справочнике.",
                }

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

            resolved_employee_id: str | None = control_employee_id
            if control_employee_id and not _is_uuid(control_employee_id):
                resolved_employee_id = await _find_employee(token, control_employee_id)

            payload: dict[str, Any] = {
                "controlTypeId": ctrl_type_id,
                "controlDateStart": _to_iso_start(today.strftime("%Y-%m-%d")),
                "controlPlanDateEnd": _to_iso_end(end_date_only),
                "controlTermDays": term_days,
            }
            if resolved_employee_id:
                payload["controlEmployeeId"] = resolved_employee_id
            if comment:
                payload["comment"] = comment.strip()

            logger.info(
                "doc_control: action=set doc=%s... type=%s days=%d " "start=%s end=%s",
                document_id[:8],
                ctrl_type_name_resolved,
                term_days,
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

    Args:
        token: JWT bearer token.
        last_name: Employee last name string.

    Returns:
        UUID string or None.
    """
    try:
        async with EmployeeClient() as emp_client:
            emp = await emp_client.find_by_last_name_fts(token, last_name)
            if emp and emp.get("id"):
                return str(emp["id"])
    except Exception as exc:
        logger.debug("Employee FTS failed for '%s': %s", last_name, exc)
    return None

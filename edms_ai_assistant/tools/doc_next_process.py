"""
EDMS AI Assistant — Document Next Process Tool.

Перевод документа на следующий этап бизнес-процесса.

API:
  POST /api/document/process/next
  Body: {id: UUID, nextId: UUID, employees: [UUID]}

  id        = documentId — UUID документа
  nextId    = DocumentProcessDto.currentId — UUID текущего этапа
              (optimistic lock: сервер проверяет, что текущий этап
               не изменился с момента загрузки данных)
  employees = список UUID исполнителей (обязателен для многих этапов)

Логика:
  1. GET /{id}/bpmn              → порядок и названия этапов
  2. GET /{documentId}/process    → currentId (optimistic lock)
  3. Если сотрудники не указаны — попытка перехода
  4. При ошибке employee.empty — запросить исполнителей у пользователя
  5. POST /process/next с employees
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from typing import Any, Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.clients.base_client import EdmsHttpClient
from edms_ai_assistant.clients.employee_client import EmployeeClient

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _is_uuid(value: str) -> bool:
    return bool(_UUID_RE.match(str(value).strip()))


PROCESS_TYPE_LABELS: dict[str, str] = {
    "NEW": "Новый",
    "REGISTRATION": "Зарегистрировать",
    "REVIEW": "Рассмотреть",
    "EXECUTION": "Исполнить",
    "SENT": "Отправить адресатам",
    "AGREEMENT": "Согласовать",
    "SIGNING": "Подписать",
    "STATEMENT": "Утвердить",
    "DISPATCH": "Отправить",
    "PREPARATION": "Подготовить",
    "PAPERWORK": "Оформить",
    "ACCEPTANCE": "Одобрить",
    "CONTRACT_EXECUTION": "Исполнить договор",
}


# ─── Process step model ──────────────────────────────────────────────────────


class ProcessStep(BaseModel):
    name: str
    process_type: str | None = None
    bpmn_id: str | None = None
    is_current: bool = False
    is_completed: bool = False

    @property
    def action_label(self) -> str:
        if self.process_type and self.process_type in PROCESS_TYPE_LABELS:
            return PROCESS_TYPE_LABELS[self.process_type]
        return self.name


# ─── BPMN parser ──────────────────────────────────────────────────────────────


def _parse_bpmn_parsed(bpmn_data: dict[str, Any]) -> list[ProcessStep]:
    """Парсит BPMN-данные в список этапов процесса."""
    parsed_items = bpmn_data.get("parsed") or []
    current_activities = bpmn_data.get("activities") or []
    history = bpmn_data.get("history") or []

    current_ids = {
        str(a.get("activityId", "")) for a in current_activities if isinstance(a, dict)
    }
    history_set = {str(h) for h in history}

    steps: list[ProcessStep] = []
    for item in parsed_items:
        if not isinstance(item, dict):
            continue
        step_type = str(item.get("type", ""))
        if step_type != "USER_TASK":
            continue

        bpmn_id = str(item.get("id", ""))
        name = str(item.get("name", "") or bpmn_id)
        process_type = item.get("processType")

        steps.append(
            ProcessStep(
                name=name,
                process_type=str(process_type) if process_type else None,
                bpmn_id=bpmn_id,
                is_current=bpmn_id in current_ids,
                is_completed=bpmn_id in history_set and bpmn_id not in current_ids,
            )
        )
    return steps


def _get_next_available(steps: list[ProcessStep]) -> list[ProcessStep]:
    """Возвращает этапы, доступные для перехода."""
    current_idx = -1
    for i, step in enumerate(steps):
        if step.is_current:
            current_idx = i
            break
    if current_idx == -1:
        return [s for s in steps if not s.is_completed and not s.is_current]
    return [s for s in steps[current_idx + 1 :] if not s.is_completed]


def _format_step_list(steps: list[ProcessStep], show_status: bool = True) -> str:
    lines = []
    for i, step in enumerate(steps, 1):
        status = ""
        if show_status:
            if step.is_current:
                status = " ← вы здесь"
            elif step.is_completed:
                status = " ✓ пройден"
        type_hint = (
            f" ({step.action_label})"
            if step.process_type and not step.is_current
            else ""
        )
        lines.append(f"  {i}. «{step.name}»{type_hint}{status}")
    return "\n".join(lines)


def _resolve_step_selection(
    steps: list[ProcessStep], selection: str
) -> ProcessStep | None:
    """Разрешает выбор этапа по номеру, названию или типу."""
    selection = selection.strip()
    if selection.isdigit():
        idx = int(selection) - 1
        if 0 <= idx < len(steps):
            return steps[idx]
    sel_lower = selection.lower()
    for step in steps:
        if step.process_type and step.process_type.lower() == sel_lower:
            return step
    for step in steps:
        if step.name.lower() == sel_lower:
            return step
    for step in steps:
        name_lower = step.name.lower()
        if sel_lower in name_lower or name_lower in sel_lower:
            return step
    return None


# ─── HTTP client ──────────────────────────────────────────────────────────────


class _ProcessClient(EdmsHttpClient):

    async def get_bpmn_activity(
        self, token: str, document_id: str
    ) -> dict[str, Any] | None:
        """GET /api/document/{id}/bpmn — BPMN-схема и активности."""
        result = await self._make_request(
            "GET",
            f"api/document/{document_id}/bpmn",
            token=token,
        )
        return result if isinstance(result, dict) and result else None

    async def get_document(self, token: str, document_id: str) -> dict[str, Any] | None:
        """GET /api/document/{id} — данные документа."""
        result = await self._make_request(
            "GET",
            f"api/document/{document_id}",
            token=token,
        )
        return result if isinstance(result, dict) and result else None

    async def get_process(self, token: str, document_id: str) -> dict[str, Any] | None:
        """GET /api/document/{documentId}/process — данные процесса."""
        result = await self._make_request(
            "GET",
            f"api/document/{document_id}/process",
            token=token,
        )
        return result if isinstance(result, dict) and result else None

    async def next_process(
        self,
        token: str,
        document_id: str,
        next_id: str,
        employees: list[str] | None = None,
    ) -> None:
        """POST /api/document/process/next — переход на следующий этап.

        Args:
            document_id: UUID документа (поле "id" в body).
            next_id: DocumentProcessDto.currentId — UUID текущего этапа
                     (поле "nextId" в body, optimistic lock).
            employees: Список UUID исполнителей (опционально, но может
                       быть обязателен для некоторых этапов).
        """
        payload: dict[str, Any] = {
            "id": document_id,
            "nextId": next_id,
        }
        if employees:
            payload["employees"] = employees
        await self._make_request(
            "POST",
            "api/document/process/next",
            token=token,
            json=payload,
            is_json_response=False,
        )


# ─── Input schema ─────────────────────────────────────────────────────────────


class DocNextProcessInput(BaseModel):
    next_step: str | None = Field(
        None,
        description=(
            "Название или номер этапа. " "Примеры: '1', 'Регистрация', 'REVIEW'."
        ),
    )
    employees: list[str] | None = Field(
        None,
        description=(
            "Список ФИО или UUID исполнителей для следующего этапа. "
            "Многие этапы (Регистрация, Рассмотрение, Исполнение и др.) "
            "требуют указания хотя бы одного исполнителя."
        ),
    )

    @field_validator("employees", mode="before")
    @classmethod
    def normalize_employees(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(e) for e in v if e]
        return None


# ─── Employee resolution ──────────────────────────────────────────────────────


async def _resolve_employees(
    token: str, employees: list[str]
) -> tuple[list[str], list[str]]:
    """Разрешает список сотрудников: UUID → как есть, ФИО → поиск."""
    resolved: list[str] = []
    unresolved: list[str] = []
    for emp in employees:
        emp = emp.strip()
        if _is_uuid(emp):
            resolved.append(emp)
        else:
            found = await _find_employee(token, emp)
            if found:
                resolved.append(found)
            else:
                unresolved.append(emp)
    return resolved, unresolved


async def _find_employee(token: str, last_name: str) -> str | None:
    try:
        async with EmployeeClient() as emp_client:
            emp = await emp_client.find_by_last_name_fts(token, last_name)
            if emp and emp.get("id"):
                return str(emp["id"])
    except Exception as exc:
        logger.debug("Employee FTS failed for '%s': %s", last_name, exc)
    return None


# ─── nextId resolution ────────────────────────────────────────────────────────


def _resolve_next_id(
    process_data: dict[str, Any] | None,
    doc_data: dict[str, Any] | None,
) -> str | None:
    """Определяет nextId для POST /process/next.

    nextId = DocumentProcessDto.currentId — UUID текущего этапа процесса.
    Сервер использует его как optimistic lock.
    """
    if process_data:
        # ── Вариант 1: currentId ───────────────────────────────────────
        current_id = process_data.get("currentId")
        if current_id and _is_uuid(str(current_id)):
            cid = str(current_id)
            logger.info("nextId from DocumentProcessDto.currentId: %s", cid[:8])
            return cid

        # ── Вариант 2: ID текущего/завершённого item ──────────────────
        items = (
            process_data.get("items")
            or process_data.get("processItems")
            or process_data.get("stages")
            or []
        )
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("started") and not item.get("completed"):
                item_id = str(item.get("id", "")).strip()
                if item_id and _is_uuid(item_id):
                    logger.info(
                        "nextId from active item: %s",
                        item_id[:8],
                    )
                    return item_id

        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("started") and item.get("completed"):
                item_id = str(item.get("id", "")).strip()
                if item_id and _is_uuid(item_id):
                    logger.info(
                        "nextId from completed current item: %s",
                        item_id[:8],
                    )
                    return item_id

        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", "")).strip()
            if item_id and _is_uuid(item_id):
                logger.info("nextId from first item: %s", item_id[:8])
                return item_id

    # ── Вариант 3: processId (LAST RESORT) ────────────────────────────
    if doc_data:
        pid = doc_data.get("processId")
        if pid and _is_uuid(str(pid)):
            process_id = str(pid)
            logger.warning(
                "nextId LAST RESORT from DocumentDto.processId: %s",
                process_id[:8],
            )
            return process_id

    return None


# ─── Error parsing ────────────────────────────────────────────────────────────


def _parse_api_error(exc: Exception) -> dict[str, Any]:
    """Разбирает ошибку API и возвращает структурированную информацию."""
    error_str = str(exc)
    result: dict[str, Any] = {
        "is_employee_empty": False,
        "is_process_changed": False,
        "is_forbidden": False,
        "is_bad_request": False,
        "target_step_name": None,
        "target_step_type": None,
        "raw": error_str,
    }

    try:
        # httpx оборачивает ошибку, пробуем найти JSON в сообщении
        if "employee.empty" in error_str:
            result["is_employee_empty"] = True
        if "process.changed" in error_str:
            result["is_process_changed"] = True
        if "403" in error_str or "NO_ACCESS" in error_str:
            result["is_forbidden"] = True
        if "400" in error_str:
            result["is_bad_request"] = True

        if result["is_employee_empty"]:
            # Ищем кириллицу (название этапа) и uppercase (тип этапа)
            parts = re.findall(r"'([^']*)'", error_str)
            for part in parts:
                if re.search(r"[а-яА-Я]", part):
                    result["target_step_name"] = part
                elif part.isupper() and len(part) > 2 and "_" not in part:
                    result["target_step_type"] = part
    except Exception:
        pass

    return result


# ─── Tool ─────────────────────────────────────────────────────────────────────


@tool("doc_next_process", args_schema=DocNextProcessInput)
async def doc_next_process(
    next_step: str | None = None,
    employees: list[str] | None = None,
    document_id: Annotated[str, InjectedToolArg] = "",
    token: Annotated[str, InjectedToolArg] = "",
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """Переводит документ на следующий этап бизнес-процесса.

    Пользователь выбирает этап по названию или номеру.
    Если next_step не указан — показывает список этапов.

    Многие этапы (Регистрация, Рассмотрение, Исполнение и др.)
    требуют указания исполнителей. Если исполнители не указаны,
    инструмент запросит их у пользователя.

    Примеры: next_step="1", next_step="Регистрация", next_step="REVIEW"

    Args:
        next_step: Название или номер этапа.
        employees: Список ФИО или UUID исполнителей.
        document_id: UUID документа (инжектируется автоматически).
        token: JWT токен авторизации (инжектируется автоматически).
        config: Конфиг Runnable (инжектируется автоматически).

    Returns:
        Dict со статусом и результатом.
    """
    logger.info(
        "doc_next_process: doc=%s next_step=%s employees=%s",
        document_id[:8] if document_id else "N/A",
        next_step,
        employees if employees else "none",
    )

    try:
        async with _ProcessClient() as client:

            # ── Шаг 1: Получить BPMN, документ и процесс параллельно ─────
            bpmn_data, doc_data, process_data = await asyncio.gather(
                client.get_bpmn_activity(token, document_id),
                client.get_document(token, document_id),
                client.get_process(token, document_id),
            )

            if not bpmn_data:
                return {
                    "status": "error",
                    "message": (
                        "Не удалось получить информацию о процессе документа. "
                        "Возможно, документ ещё не запущен в работу."
                    ),
                }

            # ── Шаг 2: Парсим BPMN ────────────────────────────────────────
            bpmn_steps = _parse_bpmn_parsed(bpmn_data)

            for k, v in bpmn_data.items():
                if k == "xml":
                    continue
                try:
                    s = json.dumps(v, ensure_ascii=False, default=str)
                    logger.info("BPMN '%s': %s", k, s[:500])
                except Exception:
                    pass

            logger.info(
                "BPMN steps: %s",
                [(s.name, s.process_type, s.is_current) for s in bpmn_steps],
            )

            # ── Шаг 3: Логируем данные процесса ───────────────────────────
            if process_data:
                current_id = process_data.get("currentId")
                next_id_dto = process_data.get("nextId")
                proc_id = process_data.get("id")
                items = process_data.get("items") or []

                logger.info(
                    "Process: id=%s currentId=%s nextId=%s items_count=%d",
                    str(proc_id)[:8] if proc_id else None,
                    str(current_id)[:8] if current_id else None,
                    str(next_id_dto)[:8] if next_id_dto else None,
                    len(items),
                )
                for item in items:
                    if isinstance(item, dict):
                        logger.info(
                            "  Item: id=%s type=%s started=%s completed=%s "
                            "processId=%s taskDefKey=%s",
                            str(item.get("id", ""))[:8],
                            item.get("type"),
                            item.get("started"),
                            item.get("completed"),
                            str(item.get("processId", ""))[:8],
                            item.get("taskDefinitionKey"),
                        )
            else:
                logger.warning(
                    "GET /process returned no data for document %s",
                    document_id[:8],
                )

            # ── Шаг 4: Определяем текущий и следующие этапы ───────────────
            current_step = next(
                (s for s in bpmn_steps if s.is_current),
                None,
            )
            next_available = _get_next_available(bpmn_steps)

            logger.info(
                "Current='%s', next_available=%s",
                current_step.name if current_step else "?",
                [(s.name, s.process_type) for s in next_available],
            )

            # ── Шаг 5: Показать список (если шаг не выбран) ──────────────
            if not next_step:
                if not bpmn_steps:
                    return {
                        "status": "error",
                        "message": "Этапы процесса не найдены.",
                    }

                full_list = _format_step_list(bpmn_steps, show_status=True)
                next_list = ""
                if next_available:
                    next_list = "\n\nДоступные для перехода:\n" + _format_step_list(
                        next_available, show_status=False
                    )

                current_info = ""
                if current_step:
                    current_info = f"\nСейчас документ на этапе: «{current_step.name}»."

                return {
                    "status": "need_input",
                    "message": (
                        f"Маршрут документа:{current_info}\n\n"
                        f"Все этапы:\n{full_list}"
                        f"{next_list}\n\n"
                        "Напишите название или номер этапа, на который "
                        "нужно перейти. Для большинства этапов также "
                        "потребуется указать исполнителя (ФИО)."
                    ),
                    "available_steps": [
                        {
                            "number": i + 1,
                            "name": s.name,
                            "process_type": s.process_type,
                            "action": s.action_label,
                        }
                        for i, s in enumerate(next_available)
                    ],
                }

            # ── Шаг 6: Разрешить выбор этапа ─────────────────────────────
            selected = _resolve_step_selection(next_available, next_step)
            if not selected:
                selected = _resolve_step_selection(bpmn_steps, next_step)

            if not selected:
                next_list = _format_step_list(next_available, show_status=False)
                return {
                    "status": "need_input",
                    "message": (
                        f"Этап «{next_step}» не найден.\n\n"
                        f"Доступные этапы:\n{next_list}\n\n"
                        "Напишите название или номер этапа."
                    ),
                    "available_steps": [
                        {
                            "number": i + 1,
                            "name": s.name,
                            "process_type": s.process_type,
                        }
                        for i, s in enumerate(next_available)
                    ],
                }

            if selected.is_current:
                return {
                    "status": "error",
                    "message": f"Документ уже на этапе «{selected.name}».",
                }
            if selected.is_completed:
                return {
                    "status": "error",
                    "message": f"Этап «{selected.name}» уже пройден.",
                }

            # ── Шаг 7: Определить nextId (optimistic lock) ────────────────
            next_id = _resolve_next_id(process_data, doc_data)

            if not next_id:
                return {
                    "status": "error",
                    "message": (
                        f"Не удалось перейти на этап «{selected.name}». "
                        "Не найден идентификатор текущего этапа процесса. "
                        "Попробуйте через основной интерфейс EDMS."
                    ),
                }

            # ── Шаг 8: Резолв сотрудников ─────────────────────────────────
            resolved_employees: list[str] | None = None
            if employees:
                resolved, unresolved = await _resolve_employees(
                    token,
                    employees,
                )
                if unresolved:
                    return {
                        "status": "need_input",
                        "message": (
                            "Не удалось найти сотрудников: "
                            + ", ".join(f"«{u}»" for u in unresolved)
                            + ". Уточните ФИО."
                        ),
                    }
                resolved_employees = resolved

            # ── Шаг 9: Выполнить переход ──────────────────────────────────
            logger.info(
                "POST /process/next: doc=%s nextId=%s target='%s' type=%s employees=%d",
                document_id[:8],
                next_id[:8],
                selected.name,
                selected.process_type,
                len(resolved_employees) if resolved_employees else 0,
            )

            try:
                await client.next_process(
                    token,
                    document_id,
                    next_id,
                    resolved_employees,
                )
            except Exception as post_exc:
                post_error = _parse_api_error(post_exc)

                # ── Ошибка: нужны исполнители ──────────────────────────
                if post_error["is_employee_empty"]:
                    step_label = post_error["target_step_name"] or selected.name
                    step_type = (
                        post_error["target_step_type"] or selected.process_type or ""
                    )

                    logger.info(
                        "Employee required for step '%s' (%s), " "requesting from user",
                        step_label,
                        step_type,
                    )

                    return {
                        "status": "need_input",
                        "message": (
                            f"⚠ Для перехода на этап «{step_label}» "
                            f"необходимо указать исполнителя.\n\n"
                            "Напишите ФИО сотрудника, которого нужно "
                            "назначить исполнителем на этом этапе. "
                            "Можно указать нескольких через запятую.\n\n"
                            "Пример: «Иванов И.И.» или "
                            "«Иванов И.И., Петров П.П.»"
                        ),
                        "need_employees_for": {
                            "step_name": step_label,
                            "step_type": step_type,
                            "document_id": document_id,
                            "next_step": next_step,
                        },
                    }

                # ── Ошибка: процесс изменён ────────────────────────────
                if post_error["is_process_changed"]:
                    return {
                        "status": "error",
                        "message": (
                            "Процесс документа был изменён с момента "
                            "последней загрузки. Попробуйте ещё раз — "
                            "данные обновятся автоматически."
                        ),
                    }

                # ── Ошибка: нет прав ────────────────────────────────────
                if post_error["is_forbidden"]:
                    return {
                        "status": "error",
                        "message": ("У вас нет прав для перехода на следующий этап."),
                    }

                # ── Другая ошибка ───────────────────────────────────────
                raise

            # ── Успех ──────────────────────────────────────────────────────
            emp_info = ""
            if resolved_employees:
                emp_info = f" Назначены исполнители: {len(resolved_employees)} чел."

            return {
                "status": "success",
                "message": (
                    f"✅ Документ переведён на этап «{selected.name}» "
                    f"({selected.action_label}).{emp_info}"
                ),
                "requires_reload": True,
            }

    except Exception as exc:
        logger.error("doc_next_process error: %s", exc, exc_info=True)

        error_info = _parse_api_error(exc)
        if error_info["is_employee_empty"]:
            step_label = error_info["target_step_name"] or "следующий"
            return {
                "status": "need_input",
                "message": (
                    f"⚠ Для перехода на этап «{step_label}» "
                    f"необходимо указать исполнителя.\n\n"
                    "Напишите ФИО сотрудника, которого нужно "
                    "назначить исполнителем на этом этапе."
                ),
            }
        if error_info["is_process_changed"]:
            return {
                "status": "error",
                "message": ("Процесс документа был изменён. " "Попробуйте ещё раз."),
            }
        if error_info["is_forbidden"]:
            return {
                "status": "error",
                "message": "У вас нет прав для перехода на следующий этап.",
            }

        return {
            "status": "error",
            "message": "❌ Произошла неожиданная ошибка при переходе.",
        }

    # 5

# edms_ai_assistant/tools/doc_next_process.py
"""
EDMS AI Assistant — Document Next Process Tool.

Перевод документа на следующий этап бизнес-процесса.

API:
  POST /api/document/process/next
  Body: {id: UUID, nextId: UUID, employees: [UUID]}

Логика:
  1. GET /{id}/bpmn              -> порядок и названия этапов
  2. GET /{documentId}/process    -> currentId (optimistic lock)
  3. Если сотрудники не указаны — попытка перехода
  4. При ошибке employee.empty — запросить исполнителей у пользователя
  5. POST /process/next с employees
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

from langchain_core.tools import InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.agent.hitl_primitives import ToolAborted, ask_human
from edms_ai_assistant.agent.interrupt_contract import (
    InterruptOption,
    SelectInterrupt,
    SelectResume,
    TextInputInterrupt,
    TextInputResume,
)
from edms_ai_assistant.agent.runnable_utils import (
    get_document_id_from_config,
    get_token_from_config,
)
from edms_ai_assistant.domain.document import DocumentNextProcessRequest
from edms_ai_assistant.utils.regex_utils import UUID_RE
from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from edms_ai_assistant.core.deps import AppDeps

logger = logging.getLogger(__name__)


def _is_uuid(value: str | UUID | None) -> bool:
    return bool(UUID_RE.match(str(value).strip()))


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


# ─── nextId resolution ────────────────────────────────────────────────────────


def _resolve_next_id(
    process_data: dict[str, Any] | None,
    doc_data: dict[str, Any] | None,
) -> str | None:
    """Определяет nextId для POST /process/next."""
    if process_data:
        current_id = process_data.get("currentId")
        if current_id and _is_uuid(str(current_id)):
            cid = str(current_id)
            logger.info("nextId from DocumentProcessDto.currentId: %s", cid[:8])
            return cid

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
                    return item_id

        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("started") and item.get("completed"):
                item_id = str(item.get("id", "")).strip()
                if item_id and _is_uuid(item_id):
                    return item_id

        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", "")).strip()
            if item_id and _is_uuid(item_id):
                return item_id

    if doc_data:
        pid = doc_data.get("processId")
        if pid and _is_uuid(str(pid)):
            return str(pid)

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
        if "employee.empty" in error_str:
            result["is_employee_empty"] = True
        if "process.changed" in error_str:
            result["is_process_changed"] = True
        if "403" in error_str or "NO_ACCESS" in error_str:
            result["is_forbidden"] = True
        if "400" in error_str:
            result["is_bad_request"] = True

        if result["is_employee_empty"]:
            parts = re.findall(r"'([^']*)'", error_str)
            for part in parts:
                if re.search(r"[а-яА-Я]", part):
                    result["target_step_name"] = part
                elif part.isupper() and len(part) > 2 and "_" not in part:
                    result["target_step_type"] = part
    except Exception:
        pass

    return result


# ─── Tool Factory ─────────────────────────────────────────────────────────────


def create_doc_next_process_tool(deps: AppDeps) -> StructuredTool:
    """Фабрика для создания инструмента перевода документа на следующий этап.

    Args:
        deps: Контейнер зависимостей приложения.

    Returns:
        Настроенный StructuredTool, готовый к регистрации в агенте.
    """

    async def _find_employee(token: str, last_name: str) -> str | None:
        try:
            emp_dto = await deps.employee_client.find_by_last_name_fts(token, last_name)
            if emp_dto and emp_dto.id:
                return str(emp_dto.id)
        except Exception as exc:
            logger.debug("Employee FTS failed for '%s': %s", last_name, exc)
        return None

    async def _resolve_employees(
        token: str, employees: list[str]
    ) -> tuple[list[str], list[str]]:
        """Разрешает список сотрудников: UUID -> как есть, ФИО -> поиск."""
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

    async def doc_next_process(
        next_step: str | None = None,
        employees: list[str] | None = None,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Переводит документ на следующий этап бизнес-процесса.

        Пользователь выбирает этап по названию или номеру.
        Если next_step не указан — показывает список этапов.

        Многие этапы (Регистрация, Рассмотрение, Исполнение и др.)
        требуют указания исполнителей. Если исполнители не указаны,
        инструмент запросит их у пользователя.

        Примеры: next_step="1", next_step="Регистрация", next_step="REVIEW"

        ВАЖНО: Токен авторизации и ID документа передаются системой АВТОМАТИЧЕСКИ.
            Тебе НЕ НУЖНО запрашивать их у пользователя.

        Args:
            next_step: Название или номер этапа.
            employees: Список ФИО или UUID исполнителей.
            config: Конфиг Runnable (инжектируется автоматически).

        Returns:
            Dict со статусом и результатом.
        """
        try:
            document_id = get_document_id_from_config(config)
            token = get_token_from_config(config)
        except Exception as e:
            logger.error("Failed to get token from config: %s", e)
            return {
                "status": "error",
                "message": f"Ошибка авторизации: токен не найден. {e}",
            }

        logger.info(
            "doc_next_process: doc=%s next_step=%s employees=%s",
            document_id[:8] if document_id else "N/A",
            next_step,
            employees if employees else "none",
        )

        try:
            try:
                # ── Шаг 1: Получить BPMN, документ и процесс параллельно ─────
                # Используем типизированные клиенты
                bpmn_activity, doc, process = await asyncio.gather(
                    deps.document_client.get_process_activity(token, document_id),
                    deps.document_client.get_document_metadata(token, document_id),
                    deps.document_process_client.get_process(token, UUID(document_id)),
                    return_exceptions=True,
                )

                # Обработка ошибок сбора данных
                if isinstance(bpmn_activity, Exception):
                    logger.error(f"Failed to fetch BPMN activity: {bpmn_activity}")
                    bpmn_activity = None
                if isinstance(doc, Exception):
                    logger.error(f"Failed to fetch document: {doc}")
                    doc = None
                if isinstance(process, Exception):
                    logger.error(f"Failed to fetch process: {process}")
                    process = None

                if not bpmn_activity:
                    return {
                        "status": "error",
                        "message": (
                            "Не удалось получить информацию о процессе документа. "
                            "Возможно, документ ещё не запущен в работу."
                        ),
                    }

                # ── Шаг 2: Парсим BPMN ────────────────────────────────────────
                # Конвертируем DTO в dict для существующей логики парсинга (или адаптируем логику)
                bpmn_data_dict = (
                    bpmn_activity.model_dump(by_alias=True) if bpmn_activity else {}
                )
                bpmn_steps = _parse_bpmn_parsed(bpmn_data_dict)

                logger.info(
                    "BPMN steps: %s",
                    [(s.name, s.process_type, s.is_current) for s in bpmn_steps],
                )

                # ── Шаг 3: Логируем данные процесса ───────────────────────────
                if process:
                    logger.info(
                        "Process: id=%s currentId=%s nextId=%s items_count=%d",
                        str(process.id)[:8] if process.id else None,
                        str(process.current_id)[:8] if process.current_id else None,
                        str(process.next_id)[:8] if process.next_id else None,
                        len(process.items) if process.items else 0,
                    )
                else:
                    logger.warning(
                        "Process data is missing for document %s",
                        document_id[:8],
                    )

                # ── Шаг 4: Определяем текущий и следующие этапы ───────────────
                current_step = next(
                    (s for s in bpmn_steps if s.is_current),
                    None,
                )
                next_available = _get_next_available(bpmn_steps)

                # ── Шаг 5: Показать список (если шаг не выбран) ──────────────
                if not next_step:
                    if not bpmn_steps:
                        return {
                            "status": "error",
                            "message": "Этапы процесса не найдены.",
                        }

                    available_for_selection = next_available or [
                        s for s in bpmn_steps if not s.is_completed
                    ]
                    if not available_for_selection:
                        return {
                            "status": "error",
                            "message": "Нет доступных этапов для перехода.",
                        }

                    current_info = (
                        f"Сейчас документ на этапе: «{current_step.name}»."
                        if current_step
                        else ""
                    )
                    options = [
                        InterruptOption(
                            id=str(i),
                            label=s.name,
                            description=(
                                s.action_label
                                if s.process_type and s.action_label != s.name
                                else None
                            ),
                        )
                        for i, s in enumerate(available_for_selection)
                    ]
                    prompt = "Выберите этап для перехода."
                    if current_info:
                        prompt += f" {current_info}"
                    resume = ask_human(SelectInterrupt(prompt=prompt, options=options))
                    if not isinstance(resume, SelectResume):
                        raise ToolAborted("Этап не выбран")
                    try:
                        selected_idx = int(resume.selected_id)
                        next_step = available_for_selection[selected_idx].name
                    except (ValueError, IndexError):
                        return {
                            "status": "error",
                            "message": "Не удалось определить выбранный этап.",
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
                # Используем типизированные данные для разрешения ID
                process_dict = process.model_dump(by_alias=True) if process else None
                doc_dict = doc.model_dump(by_alias=True) if doc else None
                next_id = _resolve_next_id(process_dict, doc_dict)

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
                request = DocumentNextProcessRequest(
                    id=UUID(document_id),
                    next_id=UUID(next_id),
                    employees=(
                        [UUID(e) for e in resolved_employees]
                        if resolved_employees
                        else None
                    ),
                )

                logger.info(
                    "POST /process/next: doc=%s nextId=%s target='%s' type=%s employees=%d",
                    document_id[:8],
                    next_id[:8],
                    selected.name,
                    selected.process_type,
                    len(resolved_employees) if resolved_employees else 0,
                )

                try:
                    await deps.document_client.next_process(token, request)
                except Exception as post_exc:
                    post_error = _parse_api_error(post_exc)

                    # ── Ошибка: нужны исполнители ──────────────────────────
                    if post_error["is_employee_empty"]:
                        step_label = post_error["target_step_name"] or selected.name

                        resume = ask_human(
                            TextInputInterrupt(
                                prompt=(
                                    f"Для перехода на «{step_label}» укажите ФИО исполнителя:"
                                ),
                                placeholder="Например: Иванов И.И.",
                            )
                        )
                        if not isinstance(resume, TextInputResume):
                            raise ToolAborted("Исполнитель не указан") from None

                        employee_name = resume.value.strip()
                        found = await _find_employee(token, employee_name)
                        if not found:
                            return {
                                "status": "error",
                                "message": (
                                    f"Сотрудник «{employee_name}» не найден. "
                                    "Уточните ФИО и попробуйте снова."
                                ),
                            }

                        retry_request = DocumentNextProcessRequest(
                            id=UUID(document_id),
                            next_id=UUID(next_id),
                            employees=[UUID(found)],
                        )
                        await deps.document_client.next_process(token, retry_request)
                        return {
                            "status": "success",
                            "message": (
                                f"✅ Документ переведён на этап «{selected.name}». "
                                f"Назначен исполнитель: {employee_name}."
                            ),
                            "requires_reload": True,
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

            except ToolAborted:
                raise
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
                        "message": (
                            "Процесс документа был изменён. " "Попробуйте ещё раз."
                        ),
                    }
                if error_info["is_forbidden"]:
                    return {
                        "status": "error",
                        "message": "У вас нет прав для перехода на следующий этап.",
                    }

                return {
                    "status": "error",
                    "message": f"❌ Произошла неожиданная ошибка при переходе: {exc!s}",
                }
        except ToolAborted:
            raise
        except Exception as exc:
            logger.critical("doc_next_process fatal error: %s", exc, exc_info=True)
            return {
                "status": "error",
                "message": f"❌ Критическая ошибка в инструменте перевода процесса: {exc!s}",
            }

    return StructuredTool.from_function(
        coroutine=doc_next_process,
        name="doc_next_process",
        description="Переводит документ на следующий этап бизнес-процесса.",
        args_schema=DocNextProcessInput,
    )

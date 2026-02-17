# edms_ai_assistant/services/task_service.py
"""
Task Service with Disambiguation Support.
"""
import logging
from typing import List, Optional, Set, Tuple, Dict, Any
from uuid import UUID
from datetime import datetime, timezone, timedelta

from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.clients.task_client import TaskClient
from edms_ai_assistant.models.task_models import (
    CreateTaskRequest,
    CreateTaskRequestExecutor,
    TaskType,
    TaskCreationResult,
)

logger = logging.getLogger(__name__)


class TaskService:
    """
    Service for managing document tasks (поручения) with Disambiguation.
    """

    def __init__(self):
        self.employee_client = EmployeeClient()
        self.task_client = TaskClient()

    async def __aenter__(self):
        await self.employee_client.__aenter__()
        await self.task_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.employee_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.task_client.__aexit__(exc_type, exc_val, exc_tb)

    async def collect_executors(
        self,
        token: str,
        last_names: List[str],
        responsible_last_name: Optional[str] = None,
    ) -> Tuple[
        Optional[List[CreateTaskRequestExecutor]], List[str], List[Dict[str, Any]]
    ]:
        """
        Собирает исполнителей с проверкой на неоднозначность.

        Returns:
            Tuple из:
            - executors: Список исполнителей (или None если ambiguous)
            - not_found: Список не найденных фамилий
            - ambiguous_results: Список неоднозначных совпадений
        """
        found_employees = []
        not_found = []
        ambiguous_results = []  # ← ДОБАВЛЕНО!

        for last_name in last_names:
            # Пробуем FTS поиск
            employee = await self.employee_client.find_by_last_name_fts(
                token, last_name
            )

            # Если FTS не нашел → полный поиск
            if not employee:
                employees = await self.employee_client.search_employees(
                    token, {"lastName": last_name, "includes": ["POST", "DEPARTMENT"]}
                )

                if not employees:
                    not_found.append(last_name)
                    logger.warning(f"[TASK-SERVICE] Employee not found: {last_name}")
                    continue

                # ═══════════════════════════════════════════════════════
                # КРИТИЧЕСКАЯ ПРОВЕРКА
                # ═══════════════════════════════════════════════════════
                if len(employees) > 1:
                    logger.info(
                        f"[TASK-SERVICE] Found {len(employees)} employees with last name '{last_name}' - requires disambiguation"
                    )

                    # ФОРМИРУЕМ СПИСОК ДЛЯ ВЫБОРА
                    ambiguous_results.append(
                        {
                            "search_query": last_name,
                            "matches": [
                                {
                                    "id": str(emp["id"]),
                                    "full_name": (
                                        f"{emp.get('lastName', '')} "
                                        f"{emp.get('firstName', '')} "
                                        f"{emp.get('middleName', '') or ''}"
                                    ).strip(),
                                    "post": (
                                        emp.get("post", {}).get(
                                            "postName", "Не указана"
                                        )
                                        if isinstance(emp.get("post"), dict)
                                        else "Не указана"
                                    ),
                                    "department": (
                                        emp.get("department", {}).get(
                                            "name", "Не указан"
                                        )
                                        if isinstance(emp.get("department"), dict)
                                        else "Не указан"
                                    ),
                                }
                                for emp in employees
                            ],
                        }
                    )
                    continue  # ← НЕ добавляем в found_employees!

                # Только одно совпадение → OK
                employee = employees[0]
                logger.debug(
                    f"[TASK-SERVICE] Found single employee: {last_name} -> {employee.get('id')}"
                )

            found_employees.append(employee)

        # ═══════════════════════════════════════════════════════
        # ЕСЛИ ЕСТЬ НЕОДНОЗНАЧНОСТИ → ВОЗВРАЩАЕМ ДЛЯ ВЫБОРА!
        # ═══════════════════════════════════════════════════════
        if ambiguous_results:
            return None, not_found, ambiguous_results

        # Если никто не найден
        if not found_employees:
            return None, not_found, []

        # ═══════════════════════════════════════════════════════
        # ВСЕ НАЙДЕНЫ ОДНОЗНАЧНО → ФОРМИРУЕМ EXECUTORS
        # ═══════════════════════════════════════════════════════
        responsible_employee = None
        if responsible_last_name:
            responsible_employee = await self.employee_client.find_by_last_name_fts(
                token, responsible_last_name
            )

            if not responsible_employee:
                logger.warning(
                    f"[TASK-SERVICE] Responsible employee '{responsible_last_name}' not found, "
                    f"will use first executor as responsible"
                )

        executors = []
        seen_ids: Set[UUID] = set()

        if responsible_employee:
            resp_id = (
                UUID(responsible_employee["id"])
                if isinstance(responsible_employee["id"], str)
                else responsible_employee["id"]
            )
            executors.append(
                CreateTaskRequestExecutor(employeeId=resp_id, responsible=True)
            )
            seen_ids.add(resp_id)

        for idx, emp in enumerate(found_employees):
            emp_id = UUID(emp["id"]) if isinstance(emp["id"], str) else emp["id"]

            if emp_id in seen_ids:
                continue

            is_responsible = not responsible_employee and idx == 0

            executors.append(
                CreateTaskRequestExecutor(employeeId=emp_id, responsible=is_responsible)
            )
            seen_ids.add(emp_id)

        return executors, not_found, []

    async def create_task(
        self,
        token: str,
        document_id: str,
        task_text: str,
        executor_last_names: List[str],
        planed_date_end: Optional[datetime] = None,
        responsible_last_name: Optional[str] = None,
        task_type: TaskType = TaskType.GENERAL,
    ) -> TaskCreationResult:
        """
        Создает поручение с проверкой на неоднозначность.

        Returns:
            TaskCreationResult с одним из статусов:
            - "success" - поручение создано
            - "requires_disambiguation" - нужен выбор из списка
            - "error" - ошибка
        """
        logger.info(f"[TASK-SERVICE] Creating task. Executors: {executor_last_names}")

        if not executor_last_names:
            return TaskCreationResult(
                success=False,
                status="error",
                error_message="Необходимо указать хотя бы одного исполнителя.",
            )

        if not task_text or not task_text.strip():
            return TaskCreationResult(
                success=False,
                status="error",
                error_message="Текст поручения не может быть пустым.",
            )

        # ═══════════════════════════════════════════════════════
        # СБОР ИСПОЛНИТЕЛЕЙ С ПРОВЕРКОЙ НЕОДНОЗНАЧНОСТИ
        # ═══════════════════════════════════════════════════════
        executors, not_found, ambiguous_results = await self.collect_executors(
            token, executor_last_names, responsible_last_name
        )

        # НЕОДНОЗНАЧНОСТЬ ОБНАРУЖЕНА!
        if ambiguous_results:
            logger.info(
                f"[TASK-SERVICE] Disambiguation required. "
                f"Ambiguous: {len(ambiguous_results)}, Not found: {len(not_found)}"
            )

            return TaskCreationResult(
                success=False,
                status="requires_disambiguation",
                ambiguous_matches=ambiguous_results,
                not_found_employees=not_found if not_found else [],
            )

        # Никто не найден
        if not executors:
            return TaskCreationResult(
                success=False,
                status="error",
                error_message=f"Не найдены сотрудники: {', '.join(not_found)}",
                not_found_employees=not_found,
            )

        # ═══════════════════════════════════════════════════════
        # ВСЕ НАЙДЕНЫ ОДНОЗНАЧНО → СОЗДАЕМ ПОРУЧЕНИЕ
        # ═══════════════════════════════════════════════════════

        # Устанавливаем дедлайн
        if not planed_date_end:
            planed_date_end = datetime.now(timezone.utc) + timedelta(days=7)
        elif planed_date_end.tzinfo is None:
            planed_date_end = planed_date_end.replace(tzinfo=timezone.utc)

        # Форматируем текст
        task_text_formatted = (
            task_text[0].upper() + task_text[1:]
            if len(task_text) > 1
            else task_text.upper()
        )

        task_request = CreateTaskRequest(
            taskText=task_text_formatted,
            planedDateEnd=planed_date_end,
            type=task_type,
            periodTask=False,
            endless=False,
            executors=executors,
        )

        try:
            success = await self.task_client.create_tasks_batch(
                token, document_id, [task_request]
            )

            if success:
                logger.info(
                    f"[TASK-SERVICE] Task created successfully. Executors: {len(executors)}"
                )
                return TaskCreationResult(
                    success=True,
                    status="success",
                    created_count=1,
                )
            else:
                return TaskCreationResult(
                    success=False,
                    status="error",
                    error_message=(
                        "Не удалось создать поручение. "
                        "Проверьте права доступа или корректность данных."
                    ),
                )

        except Exception as e:
            logger.error(f"[TASK-SERVICE] Task creation failed: {e}", exc_info=True)
            return TaskCreationResult(
                success=False,
                status="error",
                error_message=f"Ошибка создания поручения: {str(e)}",
            )

    async def create_task_by_employee_ids(
        self,
        token: str,
        document_id: str,
        task_text: str,
        employee_ids: List[UUID],
        planed_date_end: Optional[datetime] = None,
        responsible_employee_id: Optional[UUID] = None,
        task_type: TaskType = TaskType.GENERAL,
    ) -> TaskCreationResult:
        """
        Создает поручение для конкретных сотрудников (по UUID).

        Используется ПОСЛЕ выбора пользователем из списка.
        """
        logger.info(
            f"[TASK-SERVICE] Creating task with pre-selected employees: {len(employee_ids)}"
        )

        if not employee_ids:
            return TaskCreationResult(
                success=False,
                status="error",
                error_message="Необходимо указать хотя бы одного исполнителя.",
            )

        # Определяем ответственного
        responsible_id = responsible_employee_id or employee_ids[0]

        # Формируем список исполнителей
        executors = []
        seen_ids: Set[UUID] = set()

        for emp_id in employee_ids:
            if emp_id in seen_ids:
                continue

            executors.append(
                CreateTaskRequestExecutor(
                    employeeId=emp_id, responsible=(emp_id == responsible_id)
                )
            )
            seen_ids.add(emp_id)

        # Устанавливаем дедлайн
        if not planed_date_end:
            planed_date_end = datetime.now(timezone.utc) + timedelta(days=7)
        elif planed_date_end.tzinfo is None:
            planed_date_end = planed_date_end.replace(tzinfo=timezone.utc)

        # Форматируем текст
        task_text_formatted = (
            task_text[0].upper() + task_text[1:]
            if len(task_text) > 1
            else task_text.upper()
        )

        task_request = CreateTaskRequest(
            taskText=task_text_formatted,
            planedDateEnd=planed_date_end,
            type=task_type,
            periodTask=False,
            endless=False,
            executors=executors,
        )

        try:
            success = await self.task_client.create_tasks_batch(
                token, document_id, [task_request]
            )

            if success:
                return TaskCreationResult(
                    success=True,
                    status="success",
                    created_count=1,
                )
            else:
                return TaskCreationResult(
                    success=False,
                    status="error",
                    error_message="Не удалось создать поручение.",
                )

        except Exception as e:
            logger.error(f"[TASK-SERVICE] Task creation failed: {e}", exc_info=True)
            return TaskCreationResult(
                success=False,
                status="error",
                error_message=f"Ошибка создания поручения: {str(e)}",
            )

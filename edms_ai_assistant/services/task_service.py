"""
EDMS AI Assistant - Task Service.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Set, Tuple
from uuid import UUID

from edms_ai_assistant.clients.employee_client import EmployeeClient
from edms_ai_assistant.clients.task_client import TaskClient
from edms_ai_assistant.models.task_models import (
    CreateTaskRequest,
    CreateTaskRequestExecutor,
    TaskCreationResult,
    TaskType,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutorResolutionResult:
    """Immutable результат резолвинга исполнителей."""

    executors: List[CreateTaskRequestExecutor]
    not_found: List[str]


class TaskService:
    """
    Сервисный слой для управления поручениями к документам.

    Responsibilities:
    - Резолвинг исполнителей по фамилиям
    - Назначение ответственного исполнителя
    - Обработка дубликатов
    - Создание поручений через API
    - Валидация и нормализация данных

    Бизнес-правила:
    - Если ответственный не указан → первый исполнитель становится ответственным
    - Если срок не указан → +7 дней от текущей даты
    - Дубликаты исполнителей удаляются автоматически
    - Текст поручения капитализируется
    """

    DEFAULT_DEADLINE_DAYS = 7

    def __init__(self):
        """Инициализация с созданием клиентов для внешних API."""
        self.employee_client = EmployeeClient()
        self.task_client = TaskClient()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.employee_client.__aenter__()
        await self.task_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit с корректным закрытием клиентов."""
        await self.employee_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.task_client.__aexit__(exc_type, exc_val, exc_tb)

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
        Создает поручение к документу с назначением исполнителей.

        Args:
            token: JWT токен авторизации
            document_id: UUID документа
            task_text: Текст поручения
            executor_last_names: Список фамилий исполнителей
            planed_date_end: Срок выполнения (опционально, default: +7 дней)
            responsible_last_name: Фамилия ответственного (опционально)
            task_type: Тип поручения

        Returns:
            TaskCreationResult с информацией об успехе операции
        """
        if not executor_last_names:
            return TaskCreationResult(
                success=False,
                error_message="Необходимо указать хотя бы одного исполнителя.",
            )

        task_text_cleaned = task_text.strip()
        if not task_text_cleaned:
            return TaskCreationResult(
                success=False, error_message="Текст поручения не может быть пустым."
            )

        resolution_result = await self._resolve_executors(
            token=token,
            executor_last_names=executor_last_names,
            responsible_last_name=responsible_last_name,
        )

        if resolution_result.not_found:
            return TaskCreationResult(
                success=False,
                error_message=f"❌ Не найдены сотрудники: {', '.join(resolution_result.not_found)}",
                not_found_employees=resolution_result.not_found,
            )

        if not resolution_result.executors:
            return TaskCreationResult(
                success=False,
                error_message="Не удалось найти ни одного исполнителя.",
            )

        deadline = self._prepare_deadline(planed_date_end)
        normalized_text = self._normalize_task_text(task_text_cleaned)

        task_request = CreateTaskRequest(
            taskText=normalized_text,
            planedDateEnd=deadline,
            type=task_type,
            periodTask=False,
            endless=False,
            executors=resolution_result.executors,
        )

        logger.info(
            "Creating task via API",
            extra={
                "document_id": document_id,
                "executors_count": len(resolution_result.executors),
                "deadline": deadline.isoformat(),
            },
        )

        success = await self.task_client.create_tasks_batch(
            token, document_id, [task_request]
        )

        if success:
            logger.info(
                "Task created successfully",
                extra={"document_id": document_id},
            )
            return TaskCreationResult(success=True, created_count=1)

        logger.error(
            "Failed to create task via API",
            extra={"document_id": document_id},
        )
        return TaskCreationResult(
            success=False,
            error_message=(
                "❌ Не удалось создать поручение. "
                "Проверьте права доступа или корректность данных."
            ),
        )

    async def _resolve_executors(
        self,
        token: str,
        executor_last_names: List[str],
        responsible_last_name: Optional[str],
    ) -> ExecutorResolutionResult:
        """
        Резолвит исполнителей по фамилиям с обработкой дубликатов.

        Strategy:
        1. Поиск всех исполнителей по фамилиям
        2. Поиск ответственного исполнителя (если указан)
        3. Удаление дубликатов
        4. Назначение флага responsible

        Returns:
            ExecutorResolutionResult с найденными исполнителями
        """
        found_employees = []
        not_found = []

        for last_name in executor_last_names:
            employee = await self._find_employee_by_last_name(token, last_name)

            if employee:
                found_employees.append(employee)
            else:
                not_found.append(last_name)
                logger.warning(
                    f"Employee not found: {last_name}",
                    extra={"last_name": last_name},
                )

        if not_found:
            return ExecutorResolutionResult(executors=[], not_found=not_found)

        responsible_employee = None
        if responsible_last_name:
            responsible_employee = await self._find_employee_by_last_name(
                token, responsible_last_name
            )

            if not responsible_employee:
                logger.warning(
                    f"Responsible employee '{responsible_last_name}' not found, "
                    f"will use first executor as responsible"
                )

        executors = self._build_executor_list(
            found_employees=found_employees,
            responsible_employee=responsible_employee,
        )

        logger.info(
            "Executors resolved",
            extra={
                "total_count": len(executors),
                "has_responsible": any(e.responsible for e in executors),
            },
        )

        return ExecutorResolutionResult(executors=executors, not_found=[])

    async def _find_employee_by_last_name(
        self, token: str, last_name: str
    ) -> Optional[dict]:
        """
        Поиск сотрудника по фамилии с fallback стратегией.

        Strategy:
        1. FTS поиск (быстрый, точный)
        2. Fallback на обычный поиск
        3. При множественных совпадениях → первый результат + warning

        Returns:
            Dict с данными сотрудника или None
        """
        employee = await self.employee_client.find_by_last_name_fts(token, last_name)

        if employee:
            return employee

        employees = await self.employee_client.search_employees(
            token, {"lastName": last_name}
        )

        if not employees:
            return None

        if len(employees) > 1:
            logger.info(
                f"Multiple employees found for '{last_name}', using first match",
                extra={
                    "last_name": last_name,
                    "matches_count": len(employees),
                    "selected": f"{employees[0].get('firstName', '')} {employees[0].get('lastName', '')}",
                },
            )

        return employees[0]

    @staticmethod
    def _build_executor_list(
        found_employees: List[dict],
        responsible_employee: Optional[dict],
    ) -> List[CreateTaskRequestExecutor]:
        """
        Строит список исполнителей с удалением дубликатов и назначением ответственного.

        Логика:
        1. Если есть responsible_employee → он первый в списке с флагом responsible=True
        2. Остальные исполнители добавляются с флагом responsible=False
        3. Если responsible_employee не найден → первый исполнитель становится ответственным
        4. Дубликаты удаляются через Set[UUID]

        Args:
            found_employees: Найденные сотрудники-исполнители
            responsible_employee: Назначенный ответственный (опционально)

        Returns:
            Список CreateTaskRequestExecutor без дубликатов
        """
        executors: List[CreateTaskRequestExecutor] = []
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

        return executors

    @classmethod
    def _prepare_deadline(cls, planed_date_end: Optional[datetime]) -> datetime:
        """
        Подготавливает deadline с обработкой timezone и default значения.

        Rules:
        - Если не указан → +7 дней от текущей даты
        - Если указан без timezone → добавляется UTC
        - Всегда возвращает datetime с timezone

        Args:
            planed_date_end: Опциональная дата окончания

        Returns:
            datetime с UTC timezone
        """
        if not planed_date_end:
            return datetime.now(timezone.utc) + timedelta(
                days=cls.DEFAULT_DEADLINE_DAYS
            )

        if planed_date_end.tzinfo is None:
            return planed_date_end.replace(tzinfo=timezone.utc)

        return planed_date_end

    @staticmethod
    def _normalize_task_text(task_text: str) -> str:
        """
        Нормализует текст поручения.

        Rules:
        - Капитализация первой буквы
        - Удаление лишних пробелов

        Args:
            task_text: Исходный текст

        Returns:
            Нормализованный текст
        """
        cleaned = task_text.strip()

        if len(cleaned) > 1:
            return cleaned[0].upper() + cleaned[1:]

        return cleaned.upper()

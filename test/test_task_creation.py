# tests/test_task_creation.py
import pytest
from uuid import uuid4
from datetime import datetime, timezone, timedelta
from edms_ai_assistant.services.task_service import TaskService
from edms_ai_assistant.models.task_models import CreateTaskRequest, CreateTaskRequestExecutor, TaskType


@pytest.mark.asyncio
async def test_create_task_single_executor():
    """
    Test: Создание поручения с одним исполнителем.
    Ожидание: Исполнитель становится ответственным автоматически.
    """
    async with TaskService() as service:
        result = await service.create_task(
            token="test_token",
            document_id=str(uuid4()),
            task_text="Подготовить отчет",
            executor_last_names=["Уникальный"],  # Only one executor
            planed_date_end=datetime.now(timezone.utc) + timedelta(days=7),
        )

        assert result.success is True
        assert result.created_count == 1
        assert len(result.not_found_employees) == 0


@pytest.mark.asyncio
async def test_create_task_multiple_executors_first_responsible():
    """
    Test: Несколько исполнителей, ответственный не указан явно.
    Ожидание: Первый исполнитель становится ответственным.
    """
    async with TaskService() as service:
        # Mock: найдено 3 сотрудника
        executors, not_found = await service.collect_executors(
            token="test_token",
            last_names=["Иванов", "Петров", "Сидоров"],
            responsible_last_name=None,  # Not specified
        )

        # Verify: First executor is responsible
        assert len(executors) == 3
        assert executors[0].responsible is True
        assert executors[1].responsible is False
        assert executors[2].responsible is False


@pytest.mark.asyncio
async def test_create_task_explicit_responsible():
    """
    Test: Несколько исполнителей, ответственный указан явно.
    Ожидание: Указанный сотрудник становится ответственным.
    """
    async with TaskService() as service:
        executors, not_found = await service.collect_executors(
            token="test_token",
            last_names=["Иванов", "Петров", "Сидоров"],
            responsible_last_name="Петров",  # Explicit responsible
        )

        # Verify: Петров is responsible
        responsible_ids = [e.employeeId for e in executors if e.responsible]
        assert len(responsible_ids) == 1


@pytest.mark.asyncio
async def test_create_task_duplicate_removal():
    """
    Test: Дубликаты в списке исполнителей.
    Ожидание: Дубликаты автоматически удаляются.
    """
    async with TaskService() as service:
        executors, not_found = await service.collect_executors(
            token="test_token",
            last_names=["Иванов", "Иванов", "Петров"],  # Duplicate "Иванов"
        )

        # Verify: Only 2 unique executors
        unique_ids = {e.employeeId for e in executors}
        assert len(unique_ids) == 2


@pytest.mark.asyncio
async def test_create_task_employee_not_found():
    """
    Test: Один из исполнителей не найден в системе.
    Ожидание: Ошибка с указанием не найденного сотрудника.
    """
    async with TaskService() as service:
        result = await service.create_task(
            token="test_token",
            document_id=str(uuid4()),
            task_text="Анализ данных",
            executor_last_names=["Существующий", "НесуществующаяФамилия"],
        )

        assert result.success is False
        assert "НесуществующаяФамилия" in result.not_found_employees


def test_task_request_datetime_serialization():
    """
    Test: Правильная сериализация datetime в ISO 8601 с timezone.
    """
    from edms_ai_assistant.models.task_models import CreateTaskRequest

    deadline = datetime(2026, 2, 15, 23, 59, 59, tzinfo=timezone.utc)

    request = CreateTaskRequest(
        taskText="Тестовое поручение",
        planedDateEnd=deadline,
        executors=[
            CreateTaskRequestExecutor(employeeId=uuid4(), responsible=True)
        ],
    )

    payload = request.model_dump(mode="json")

    # Verify: ISO 8601 format with timezone
    assert payload["planedDateEnd"] == "2026-02-15T23:59:59Z"
    assert "taskText" in payload
    assert isinstance(payload["executors"], list)


def test_task_request_naive_datetime_conversion():
    """
    Test: Naive datetime (без timezone) автоматически конвертируется в UTC.
    """
    from edms_ai_assistant.models.task_models import CreateTaskRequest

    # Naive datetime (no timezone)
    naive_deadline = datetime(2026, 2, 15, 23, 59, 59)

    request = CreateTaskRequest(
        taskText="Тестовое поручение",
        planedDateEnd=naive_deadline,
        executors=[
            CreateTaskRequestExecutor(employeeId=uuid4(), responsible=True)
        ],
    )

    payload = request.model_dump(mode="json")

    # Verify: Adds 'Z' suffix (UTC)
    assert payload["planedDateEnd"].endswith("Z")


def test_task_request_validation_empty_executors():
    """
    Test: Валидация - нельзя создать поручение без исполнителей.
    """
    from edms_ai_assistant.models.task_models import CreateTaskRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        CreateTaskRequest(
            taskText="Тестовое поручение",
            planedDateEnd=datetime.now(timezone.utc) + timedelta(days=7),
            executors=[],  # Empty list - INVALID
        )

    # Verify: Error mentions "executors"
    assert "executors" in str(exc_info.value)


def test_task_request_validation_empty_text():
    """
    Test: Валидация - текст поручения не может быть пустым.
    """
    from edms_ai_assistant.models.task_models import CreateTaskRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CreateTaskRequest(
            taskText="",  # Empty text - INVALID
            planedDateEnd=datetime.now(timezone.utc) + timedelta(days=7),
            executors=[
                CreateTaskRequestExecutor(employeeId=uuid4(), responsible=True)
            ],
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
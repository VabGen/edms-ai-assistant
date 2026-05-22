import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from edms_ai_assistant.services.resolution_service import ResolutionService
from edms_ai_assistant.domain.employee import EmployeeDto, DepartmentDto

@pytest.mark.asyncio
async def test_resolve_employees():
    mock_emp_client = MagicMock()
    mock_dep_client = MagicMock()
    mock_group_client = MagicMock()

    service = ResolutionService(
        employee_client=mock_emp_client,
        department_client=mock_dep_client,
        group_client=mock_group_client
    )

    emp_id = uuid4()
    # Mock search result
    mock_emp_client.search_employees_post = AsyncMock(return_value=[
        EmployeeDto(id=emp_id, first_name="Test", last_name="User")
    ])

    found_ids, not_found, ambiguous = await service.resolve_employees("token", ["User"])

    assert len(found_ids) == 1
    assert emp_id in found_ids
    assert len(not_found) == 0
    assert len(ambiguous) == 0

@pytest.mark.asyncio
async def test_resolve_departments():
    mock_emp_client = MagicMock()
    mock_dep_client = MagicMock()
    mock_group_client = MagicMock()

    service = ResolutionService(
        employee_client=mock_emp_client,
        department_client=mock_dep_client,
        group_client=mock_group_client
    )

    dep_id = uuid4()
    mock_dep_client.find_by_name = AsyncMock(return_value=DepartmentDto(id=dep_id, name="Test Dep"))
    mock_dep_client.get_employees_by_department_id = AsyncMock(return_value=[
        EmployeeDto(id=uuid4(), first_name="Dep", last_name="Member")
    ])

    found_ids, _not_found, total = await service.resolve_departments("token", ["Test Dep"])

    assert len(found_ids) == 1
    assert total == 1
    mock_dep_client.find_by_name.assert_called_once()

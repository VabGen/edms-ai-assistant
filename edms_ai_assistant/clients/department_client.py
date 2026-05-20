# edms_ai_assistant/clients/department_client.py
import logging
from uuid import UUID

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.employee import DepartmentDto, EmployeeDto

logger = logging.getLogger(__name__)


class DepartmentClient:
    """HTTP client for EDMS Department API.

    Использует композицию: делегирует HTTP-логику базовому клиенту.
    Возвращает строгие Pydantic DTO.
    """

    def __init__(self, base_client: EdmsBaseClient):
        self._client = base_client

    async def find_by_name(
            self, token: str, department_name: str
    ) -> DepartmentDto | None:
        """Ищет отдел по названию через FTS.

        Args:
            token: JWT authorization token.
            department_name: Название отдела для поиска.

        Returns:
            DepartmentDto или None, если не найден.
        """
        try:
            result = await self._client._make_request(
                "GET",
                "api/department/fts-name",
                token=token,
                params={"fts": department_name}
            )
            if isinstance(result, dict) and result:
                dept = DepartmentDto.model_validate(result)
                logger.info(
                    "Department found: %s (id=%s)",
                    dept.name,
                    str(dept.id)[:8] if dept.id else "Unknown"
                )
                return dept
        except EdmsNotFoundError:
            logger.info("Department not found: '%s'", department_name)

        return None

    async def get_employees_by_department_id(
            self, token: str, department_id: UUID
    ) -> list[EmployeeDto]:
        """Получает список сотрудников отдела по его UUID.

        Args:
            token: JWT authorization token.
            department_id: UUID отдела.

        Returns:
            Список EmployeeDto или пустой список, если отдел не найден.
        """
        try:
            result = await self._client._make_request(
                "GET",
                f"api/department/{department_id}/employees/all",
                token=token
            )
            if isinstance(result, list):
                employees = [EmployeeDto.model_validate(e) for e in result]
                logger.info(
                    "Found %d employees in department %s",
                    len(employees),
                    str(department_id)[:8]
                )
                return employees
        except EdmsNotFoundError:
            pass

        return []
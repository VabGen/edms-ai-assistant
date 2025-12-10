# edms_ai_assistant/tools/employee.py

from langchain_core.tools import tool
import json
import logging
from uuid import UUID
from edms_ai_assistant.infrastructure.api_clients.employee_client import EmployeeClient

logger = logging.getLogger(__name__)


async def _get_employee_client(service_token: str) -> EmployeeClient:
    return EmployeeClient(service_token=service_token)


@tool
async def find_responsible_tool(query: str, service_token: str) -> str:
    """
    Инструмент для поиска сотрудников по запросу (ФИО, должность).
    Возвращает JSON-список найденных сотрудников.
    """
    logger.info(f"Вызов find_responsible_tool с query: {query}")

    async with _get_employee_client(service_token) as client:
        try:
            result = await client.search_employees(query)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Ошибка в find_responsible_tool: {e}")
            return json.dumps({"error": f"Сбой поиска сотрудников: {str(e)}"}, ensure_ascii=False)


@tool
async def get_employee_by_id_tool(employee_id: str, service_token: str) -> str:
    """
    Инструмент для получения сотрудника по ID.
    Требует валидный UUID сотрудника.
    """
    try:
        employee_uuid = UUID(employee_id)
    except ValueError:
        return json.dumps({"error": f"Неверный формат ID сотрудника: '{employee_id}'. Ожидается UUID."},
                          ensure_ascii=False)

    logger.info(f"Вызов get_employee_by_id_tool с employee_id: {employee_id}")

    async with _get_employee_client(service_token) as client:
        try:
            result = await client.get_employee(employee_uuid)

            if result is None:
                return json.dumps({"employee": "Сотрудник не найден."}, ensure_ascii=False)

            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Ошибка в get_employee_by_id_tool: {e}")
            return json.dumps({"error": f"Сбой получения сотрудника по ID: {str(e)}"}, ensure_ascii=False)
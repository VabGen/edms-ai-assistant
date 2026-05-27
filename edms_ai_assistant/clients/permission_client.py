# edms_ai_assistant/clients/permission_client.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.core.exceptions import EdmsNotFoundError
from edms_ai_assistant.domain.document import PermissionDto, PermissionRoleDto
from edms_ai_assistant.domain.enums import (
    AcceptanceInventoryStatus,
    DestructionActStatus,
    DocCategory,
    DocumentStatus,
    NomenclatureDepartmentStatus,
    SummaryNomenclatureDepartmentStatus,
    TaskType,
)

if TYPE_CHECKING:
    from edms_ai_assistant.clients.transport import IAsyncTransport
    from edms_ai_assistant.config import EdmsSettings

logger = logging.getLogger(__name__)


class PermissionClient(EdmsBaseClient):
    """Client for EDMS Permission API."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    async def get_permission(
        self, token: str, permission_id: str | UUID
    ) -> PermissionDto:
        """Fetches permission by ID."""
        logger.info(f"Fetching permission {permission_id}")
        return await self._request_dto(
            "GET", f"api/permission/{permission_id}", token, PermissionDto
        )

    async def get_definition(self, token: str, system_name: str) -> dict[str, Any]:
        """Fetches permission rule definition by system name."""
        logger.info(f"Fetching permission definition for {system_name}")
        return await self.make_request(
            "GET", f"api/permission/{system_name}/definition", token
        )

    async def add_role(
        self, token: str, permission_id: str | UUID, role_id: str | UUID
    ) -> None:
        """Adds a role to a permission."""
        logger.info(f"Adding role {role_id} to permission {permission_id}")
        await self.make_request(
            "POST",
            f"api/permission/{permission_id}/role",
            token,
            json_data={"id": str(role_id)},
            is_json_response=False,
        )

    async def add_roles_batch(
        self, token: str, permission_id: str | UUID, role_ids: list[UUID]
    ) -> None:
        """Adds multiple roles to a permission."""
        logger.info(f"Adding roles batch to permission {permission_id}")
        await self.make_request(
            "POST",
            f"api/permission/{permission_id}/role/batch",
            token,
            json_data={"ids": role_ids},
            is_json_response=False,
        )

    async def remove_role(
        self, token: str, permission_id: str | UUID, role_id: str | UUID
    ) -> None:
        """Removes a role from a permission."""
        logger.info(f"Removing role {role_id} from permission {permission_id}")
        await self.make_request(
            "DELETE",
            f"api/permission/{permission_id}/role",
            token,
            json_data={"id": str(role_id)},
            is_json_response=False,
        )

    async def remove_roles_batch(
        self, token: str, permission_id: str | UUID, role_ids: list[UUID]
    ) -> None:
        """Removes multiple roles from a permission."""
        logger.info(f"Removing roles batch from permission {permission_id}")
        await self.make_request(
            "DELETE",
            f"api/permission/{permission_id}/role/batch",
            token,
            json_data={"ids": role_ids},
            is_json_response=False,
        )

    async def copy_permission(
        self, token: str, permission_id: str | UUID, copy_request: dict[str, Any]
    ) -> list[PermissionDto]:
        """Copies permission settings."""
        logger.info(f"Copying permission settings from {permission_id}")
        return await self._request_list(
            "POST",
            f"api/permission/{permission_id}/copy",
            token,
            PermissionDto,
            json_data=copy_request,
        )

    async def reload_permissions(self, token: str) -> None:
        """Reloads all permissions."""
        logger.info("Reloading permissions")
        await self.make_request(
            "POST", "api/permission/reload", token, is_json_response=False
        )

    async def delete_permission(self, token: str, permission_id: str | UUID) -> None:
        """Deletes a permission entry."""
        logger.info(f"Deleting permission {permission_id}")
        await self.make_request(
            "DELETE",
            f"api/permission/{permission_id}",
            token,
            json_data={"id": str(permission_id)},
            is_json_response=False,
        )

    async def update_permission(
        self, token: str, permission: PermissionDto
    ) -> PermissionDto:
        """Updates a permission entry."""
        logger.info(f"Updating permission {permission.id}")
        return await self._request_dto(
            "PUT",
            "api/permission",
            token,
            PermissionDto,
            json_data=permission.model_dump(by_alias=True),
        )

    async def get_by_system_name(
        self,
        token: str,
        system_name: str,
        status: DocumentStatus | None = None,
        category: DocCategory | None = None,
        task_type: TaskType | None = None,
        name: str | None = None,
        nomenclature_department_status: NomenclatureDepartmentStatus | None = None,
        summary_nomenclature_department_status: (
            SummaryNomenclatureDepartmentStatus | None
        ) = None,
        destruction_act_status: DestructionActStatus | None = None,
        acceptance_inventory_status: AcceptanceInventoryStatus | None = None,
    ) -> list[PermissionDto]:
        """Fetches permissions by system name and other filters."""
        logger.info(f"Fetching permissions by system name: {system_name}")
        params = {"systemName": system_name}
        if status:
            params["status"] = status
        if category:
            params["category"] = category
        if task_type:
            params["taskType"] = task_type
        if name:
            params["name"] = name
        if nomenclature_department_status:
            params["nomenclatureDepartmentStatus"] = nomenclature_department_status
        if summary_nomenclature_department_status:
            params["summaryNomenclatureDepartmentStatus"] = (
                summary_nomenclature_department_status
            )
        if destruction_act_status:
            params["destructionActStatus"] = destruction_act_status
        if acceptance_inventory_status:
            params["acceptanceInventoryStatus"] = acceptance_inventory_status

        return await self._request_list(
            "GET", "api/permission/system-name", token, PermissionDto, params=params
        )

    async def create_by_system_name(
        self,
        token: str,
        system_name: str,
        status: DocumentStatus | None = None,
        category: DocCategory | None = None,
        task_type: TaskType | None = None,
        name: str | None = None,
        nomenclature_department_status: NomenclatureDepartmentStatus | None = None,
        summary_nomenclature_department_status: (
            SummaryNomenclatureDepartmentStatus | None
        ) = None,
        destruction_act_status: DestructionActStatus | None = None,
        acceptance_inventory_status: AcceptanceInventoryStatus | None = None,
    ) -> PermissionDto:
        """Creates a permission entry by system name."""
        logger.info(f"Creating permission by system name: {system_name}")
        params = {"systemName": system_name}
        if status:
            params["status"] = status
        if category:
            params["category"] = category
        if task_type:
            params["taskType"] = task_type
        if name:
            params["name"] = name
        if nomenclature_department_status:
            params["nomenclatureDepartmentStatus"] = nomenclature_department_status
        if summary_nomenclature_department_status:
            params["summaryNomenclatureDepartmentStatus"] = (
                summary_nomenclature_department_status
            )
        if destruction_act_status:
            params["destructionActStatus"] = destruction_act_status
        if acceptance_inventory_status:
            params["acceptanceInventoryStatus"] = acceptance_inventory_status

        return await self._request_dto(
            "POST", "api/permission/system-name", token, PermissionDto, params=params
        )

    async def get_permission_roles(
        self, token: str, permission_id: str | UUID
    ) -> list[PermissionRoleDto]:
        """Fetches roles for a permission."""
        logger.info(f"Fetching roles for permission {permission_id}")
        return await self._request_list(
            "GET", f"api/permission/{permission_id}/role", token, PermissionRoleDto
        )

    async def export_permissions(self, token: str) -> bytes:
        """Exports current permission settings."""
        logger.info("Exporting permissions")
        return await self.make_request("GET", "api/permission/export", token)

    async def import_permissions(
        self, token: str, import_definition: dict[str, Any]
    ) -> None:
        """Imports permission settings."""
        logger.info("Importing permissions")
        await self.make_request(
            "POST",
            "api/permission/import",
            token,
            json_data=import_definition,
            is_json_response=False,
        )

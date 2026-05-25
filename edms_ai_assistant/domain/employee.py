from __future__ import annotations

from typing import Annotated, TYPE_CHECKING
from pydantic import Field
from uuid import UUID
from datetime import datetime
from edms_ai_assistant.domain.enums import GroupType, RoleObjectType

from edms_ai_assistant.domain.base import EdmsBaseDto

if TYPE_CHECKING:
    from edms_ai_assistant.domain.enums import GroupType, RoleObjectType
    from uuid import UUID
    from datetime import datetime


class PostDto(EdmsBaseDto):
    id: int | None = Field(None, description="Идентификатор должности")
    post_name: str | None = Field(None, description="Наименование должности")
    post_code: str | None = Field(None, description="Код должности")
    create_date: datetime | None = None


class MiniUserInfoDto(EdmsBaseDto):
    id: UUID | None = Field(None, description="ИД сотрудника")
    first_name: str | None = Field(None, description="Имя сотрудника")
    last_name: str | None = Field(None, description="Фамилия сотрудника")
    middle_name: str | None = Field(None, description="Отчество сотрудника")


class UserInfoDto(EdmsBaseDto):
    first_name: Annotated[str | None, Field(description="Имя сотрудника")] = None
    last_name: Annotated[str | None, Field(description="Фамилия сотрудника")] = None
    middle_name: Annotated[str | None, Field(description="Отчество сотрудника")] = None
    author_post: Annotated[str | None, Field(description="Наименование должность сотрудника")] = None
    author_department_name: Annotated[str | None, Field(description="Наименование департамента сотрудника")] = None
    author_department: Annotated[DepartmentDto | None, Field(description="Департамент сотрудника")] = None
    author_department_id: Annotated[UUID | None, Field(description="Идентификатор департамента сотрудника")] = None
    author_department_org_id: str | None = None
    employee: Annotated[EmployeeDto | None, Field(description="Сотрудник")] = None
    employee_id: Annotated[UUID | None, Field(description="Идентификатор сотрудник")] = None
    employee_org_id: str | None = None


class DeputyLeaderDepartmentDto(EdmsBaseDto):
    id: UUID | None = None


class DepartmentEmployeeNomenclatureDto(EdmsBaseDto):
    id: UUID | None = None


class DepartmentDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = None
    number: str | None = None
    parent_department_id: UUID | None = None
    parent_department: DepartmentDto | None = None
    rank: str | None = None
    department_code: str | None = None
    phone: str | None = None
    email: str | None = None
    address: str | None = None
    room: str | None = None
    leader_id: UUID | None = None
    leader: EmployeeDto | None = None
    employees: list[EmployeeDto] | None = None
    deputy_leaders: list[DeputyLeaderDepartmentDto] | None = None
    employee_nomenclatures: list[DepartmentEmployeeNomenclatureDto] | None = None
    create_date: datetime | None = None
    order: int | None = None
    current_user_leader: bool | None = None


class EmployeeDto(MiniUserInfoDto):
    post: PostDto | None = None
    department: DepartmentDto | None = None
    email: str | None = None
    phone: str | None = None
    active: bool | None = None
    create_date: datetime | None = None


class CurrentUserDto(EdmsBaseDto):
    id: UUID | None = None
    username: str | None = None
    employee: EmployeeDto | None = None


class RoleDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = Field(None, max_length=255, min_length=0)
    system_name: str | None = None
    system: bool | None = None
    type: RoleObjectType | None = None
    system_manage: bool | None = None
    create_date: datetime | None = None


class AccessGriefDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = None
    short_name: str | None = None
    active: bool | None = None
    create_date: datetime | None = None


class EmployeeAccessGriefDto(AccessGriefDto):
    pass


class GroupDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = Field(None, max_length=255, min_length=1)
    type: GroupType
    mixed: bool
    create_date: datetime | None = None


EmployeeDto.model_rebuild()
DepartmentDto.model_rebuild()

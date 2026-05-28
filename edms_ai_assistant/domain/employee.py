from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Generic, TypeVar
from uuid import UUID

from pydantic import Field

from edms_ai_assistant.domain.base import EdmsBaseDto, SliceDto
from edms_ai_assistant.domain.enums import (
    BlockedField,
    EmployeeCreateType,
    EmployeeIoType,
    GroupType,
    RoleType,
)
from edms_ai_assistant.domain.reference import OrgDto

if TYPE_CHECKING:
    from edms_ai_assistant.domain.reference import OrgDto

T = TypeVar("T")


class SliceDto(EdmsBaseDto, Generic[T]):
    """DTO for Slice (pagination)."""

    number: int = Field(0, description="Номер страницы")
    size: int = Field(20, description="Размер страницы")
    number_of_elements: int = Field(0, description="Кол-во элементов в странице")
    has_next: bool = Field(False, description="Есть ли следующая страница")
    content: list[T] = Field(default_factory=list)


class PostDto(EdmsBaseDto):
    id: int | None = Field(None, description="Идентификатор должности")
    post_name: str | None = Field(None, description="Наименование должности")
    post_code: str | None = Field(None, description="Код должности")
    create_date: datetime | None = None


class PostRequest(EdmsBaseDto):
    id: int | None = None
    post_name: str | None = None
    post_code: str | None = None


class RoleDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = Field(
        None, max_length=255, description="Value length should be 255 signs"
    )
    system_name: str | None = None
    system: bool | None = None
    type: RoleType | None = None
    system_manage: bool | None = None
    create_date: datetime | None = None


class AccessGriefDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = None
    short_name: str | None = None
    active: bool | None = None
    create_date: datetime | None = None


class AccessGriefRequest(EdmsBaseDto):
    id: UUID | None = None
    name: str = Field(..., max_length=255)
    short_name: str = Field(..., max_length=255)
    organization_id: str | None = None
    active: bool = True
    employee_add: set[UUID] | None = None
    employee_delete: list[UUID] | None = None


class AccessGriefFilter(EdmsBaseDto):
    search: str | None = Field(
        None, description="Строка поиска по названию или краткому названию"
    )
    active: bool | None = Field(None, description="Признак активности")


class EmployeeAccessGriefDto(EdmsBaseDto):
    id: UUID | None = None
    employee_id: UUID | None = None
    access_grief_id: UUID | None = None
    organization_id: str | None = None
    employee: EmployeeDto | None = None
    access_grief: AccessGriefDto | None = None


class MiniUserInfoDto(EdmsBaseDto):
    id: UUID | None = Field(None, description="ИД сотрудника")
    first_name: str | None = Field(None, description="Имя сотрудника")
    last_name: str | None = Field(None, description="Фамилия сотрудника")
    middle_name: str | None = Field(None, description="Отчество сотрудника")


class EmployeeDto(MiniUserInfoDto):
    """DTO model for Employee."""

    u_id: str | None = Field(None, description="Идентификатор аккаунта сотрудника")
    personal_number: str | None = Field(
        None, description="Персональный номер сотрудника"
    )
    fired: bool | None = Field(None, description="Признак уволен ли сотрудник")
    active: bool | None = Field(None, description="Признак активен ли сотрудник")
    organization_id: str | None = None
    full_post_name: str | None = Field(None, description="ФИО сотрудника")
    post_id: int | None = Field(None, description="Идентификатор должности")
    post: PostDto | None = Field(None, description="Должность сотрудника")
    department_id: UUID | None = Field(
        None, description="Идентификатор департамента в котором работает сотрудник"
    )
    department: DepartmentDto | None = Field(None, description="Департамент сотрудника")
    org: OrgDto | None = Field(None, description="Организация сотрудника")
    address: str | None = Field(None, description="Адрес")
    place: str | None = Field(None, description="Площадка")
    url: str | None = Field(None, description="URL")
    email: str | None = Field(None, description="E-mail")
    phone: str | None = Field(None, description="Телефон")
    extension_number: str | None = Field(None, description="Внутренний телефон")
    office_room: str | None = Field(None, description="Кабинет")
    ldap_name: str | None = Field(None, description="LDAP Имя")
    io: bool | None = Field(None, description="Является ИО")
    have_io: bool | None = Field(None, description="Присутствуют ИО")
    notify: bool | None = Field(
        None, description="Уведомлять ли сотрудника сообщениями на почту"
    )
    create_date: datetime | None = None
    last_manual_avatar_upload_date: datetime | None = Field(
        None, description="Дата последней ручной загрузки аватара"
    )
    create_type: EmployeeCreateType | None = None
    sid: str | None = None
    current_user_leader: bool | None = None
    blocked_fields: list[BlockedField] | None = None
    order: int | None = Field(0, description="Порядок сортировки")
    ios: list[EmployeeDto] | None = None
    secretary: list[EmployeeDto] | None = None


class UserInfoDto(EdmsBaseDto):
    first_name: Annotated[str | None, Field(description="Имя сотрудника")] = None
    last_name: Annotated[str | None, Field(description="Фамилия сотрудника")] = None
    middle_name: Annotated[str | None, Field(description="Отчество сотрудника")] = None
    author_post: Annotated[
        str | None, Field(description="Наименование должность сотрудника")
    ] = None
    author_department_name: Annotated[
        str | None, Field(description="Наименование департамента сотрудника")
    ] = None
    author_department: Annotated[
        DepartmentDto | None, Field(description="Департамент сотрудника")
    ] = None
    author_department_id: Annotated[
        UUID | None, Field(description="Идентификатор департамента сотрудника")
    ] = None
    author_department_org_id: str | None = None
    employee: Annotated[EmployeeDto | None, Field(description="Сотрудник")] = None
    employee_id: Annotated[
        UUID | None, Field(description="Идентификатор сотрудник")
    ] = None
    employee_org_id: str | None = None


class EmployeeIoDto(EdmsBaseDto):
    id: UUID | None = None
    create_date: datetime | None = None
    target_id: UUID | None = None
    target: EmployeeDto | None = None
    io_id: UUID | None = None
    io: EmployeeDto | None = None
    system: bool | None = None
    revocation_date: datetime | None = None
    transfer_docs: bool | None = None
    type: EmployeeIoType | None = None


class SecretaryRequest(EdmsBaseDto):
    target_id: UUID = Field(
        ..., description="Идентификатор сотрудника за которого исполняют"
    )
    io_ids: list[UUID] | None = Field(
        None, description="Идентификаторы сотрудников которые исполняют"
    )


class EmployeeIoRequest(EdmsBaseDto):
    target_id: UUID
    io_id: UUID


class DeputyLeaderDepartmentDto(EdmsBaseDto):
    id: UUID | None = None
    organization_id: str | None = None
    department_id: UUID | None = None
    department_org_id: str | None = None
    department: Any | None = None  # DepartmentDto
    employee_id: UUID | None = None
    employee_org_id: str | None = None
    employee: EmployeeDto | None = None


class DepartmentEmployeeNomenclatureDto(EdmsBaseDto):
    id: UUID | None = None
    department_id: UUID | None = None
    department_org_id: str | None = None
    department: Any | None = None  # DepartmentDto
    employee_id: UUID | None = None
    employee: EmployeeDto | None = None


class DepartmentDto(EdmsBaseDto):
    id: UUID | None = None
    organization_id: str | None = None
    external_id: str | None = None
    name: str | None = None
    number: str | None = None
    parent_department_id: UUID | None = None
    parent_department_org_id: str | None = None
    parent_department: DepartmentDto | None = None
    rank: str | None = None
    department_code: str | None = None
    phone: str | None = None
    email: str | None = None
    address: str | None = None
    room: str | None = None
    leader_id: UUID | None = None
    leader_org_id: str | None = None
    leader: EmployeeDto | None = None
    employees: list[EmployeeDto] | None = None
    deputy_leaders: list[DeputyLeaderDepartmentDto] | None = None
    employee_nomenclatures: list[DepartmentEmployeeNomenclatureDto] | None = None
    create_date: datetime | None = None
    current_user_leader: bool | None = None
    order: int | None = 0
    child_department_write_nomenclature_affair: bool = False


class ScanSettingJsonB(EdmsBaseDto):
    """Placeholder for scan settings."""

    pass


class CurrentUserDto(EdmsBaseDto):
    """DTO reflecting the CustomUserDetails Java class."""

    principal: str | None = None
    id: UUID | None = None
    u_id: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    middle_name: str | None = None
    fired: bool | None = None
    active: bool | None = None
    organization_id: str | None = None
    full_post_name: str | None = None
    authorities: list[Any] | None = None
    role_ids: set[UUID] | None = None
    me_io: list[Any] | None = None
    secretaries: list[Any] | None = None
    post_id: int | None = None
    department_id: UUID | None = None
    department_org_id: str | None = None
    department: DepartmentDto | None = None
    post: PostDto | None = None
    group_ids: list[UUID] | None = None
    country: Any | None = None  # CountryDto
    localization: Any | None = None
    subordinates_departments: list[UUID] = Field(default_factory=list)
    subordinates: list[UUID] = Field(default_factory=list)
    org_boss: bool = False
    widgets: bool = False
    scan_wia: bool = False
    doc_ellipsis: bool = False
    doc_ellipsis_row: int = 4
    show_event_design: bool = True
    settings: ScanSettingJsonB | None = None
    responsible_nomenclature_department_ids: list[UUID] = Field(default_factory=list)
    archivist: bool = False
    grief_ids: set[UUID] | None = None
    views: list[Any] | None = None  # CustomViewDto


class UserLoginHistoryEntryDto(EdmsBaseDto):
    id: UUID | None = Field(None, description="Идентификатор записи")
    employee_id: UUID | None = Field(None, description="Идентификатор сотрудника")
    create_date: datetime | None = Field(None, description="Дата создания")
    os: str | None = Field(None, description="Операционная система")
    user_agent: str | None = Field(None, description="User Agent")
    login: str | None = Field(None, description="Логин")


class EmployeeRequest(EdmsBaseDto):
    id: UUID | None = Field(None, description="Идентификатор сотрудника")
    u_id: str = Field(..., description="Идентификатор аккаунта сотрудника")
    first_name: str = Field(..., description="Имя сотрудника")
    last_name: str = Field(..., description="Фамилия сотрудника")
    middle_name: str | None = Field(None, description="Отчество сотрудника")
    personal_number: str | None = Field(
        None, description="Персональный номер сотрудника"
    )
    ldap_name: str | None = Field(None, description="LDAP Имя")
    address: str | None = Field(None, description="Адрес")
    phone: str | None = Field(None, description="Телефон")
    extension_number: str | None = Field(None, description="Внутренний телефон")
    office_room: str | None = Field(None, description="Кабинет")
    email: str | None = Field(None, description="E-mail")
    place: str | None = Field(None, description="Площадка")
    post_id: int | None = Field(None, description="Идентификатор должности")
    url: str | None = Field(None, description="URL")
    department_id: UUID = Field(
        ..., description="Идентификатор департамента в котором работает сотрудник"
    )
    order: int = Field(0, description="Порядок сортировки")


class EmployeeAddRequest(EdmsBaseDto):
    employee: EmployeeRequest
    io_add: Any | None = None  # EmployeeIoRequest
    secretary_add: list[UUID] | None = None


class EmployeeUpdateRequest(EdmsBaseDto):
    employee: EmployeeRequest
    io_add: Any | None = None
    io_delete: UUID | None = None
    secretary_add: list[UUID] | None = None
    secretary_delete: list[UUID] | None = None


class BasicSearchEmployeeRequest(EdmsBaseDto):
    search: str | None = None
    active: bool | None = None
    fired: bool | None = None
    department_id: UUID | None = None


class EmployeeFilter(EdmsBaseDto):
    first_name: str | None = Field(None, description="Имя сотрудника")
    last_name: str | None = Field(None, description="Фамилия сотрудника")
    middle_name: str | None = Field(None, description="Отчество сотрудника")
    fired: bool | None = Field(None, description="Признак уволен ли сотрудник")
    active: bool | None = Field(None, description="Признак активен ли сотрудник")
    full_post_name: str | None = Field(None, description="ФИО сотрудника")
    post_id: int | None = Field(None, description="Идентификатор должности")
    ids: list[UUID] | None = Field(None, description="Список идентификаторов")
    department_id: list[UUID] | None = Field(
        None, description="Идентификатор департамента в котором работает сотрудник"
    )
    employee_leader_department_id: UUID | None = Field(
        None, description="Где сотрудник непосредственно их руководитель"
    )
    include_child_leaders_employee_leader_department_id: bool | None = None
    employee_leader_department_all_id: UUID | None = Field(
        None, description="Где сотрудник руководитель (включая дочерние)"
    )
    only_leaders_employee_leader_department_all: bool | None = None
    includes: list[str] | None = Field(
        None, description="Список моделей которые могу быть добавлены при отображении"
    )
    org_id: str | None = Field(None, description="ИД филиала")
    exclude_role_id: UUID | None = None
    exclude_group_id: UUID | None = None
    exclude_personal_group_id: UUID | None = None
    grief_id: UUID | None = None
    exclude_grief_id: UUID | None = None
    exclude_ids: list[UUID] | None = None
    all: bool | None = None
    child_departments: bool = Field(
        False,
        description="Признак поиска по всех коллег включая дочернии подразделения",
    )


class EmployeeApi(EdmsBaseDto):
    """Simplified model for employee management."""

    id: UUID | None = Field(None, description="Идентификатор сотрудника")
    u_id: str = Field(..., description="Идентификатор аккаунта сотрудника")
    first_name: str = Field(..., description="Имя сотрудника")
    last_name: str = Field(..., description="Фамилия сотрудника")
    middle_name: str | None = Field(None, description="Отчество сотрудника")
    fired: bool = Field(..., description="Признак уволен ли сотрудник")
    active: bool = Field(..., description="Признак активен ли сотрудник")
    full_post_name: str | None = Field(None, description="ФИО сотрудника")
    roles: set[Any] = Field(..., description="Список ролей")
    post_id: int | None = Field(None, description="Идентификатор должности")
    department_id: UUID = Field(
        ..., description="Идентификатор департамента в котором работает сотрудник"
    )


class GroupDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = Field(None, max_length=255, min_length=1)
    type: GroupType
    mixed: bool
    create_date: datetime | None = None


class DepartmentFilter(EdmsBaseDto):
    name: str | None = Field(
        None, description="Наименование департамента/подразделения"
    )
    number: str | None = Field(
        None, description="Номер департамента/подразделения в номенклатуре дел"
    )
    parent_department_id: UUID | None = Field(
        None, description="Идентификатор родительского департамента/подразделения"
    )
    rank: int | None = Field(None, description="Код сортировки")
    department_code: str | None = Field(
        None, description="Код департамента/подразделения"
    )
    phone: str | None = Field(None, description="Телефон")
    email: str | None = Field(None, description="Email")
    address: str | None = Field(None, description="Адрес")
    room: str | None = Field(None, description="Площадка")
    leader_id: UUID | None = Field(
        None, description="Идентификатор руководителя департамента/подразделения"
    )
    employees: list[Any] | None = Field(
        None, description="Список сотрудников департамента/подразделения"
    )
    ids: list[UUID] | None = None
    responsible_deps: bool = False
    includes: list[str] | None = Field(
        None, description="Список моделей которые могу быть добавлены при отображении"
    )


EmployeeDto.model_rebuild()
DepartmentDto.model_rebuild()
EmployeeIoDto.model_rebuild()
EmployeeAccessGriefDto.model_rebuild()
EmployeeAddRequest.model_rebuild()
EmployeeUpdateRequest.model_rebuild()

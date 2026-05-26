from __future__ import annotations

from typing import Any, TYPE_CHECKING, Annotated
from uuid import UUID
from datetime import datetime
from edms_ai_assistant.domain.enums import (
    DocCategory,
    RequiredFieldEnum,
    DocumentStatus,
    DocumentProcessType,
    CreateType,
    AttachmentDocumentType,
    AttachmentType,
    DeclarantType,
    FormMeetingType,
    AppealType,
    Merge,
    TaskStatus,
    TaskType,
    PeriodTaskInterval,
    PermissionType,
    ResolvePolicy,
    NomenclatureDepartmentStatus,
    SummaryNomenclatureDepartmentStatus,
    DestructionActStatus,
    AcceptanceInventoryStatus,
    ContentType,
    DocFileExtension,
    ContractControlPointStatus,
    DocumentLinkType,
)
from edms_ai_assistant.domain.appeal_fields import SubmissionFormAppeal

from pydantic import Field

from edms_ai_assistant.domain.base import EdmsBaseDto

from edms_ai_assistant.domain.employee import (
    EmployeeDto,
    UserInfoDto,
    MiniUserInfoDto,
    AccessGriefDto,
    DepartmentDto,
    RoleDto,
    GroupDto,
)
from edms_ai_assistant.domain.reference import (
    DeliveryMethodDto,
    InvestmentProgramDto,
    CitizenTypeDto,
    SubjectDto,
    SolutionResultDto,
    SubscriberDto,
    RegistrationJournalDto,
    CurrencyDto,
)

if TYPE_CHECKING:
    from edms_ai_assistant.domain.enums import (
        AttachmentDocumentType,
        CreateType,
        DocCategory,
        DeclarantType,
        DocumentProcessType,
        DocumentStatus,
    )
    from uuid import UUID
    from edms_ai_assistant.domain.appeal_fields import SubmissionFormAppeal
    from datetime import datetime


class DocumentTypeDto(EdmsBaseDto):
    id: int | None = None
    type_name: str | None = None
    doc_category_const: DocCategory | None = None
    document_appeal: DocumentAppealDto | None = None
    active: bool | None = None
    object_type: str | None = Field(None, max_length=3, min_length=3, description="вид по таблицы 2 окб")
    create_date: datetime | None = None


class AttachmentDto(EdmsBaseDto):
    name: str | None = Field(None, description="Наименование вложенного файла")
    size: int | None = Field(None, description="Размер вложенного файла")


class AttachmentDocumentDto(EdmsBaseDto):
    id: Annotated[UUID | None, Field(description="Идентификатор вложения")] = None
    name: Annotated[str | None, Field(description="Наименование вложения")] = None
    upload_date: datetime | None = None
    size: Annotated[int | None, Field(description="Размер вложения")] = None
    type: AttachmentType | None = None
    signs: Annotated[list[AttachmentSignature] | None, Field(description="Список ЭЦП вложения")] = None
    document_id: Annotated[UUID | None, Field(description="Идентификатор документа")] = None
    attachment_document_type: Annotated[AttachmentDocumentType | None, Field(description="Тип вложения")] = None
    author_id: UUID | None = None
    last_modify_user_id: UUID | None = None
    modify_date: datetime | None = None
    required_password: bool | None = None
    source_minio_name: str | None = None
    source_bucket_name: str | None = None
    source_original_name: str | None = None


class TemporaryAttachmentDto(EdmsBaseDto):
    id: UUID | None = None
    organization_id: str | None = None
    name: str | None = None
    upload_date: datetime | None = None
    size: int | None = None
    type: AttachmentType | None = None
    signs: list[AttachmentSignature] | None = None
    attachment_document_type: AttachmentDocumentType | None = None
    minio_name: str | None = None
    bucket_name: str | None = None
    tag: list[str] | None = None
    content_type: ContentType | None = None


class RenameFileRequest(EdmsBaseDto):
    name: str = Field(..., min_length=1)


class ChangeAttachmentDocumentTypeRequest(EdmsBaseDto):
    attachment_document_type: AttachmentDocumentType


class CheckAttachmentSignRequest(EdmsBaseDto):
    id: UUID
    check_smdo_cert: bool = False


class CreateEmptyFileRequest(EdmsBaseDto):
    type: AttachmentType = AttachmentType.MAIN_ATTACHMENT
    attachment_document_type: AttachmentDocumentType = AttachmentDocumentType.ATTACHMENT
    name: str | None = None
    extension: DocFileExtension = DocFileExtension.DOCX


class SimpleCmsDto(EdmsBaseDto):
    cms: str


class DocumentDto(EdmsBaseDto):
    id: Annotated[UUID | None, Field(description="Идентификатор документа")] = None
    organization_id: str | None = None
    doc_category_constant: DocCategory | None = None
    create_date: Annotated[datetime | None, Field(description="Дата создания")] = None
    days_execution: Annotated[int | None, Field(description="Кол-во дней исполнения")] = None
    profile_name: str | None = None
    required_field: Annotated[list[RequiredFieldEnum] | None, Field(description="Список обязательных полей")] = None
    author: Annotated[UserInfoDto | None, Field(description="Автор документа")] = None
    responsible_executor: UserInfoDto | None = None
    status: Annotated[DocumentStatus | None, Field(description="Статус документа")] = None
    color: DocumentUserColorDto | None = None
    prev_status: Annotated[DocumentStatus | None, Field(description="Предыдущий статус")] = None
    reg_number: Annotated[str | None, Field(description="Регистрационный номер документа")] = None
    reserved_reg_number: Annotated[str | None, Field(description="Зарезервированный рег. номер")] = None
    reserved_reg_date: Annotated[datetime | None, Field(description="Дата резервирования рег. номера")] = None
    reg_date: Annotated[datetime | None, Field(description="Дата регистрации документа")] = None
    dsp_flag: Annotated[bool | None, Field(description="Признак ДСП")] = None
    pages: Annotated[int | None, Field(description="Кол-во страниц")] = None
    exemplar_number: Annotated[int | None, Field(description="Номер экземпляра")] = None
    short_summary: Annotated[str | None, Field(description="Краткое содержание")] = None
    summary: Annotated[str | None, Field(description="Текст документа")] = None
    correspondent_name: Annotated[str | None, Field(description="Наименование корреспондента")] = None
    recipients: Annotated[bool | None, Field(description="Есть ли в документе адресаты")] = None
    profile_id: Annotated[UUID | None, Field(description="Идентификатор профиля")] = None
    document_version_id: Annotated[UUID | None, Field(description="Идентификатор версии")] = None
    control_flag: Annotated[bool | None, Field(description="Признак контроля")] = None
    remove_control: Annotated[bool | None, Field(description="Метка снятия с контроля")] = None
    journal_id: Annotated[UUID | None, Field(description="Идентификатор журнала регистрации")] = None
    document_type: Annotated[DocumentTypeDto | None, Field(description="Вид документа")] = None
    document_type_id: Annotated[int | None, Field(description="Идентификатор вида документа")] = None
    ref_doc_id: UUID | None = None
    ref_doc_org_id: str | None = None
    version_flag: bool | None = None
    formula: Annotated[list[str] | None, Field(description="Формула рег. номера")] = None
    task_list: Annotated[list[TaskDto] | None, Field(description="Список поручений")] = None
    delivery_method: Annotated[DeliveryMethodDto | None, Field(description="Способ получения")] = None
    delivery_method_id: int | None = None
    correspondent_id: Annotated[UUID | None, Field(description="Идентификатор корреспондент")] = None
    out_reg_number: Annotated[str | None, Field(description="Исходящий регистрационный номер")] = None
    out_reg_date: Annotated[datetime | None, Field(description="Исходящая дата регистрации")] = None
    additional_pages: Annotated[str | None, Field(description="Кол-во листов приложений")] = None
    exemplar_count: Annotated[int | None, Field(description="Кол-во экземлпяров")] = None
    invest_program_id: Annotated[UUID | None, Field(description="Ид инвест программы")] = None
    investment_program: Annotated[InvestmentProgramDto | None, Field(description="Инвест программа")] = None
    received_doc_id: Annotated[UUID | None, Field(description="Идентификатор полученного в ответ документа")] = None
    answer_doc_id: Annotated[UUID | None, Field(description="Идентификатор созданного в ответ документа")] = None
    received_as_answer_by_doc: Annotated[DocumentDto | None, Field(description="Документ получен в ответ")] = None
    create_as_answer_by_doc: Annotated[DocumentDto | None, Field(description="Документ создан в ответ")] = None
    version: Annotated[DocumentVersionDto | None, Field(description="Версия документа")] = None
    process: Annotated[DocumentProcessDto | None, Field(description="Процесс документа")] = None
    process_id: UUID | None = None
    attachment_document: Annotated[list[AttachmentDocumentDto] | None, Field(description="Список вложений")] = None
    correspondent: Annotated[DocumentRecipientDto | None, Field(description="Корреспондент")] = None
    recipient_list: Annotated[list[DocumentRecipientDto] | None, Field(description="Список адресатов")] = None
    control: Annotated[ControlDto | None, Field(description="Контроль")] = None
    create_type: Annotated[CreateType | None, Field(description="Тип создания")] = None
    date_meeting: Annotated[datetime | None, Field(description="Дата совещания")] = None
    date_meeting_question: Annotated[datetime | None, Field(description="Дата заседания")] = None
    start_meeting: Annotated[datetime | None, Field(description="Время начала совещания")] = None
    end_meeting: Annotated[datetime | None, Field(description="Время завершения совещания")] = None
    place_meeting: Annotated[str | None, Field(description="Место совещания")] = None
    chairperson: Annotated[UserInfoDto | None, Field(description="Председатель")] = None
    secretary: Annotated[UserInfoDto | None, Field(description="Секретарь")] = None
    external_invitees: Annotated[str | None, Field(description="Внешние приглашенные")] = None
    invitees_count: Annotated[int | None, Field(description="Количество внутренних приглашенных")] = None
    addition: Annotated[bool | None, Field(description="Признак дополнения")] = None
    addition_meeting_question_id: Annotated[UUID | None, Field(description="Ид доп. заседания")] = None
    addition_meeting_question: Annotated[DocumentDto | None, Field(description="Документ доп. заседания")] = None
    form_meeting_type: Annotated[FormMeetingType | None, Field(description="Форма заседания")] = None
    number_question: Annotated[int | None, Field(description="Порядковый номер вопроса")] = None
    document_questions: Annotated[list[Any] | None, Field(description="Список вопросов")] = None
    has_question: Annotated[bool | None, Field(description="Отметка о наличии вопросов")] = None
    document_meeting_question_id: Annotated[UUID | None, Field(description="Ид документа заседание по вопросам")] = None
    document_meeting_question_org_id: Annotated[str | None, Field(description="Ид организации документа заседание по вопросам")] = None
    document_meeting_question: Annotated[DocumentDto | None, Field(description="Документ заседание по вопросам")] = None
    who_addressed: Annotated[list[MiniUserInfoDto] | None, Field(description="Лист 'Кто подписал'")] = None
    write_off_affair_count: int | None = None
    pre_affair_count: int | None = None
    in_doc_signers: str | None = None
    country_name: str | None = None
    country_id: UUID | None = None
    introduction: list[IntroductionDto] | None = None
    introduction_count: int | None = None
    introduction_complete_count: int | None = None
    auto_control: AutoControl | None = None
    skip_registration: bool | None = None
    document_appeal: DocumentAppealDto | None = None
    responsible_executors: Annotated[list[DocumentResponsibleExecutorDto] | None, Field(description="Список ответственных")] = None
    has_responsible_executor: Annotated[bool | None, Field(description="Присутствие ответственных")] = None
    responsible_executors_count: Annotated[int | None, Field(description="количество ответственных")] = None
    document_links_count: Annotated[int | None, Field(description="Кол-во связей")] = None
    date_question: Annotated[datetime | None, Field(description="Дата заседания для вопроса")] = None
    comment_question: Annotated[str | None, Field(description="Комментарии для руководителя")] = None
    initiator: UserInfoDto | None = None
    pre_nomenclature_affairs: Annotated[list[DocumentPreNomenclatureDto] | None, Field(description="Список предварительных номенклатур")] = None
    registration_journal: RegistrationJournalDto | None = None
    who_signed: Annotated[UserInfoDto | None, Field(description="Кем подписан")] = None
    meeting_question_notify_count: int | None = None
    user_props: DocumentUserPropsDto | None = None
    note: str | None = None
    enable_access_grief: Annotated[bool | None, Field(description="Включение грифа доступа")] = None
    access_grief_id: Annotated[UUID | None, Field(description="Идентификатор грифа доступа")] = None
    access_grief: Annotated[AccessGriefDto | None, Field(description="Гриф доступа")] = None
    auto_routing: bool | None = None
    contract_sum: float | None = None
    contract_duration_start: datetime | None = None
    currency: CurrencyDto | None = None
    currency_id: UUID | None = None
    contract_signing_date: datetime | None = None
    contract_start_date: datetime | None = None
    contract_duration_end: datetime | None = None
    contract_agreement: bool | None = None
    contract_auto_prolongation: bool | None = None
    contract_typical: bool | None = None
    contract_number: str | None = None
    contract_date: datetime | None = None
    count_task: int | None = None
    task_project_count: int | None = None
    completed_task_count: int | None = None
    document_inventory_data: Any | None = None
    current_bpmn_task_name: str | None = None
    document_form_definition: Any | None = None
    custom_fields: Annotated[dict[str, Any] | None, Field(description="Пользовательские поля")] = None
    additional_documents: list[Any] | None = None
    document_form_id: UUID | None = None
    journal_number: Annotated[int | None, Field(description="Номер журнала регистрации")] = None


class DocPermissionContainer(EdmsBaseDto):
    permissions: Annotated[list[Any] | None, Field(description="Список прав доступа")] = None
    context: Annotated[Any | None, Field(description="Контекст документа")] = None


class DocumentWithPermissions(EdmsBaseDto):
    document: DocumentDto
    permission: DocPermissionContainer


class DocumentPropertiesDto(EdmsBaseDto):
    id: UUID | None = None
    organization_id: str | None = None
    auto_control: AutoControl | None = None
    gtb_auto_routing_after_signing: bool | None = None
    auto_registration: bool | None = None
    formula: list[str] | None = None
    registration_audit: list[Any] | None = None
    any_num: bool | None = None
    fields: Any | None = None


class DocumentHistoryDto(EdmsBaseDto):
    id: UUID | None = None
    action_name: str | None = None
    employee_id: UUID | None = None
    create_date: datetime | None = None
    old_value: str | None = None
    new_value: str | None = None


class ControlDto(EdmsBaseDto):
    control_type_id: Annotated[UUID | None, Field(description="Идентификатор типа контроля")] = None
    control_type_org_id: str | None = None
    control_type: Annotated[ControlTypeDto | None, Field(description="Тип контроля")] = None
    control_date_start: Annotated[datetime | None, Field(description="Дата начала контроля")] = None
    control_plan_date_end: Annotated[datetime | None, Field(description="Планируемая дата окончания контроля")] = None
    control_real_date_end: Annotated[datetime | None, Field(description="Реальная дата окончания контроля")] = None
    control_term_days: Annotated[int | None, Field(description="Срок контроля в днях")] = None
    control_initiator: Annotated[EmployeeDto | None, Field(description="Сотрудник который поставил на контроль")] = None
    control_initiator_id: Annotated[UUID | None, Field(description="Идентификатор сотрудника который поставил на контроль")] = None
    control_executor: Annotated[EmployeeDto | None, Field(description="Сотрудник который снял с контроля")] = None
    control_executor_id: Annotated[UUID | None, Field(description="Идентификатор сотрудника который снял с контроля")] = None
    on_control: Annotated[bool | None, Field(description="Активен ли сейчас контроль")] = None
    remove_control: Annotated[bool | None, Field(description="Метка снятия с контроля")] = None
    control_employees: list[EmployeeDto] | None = None


class BpmnProcessActivityDto(EdmsBaseDto):
    xml: str | None = None
    activities: list[Any] | None = None
    transient_activities: list[str] | None = None
    parsed: list[Any] | None = None
    history: set[str] | None = None
    transient_end: list[str] | None = None
    transient_start: list[str] | None = None
    incidents: list[Any] = Field(default_factory=list)


class BpmnProcessDirectoryDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = None
    doc_category: DocCategory | None = None
    create_date: datetime | None = None
    active: bool | None = None


class KanbanBoard(EdmsBaseDto):
    columns: list[Any] = Field(default_factory=list)


class TaskExecutionResult(EdmsBaseDto):
    success: bool = True
    message: str | None = None


class ChildTaskInfo(EdmsBaseDto):
    id: UUID | None = None
    count: int = 0
    children: list[TaskDto] = Field(default_factory=list)


class TasksAndProjectsDto(EdmsBaseDto):
    tasks: list[TaskDto] = Field(default_factory=list)
    task_projects: list[TaskProjectDto] = Field(default_factory=list)


class DocumentVersionDto(EdmsBaseDto):
    id: UUID | None = Field(None, description="Идентификатор версии")
    version: int | None = Field(None, description="Номер версии")
    document_id: UUID | None = Field(None, description="Идентификатор документа")
    document: DocumentDto | None = Field(None, description="Документ")
    deleted: bool | None = None


class DocumentRecipientDto(EdmsBaseDto):
    id: Annotated[UUID | None, Field(description="Идентификатор адресата документа")] = None
    document_id: Annotated[UUID | None, Field(description="Идентификатор документа")] = None
    name: Annotated[str | None, Field(description="Наименование адресата")] = None
    status: str | None = None
    sender: Annotated[EmployeeDto | None, Field(description="Отправитель", alias="from")] = None
    sender_id: Annotated[UUID | None, Field(description="ИД отправителя", alias="fromId")] = None
    date_send: Annotated[datetime | None, Field(description="Дата отправки")] = None
    to_people: Annotated[str | None, Field(description="Кому отправить")] = None
    delivered: Annotated[bool | None, Field(description="Признак доставки")] = None
    system: Annotated[bool | None, Field(description="Признак отправлено ли системой")] = None
    subscriber_id: Annotated[UUID | None, Field(description="Идентификатор адресата СМДО")] = None
    delivery_method_id: Annotated[int | None, Field(description="Идентификатор способа доставки")] = None
    delivery_method: Annotated[DeliveryMethodDto | None, Field(description="Способ доставки")] = None
    correspondent: Any | None = None
    subscriber: SubscriberDto | None = None
    correspondent_id: UUID | None = None
    type: str | None = None
    lock: bool | None = None
    aismv_package_type: str | None = None
    aismv_current_delivery_id: UUID | None = None
    aismv_router_accept: bool | None = None
    aismv_abonent_accept: bool | None = None
    aismv_router_error: bool | None = None
    aismv_abobent_accept_ack: bool | None = None
    aismv_abonent_accept_error: bool | None = None
    aismv_abonent_reg_ack: bool | None = None
    aismb_abonent_reg_reject: bool | None = None
    unp: str | None = None
    sign_date: Annotated[datetime | None, Field(description="Договор подписан контрогентом")] = None
    contract_number: Annotated[str | None, Field(description="Договор номер контракта контрагента")] = None


class ExecutionDocumentStatCount(EdmsBaseDto):
    work: int = 0
    limited: int = 0
    expire: int = 0


class DocumentUserColorDto(EdmsBaseDto):
    id: UUID | None = None
    document_id: UUID | None = None
    document_org_id: str | None = None
    color: str | None = None


class DocumentUserPropsDto(EdmsBaseDto):
    document_id: UUID | None = None
    employee_id: UUID | None = None
    create_task_count: int | None = None
    create_task_executed_count: int | None = None


class UserSmdoStat(EdmsBaseDto):
    new: Annotated[int, Field(alias="new")] = 0
    new_appeal: int = 0
    dispath: int = 0
    failed: int = 0


class CustomViewDto(EdmsBaseDto):
    id: UUID | None = None
    name: Annotated[str | None, Field(max_length=50)] = None
    columns: list[Any] | None = None
    filter: dict[str, Any] | None = None
    employee_id: UUID | None = None
    default_sort_column: Annotated[str | None, Field(max_length=100)] = None
    default_sort_direction: Annotated[str | None, Field(max_length=10)] = None


class OrgKey(EdmsBaseDto):
    id: UUID
    organization_id: str


class DocumentAccessEntryDto(EdmsBaseDto):
    id: UUID | None = None
    document_id: UUID | None = None
    document_org_id: str | None = None
    task_id: UUID | None = None
    employee_id: UUID | None = None
    department_id: UUID | None = None
    department_org_id: str | None = None
    role_id: UUID | None = None
    group_id: UUID | None = None
    group_organization_id: str | None = None
    access_id: UUID | None = None
    source_id: str | None = None
    link_type: str | None = None
    entry_type: str | None = None
    employee: EmployeeDto | None = None
    department: DepartmentDto | None = None
    role: RoleDto | None = None
    group: GroupDto | None = None


class DocumentProfileDto(EdmsBaseDto):
    id: Annotated[UUID | None, Field(description="Идентификатор профиля документа")] = None
    organization_id: str | None = None
    name: Annotated[str | None, Field(description="Наименование профиля документа")] = None
    formula: Annotated[list[str] | None, Field(description="Формула рег. номера")] = None
    auto_create_incoming_doc: bool | None = None


class RoleMergeDto(EdmsBaseDto):
    role_id: Annotated[UUID | None, Field(description="ИД роли")] = None
    merge: Annotated[Merge | None, Field(description="Политика обработки слияния ИД")] = None


class CountResult(EdmsBaseDto):
    count: int = 0


class DocumentBasedExistingBody(EdmsBaseDto):
    document_link_copy_type: str = "NONE"
    document_introduction_copy_type: str = "NONE"
    recipient_copy_type: str = "NONE"
    attachment_copy_type: str = "NONE"
    document_formula_copy_type: str | None = None
    document_affairs_copy_type: str = "DOCUMENT"
    process_copy_type: str
    base_fields_copy_type: str
    responsible_executors_copy_type: str = "NONE"


class DocumentRecipientDeliveryHistoryDto(EdmsBaseDto):
    id: Annotated[UUID | None, Field(description="Идентификатор истории доставки")] = None
    create_date: Annotated[datetime | None, Field(description="Дата создания")] = None
    log: Annotated[str | None, Field(description="Информация об изменениях")] = None
    doc_recipient_id: Annotated[UUID | None, Field(description="Идентификатор адресата")] = None


class TaskDto(EdmsBaseDto):
    id: Annotated[UUID | None, Field(description="Идентификатор поручения")] = None
    external_id: str | None = None
    type: TaskType | None = None
    organization_id: str | None = None
    parent_id: Annotated[UUID | None, Field(description="Идентификатор родительского поручения")] = None
    parent_org: str | None = None
    document_reg_date: datetime | None = None
    task_number: Annotated[str | None, Field(description="Номер поручения")] = None
    create_date: Annotated[datetime | None, Field(description="Дата создания поручения")] = None
    task_status: TaskStatus | None = None
    author: Annotated[UserInfoDto | None, Field(description="Автор поручения")] = None
    task_executors: Annotated[list[TaskExecutorsDto] | None, Field(description="Список исполнителей поручения")] = None
    planed_date_end: Annotated[datetime | None, Field(description="Запланированная дата окончания")] = None
    real_date_end: Annotated[datetime | None, Field(description="Реальная дата окончания")] = None
    task_change_date_requests: Annotated[list[Any] | None, Field(description="Список изменений переносов")] = None
    task_text: Annotated[str | None, Field(description="Текст поручения")] = None
    on_control: Annotated[bool | None, Field(description="Признак контроля")] = None
    remove_control: Annotated[bool | None, Field(description="Метка снятия с контроля")] = None
    control: Annotated[ControlDto | None, Field(description="Контроль поручения")] = None
    document_id: Annotated[UUID | None, Field(description="Идентификатор документа")] = None
    document_org_id: str | None = None
    document: Annotated[Any | None, Field(description="Документ")] = None
    revision: Annotated[bool | None, Field(description="Признак доработки")] = None
    count_exec: Annotated[int | None, Field(description="Количество исполнителей")] = None
    document_reg_num: Annotated[str | None, Field(description="Рег номер документа")] = None
    period_task_parent_number: str | None = None
    parent_task_number: str | None = None
    count_completed_exec: Annotated[int | None, Field(description="Количество исполнивших")] = None
    date_request_count: int | None = None
    executed_date_request_count: int | None = None
    endless: Annotated[bool | None, Field(description="Без срока исполнения")] = None
    period_task: bool = False
    period_task_count: int | None = None
    period: PeriodTaskInterval | None = None
    period_task_parent_id: UUID | None = None
    period_task_parent_org_id: str | None = None
    first_execution: datetime | None = None


class DocumentNextProcessRequest(EdmsBaseDto):
    id: Annotated[UUID, Field(description="Ид документа")]
    next_id: Annotated[UUID, Field(description="Ид следующего этапа")]
    employees: Annotated[list[UUID] | None, Field(description="Ид исполнителей")] = None


class DocumentCancelAction(EdmsBaseDto):
    id: UUID
    comment: str | None = None


class DocumentAismvRecreateRequest(EdmsBaseDto):
    id: UUID
    profile_id: UUID


class TaskExecutorsDto(EdmsBaseDto):
    id: Annotated[UUID | None, Field(description="Идентификатор исполнителя поручения")] = None
    organization_id: str | None = None
    task_id: Annotated[UUID | None, Field(description="Идентификатор поручения")] = None
    task_org_id: str | None = None
    task: TaskDto | None = Field(None, description="Поручение")
    create_date: Annotated[datetime | None, Field(description="Дата назначения")] = None
    executed_date: Annotated[datetime | None, Field(description="Дата исполнения")] = None
    executor: Annotated[UserInfoDto | None, Field(description="Исполнитель поручения")] = None
    task_document_links: list[Any] | None = Field(None, description="Список документов исполнения")
    attachments: list[Any] | None = Field(None, description="Список вложений прикрепленных к поручению")
    responsible: Annotated[bool | None, Field(description="Ответственный исполнитель")] = None
    executed_by_execute_for_all: Annotated[bool | None, Field(description="Исполнен методом выполнения за всех")] = None
    executed_for_all: Annotated[bool | None, Field(description="Исполнил за всех")] = None
    revision: Annotated[bool | None, Field(description="Отправлено на доработку")] = None
    parent_task_id: Annotated[UUID | None, Field(description="ИД родительского поручения")] = None
    stamp_text: Annotated[str | None, Field(description="Текст исполнения поручения")] = None
    link_count: Annotated[int | None, Field(description="Кол-во ссылок на документы")] = None
    execution_doc_count: Annotated[int | None, Field(description="Кол-во загруженных файлов")] = None
    child_task_count: int | None = None


class TaskProjectDto(EdmsBaseDto):
    id: UUID | None = None
    organization_id: str | None = None
    document_reg_num: str | None = None
    document_reg_date: datetime | None = None
    type: TaskType | None = None
    parent_id: UUID | None = None
    parent_org: str | None = None
    parent_task: Any | None = None
    create_date: datetime | None = None
    author: UserInfoDto | None = None
    task_executors: list[Any] | None = None
    planed_date_end: datetime | None = None
    task_text: str | None = None
    document_id: UUID | None = None
    document_org_id: str | None = None
    count_exec: int = 0


class ExecutionTaskStatCount(EdmsBaseDto):
    work: int = 0
    limited: int = 0
    expire: int = 0


class TaskExecutionStatByPeriod(EdmsBaseDto):
    count: int = 0


class RepeatIdenticalAppealDto(EdmsBaseDto):
    id: UUID | None = None
    doc_id: Annotated[UUID | None, Field(description="Идентификатор документа")] = None
    doc_org_id: str | None = None
    reg_number: str | None = None
    reg_date: datetime | None = None
    short_summary: str | None = None
    number: Annotated[int | None, Field(description="Номер повторного обращения")] = None
    repeat_appeal_group_id: Annotated[UUID | None, Field(description="Идентификатор группы")] = None
    type: Annotated[AppealType | None, Field(description="Тип повторных обращений")] = None
    deleted: bool = False


class PermissionDto(EdmsBaseDto):
    id: UUID | None = Field(None, description="ИД")
    system_name: str | None = Field(None, description="Системное имя")
    name: str | None = Field(None, description="Наименование")
    type: PermissionType | None = Field(None, description="Тип")
    doc_status: DocumentStatus | None = Field(None, description="Статус документа")
    doc_category: DocCategory | None = Field(None, description="Тип документа")
    profile_id: UUID | None = Field(None, description="ИД профиля документа")
    profile: DocumentProfileDto | None = Field(None, description="Профиль документа")
    merge_roles: list[RoleMergeDto] | None = Field(None, description="Политика обработки слияния ИД")
    process_completed: bool | None = Field(None, description="Признак выполнения процесса")
    current_step_completed: bool | None = Field(None, description="Признак выполнения текущего этапа")
    last_step: bool | None = Field(None, description="Признак того что текущий этап является последним")
    process_started: bool | None = Field(None, description="Признак того что процесс начал выполнение")
    on_control: bool | None = Field(None, description="Документ стоит на контроле")
    task_on_control: bool | None = Field(None, description="Поручение стоит на контроле")
    resolve_policy: ResolvePolicy | None = Field(None, description="Политики обработки доступа")
    has_reg_number: bool | None = Field(None, description="Регномер")
    document_has_items: list[str] | None = Field(None, description="Типы этапов в документе при которых доступно")
    create_type: CreateType | None = Field(None, description="Типсоздания документа")
    task_completed: bool | None = Field(None, description="Поручение исполненно")
    task_on_revision: bool | None = Field(None, description="Поручение на доработке")
    child_task: bool | None = Field(None, description="Дочернее поручение")
    task_type: TaskType | None = Field(None, description="Тип поручения")
    task_begin_execution: bool | None = Field(None, description="Поручение начало исполнение")
    archive: bool | None = Field(None, description="Документ находится в архиве")
    task_create_by_period: bool | None = Field(None, description="Поручение создано для из-за переодического выполнения")
    has_period_tasks: bool | None = Field(None, description="На основании этого поручения были созданны переодические поручения")
    nomenclature_department_status: NomenclatureDepartmentStatus | None = Field(None, description="Статус нумераторы подразделения")
    summary_nomenclature_department_status: SummaryNomenclatureDepartmentStatus | None = Field(None, description="Сводный статус подразделения")
    destruction_act_status: DestructionActStatus | None = Field(None, description="Статус акта уничтожения")
    acceptance_inventory_status: AcceptanceInventoryStatus | None = None


class PermissionRoleDto(EdmsBaseDto):
    id: UUID | None = None
    role: RoleDto | None = None
    role_id: UUID | None = None
    permission: PermissionDto | None = None
    permission_id: UUID | None = None


class ContractVersionInfoDto(EdmsBaseDto):
    id: Annotated[UUID | None, Field(description="Идентификатор версии договора")] = None
    organization_id: str | None = None
    document_id: Annotated[UUID | None, Field(description="Идентификатор договора")] = None
    document_org_id: str | None = None
    version_number: Annotated[int | None, Field(description="Номер версии договора")] = None
    create_date: Annotated[datetime | None, Field(description="Дата создания договора")] = None
    attachments: list[Any] | None = None
    file_name: Annotated[str | None, Field(description="Имя файла договора")] = None


class ContractControlPointDto(EdmsBaseDto):
    id: UUID | None = None
    organization_id: str | None = None
    sequence: int | None = None
    number: str | None = None
    status: ContractControlPointStatus | None = None
    description: str | None = None
    create_date: datetime | None = None
    deadline: datetime | None = None
    responsible_employees: list[ContractControlPointResponsibleDto] | None = None
    responsible_contractor: str | None = None
    execution_contract: DocumentDto | None = None
    execution_date: datetime | None = None
    document_id: UUID | None = None
    document_organization_id: str | None = None
    comment: str | None = None
    attachments: list[ContractControlPointAttachmentDto] | None = None
    document_links: list[ContractControlPointLinkDto] | None = None


class ContractControlPointAttachmentDto(EdmsBaseDto):
    id: UUID | None = None
    attachment: AttachmentDto | None = None
    contract_control_point_id: UUID | None = None
    contract_control_point_org_id: str | None = None
    contract_control_point: Any | None = None  # To avoid circularity, will be ContractControlPointDto
    create_date: datetime | None = None


class ContractControlPointLinkDto(EdmsBaseDto):
    id: UUID | None = None
    contract_control_point_id: UUID | None = None
    doc_link_id: UUID | None = None
    doc_link: DocumentDto | None = None
    type: DocumentLinkType | None = None


class ContractControlPointResponsibleDto(EdmsBaseDto):
    id: UUID | None = None
    organization_id: str | None = None
    contract_control_point_id: UUID | None = None
    contract_control_point_org_id: str | None = None
    user: UserInfoDto | None = None
    completed: bool = False
    create_date: datetime | None = None


class ControlPointMainFields(EdmsBaseDto):
    description: str | None = None
    number: str | None = None
    deadline: datetime | None = None
    responsible_contractor: str | None = None
    responsible_employees_add_ids: list[UUID] | None = None
    responsible_employees_del_ids: list[UUID] | None = None


class ControlPointRevisionRequest(EdmsBaseDto):
    revision_comment: str | None = None


class ContractControlPointFilter(EdmsBaseDto):
    search: str | None = None
    status: bool | None = None
    includes: list[str] | None = None


class ControlPointWithPermission(EdmsBaseDto):
    point: ContractControlPointDto
    permission: list[Any]  # list[UserDocPermission]


class ControlTypeDto(EdmsBaseDto):
    id: Annotated[UUID | None, Field(description="Идентификатор типа контроля")] = None
    organization_id: str | None = None
    name: Annotated[str | None, Field(description="Наименование типа контроля")] = None
    short_name: Annotated[str | None, Field(description="Краткое наименование")] = None
    term: Annotated[int | None, Field(description="Срок контроля (дней)")] = None
    employees: list[EmployeeDto] | None = None
    active: Annotated[bool | None, Field(description="Признак активности")] = None
    deleted: bool | None = None
    create_date: datetime | None = None


class SignatureDto(EdmsBaseDto):
    key_id: str | None = Field(None, description="Идентификатор открытого ключа подписавшего")
    signer: str | None = Field(None, description="Имя подписавшего")
    signtime: datetime | None = Field(None, description="Дата/время подписи")
    operation_type: str | None = Field(None, description="Тип операции подписания")
    orig_signature: str | None = Field(None, description="Значение ЭЦП в исходной системе")
    data: str = Field(..., description="ЭЦП подпись в base64")
    cert_serial: str | None = Field(None, description="Номер сертификата")
    signer_fio: str | None = Field(None, description="ФИО подписавшего")
    signer_date: datetime | None = Field(None, description="Дата подписания")
    signer_post: str | None = Field(None, description="Должность подписавшего")
    signer_org: str | None = Field(None, description="Организация подписавшего")
    personal_number: str | None = Field(None, description="Личный номер")
    issuer: str | None = Field(None, description="Издатель сертификата")
    start: datetime | None = None
    end: datetime | None = None
    sign_count: int | None = None
    attr_cert_issuer: str | None = None
    attr_cert_issuer_id: str | None = None
    attr_organization_name: str | None = None
    attr_post: str | None = None
    attr_unp: str | None = None
    attr_unpf: str | None = None
    attr_address: str | None = None
    attr_start: datetime | None = None
    attr_end: datetime | None = None


class AttachmentSignature(EdmsBaseDto):
    id: Annotated[UUID | None, Field(description="Идентификатор ЭЦП")] = None
    date: Annotated[datetime | None, Field(description="Дата формирования ЭЦП")] = None
    check: Annotated[bool | None, Field(description="Проверка валидности ЭЦП")] = None
    sign: SignatureDto | None = None


class DocumentPreNomenclatureDto(EdmsBaseDto):
    id: Annotated[UUID | None, Field(description="Идентификатор документа в номенклатуре")] = None
    document_id: Annotated[UUID | None, Field(description="Идентификатор документа")] = None
    document: Annotated[Any | None, Field(description="Документ")] = None
    nomenclature_affair_id: Annotated[UUID | None, Field(description="Идентификатор номенклатуры")] = None
    nomenclature_affair: Annotated[Any | None, Field(description="Номенклатура дел")] = None
    write_off: Annotated[bool | None, Field(description="Метка списания")] = None
    deleted: bool | None = None


class NomenclatureAffairDto(EdmsBaseDto):
    id: Annotated[UUID | None, Field(description="Идентификатор номенклатуры дел")] = None
    external_id: str | None = None
    organization_id: str | None = None
    status: str | None = None
    create_type: str | None = None
    calendar_year: int | None = None
    transitional_affair: bool | None = None
    collecting_start: datetime | None = None
    collecting_end: datetime | None = None
    personnel: bool | None = None
    active: bool | None = None
    department_id: UUID | None = None
    department: DepartmentDto | None = None
    name: str | None = None
    index: str | None = None
    storage_period_id: UUID | None = None
    period: Any | None = None
    article_number: str | None = None
    storage_place: str | None = None
    note: str | None = None
    attachment: Any | None = None
    department_path: list[Any] | None = None
    write_off: bool | None = None
    doc_counter: int | None = None
    inner_inventory_data: Any | None = None
    expiration_date: int | None = None
    expiration_date_full: datetime | None = None
    method_assembly_type: str | None = None
    expertise_type: str | None = None
    expertise: bool = False
    document_classes_id: UUID | None = None
    document_classes: Any | None = None
    count_sheets: int | None = None
    number_pages_per_volume: int | None = None
    case_file_cover: Any | None = None
    certification_sheet: Any | None = None
    acceptance_inventory_id: UUID | None = None
    acceptance_inventory_org_id: str | None = None
    acceptance_inventory: Any | None = None
    has_inner: bool = False
    has_inner_archive: bool = False
    archive_date: datetime | None = None
    who_accepted: UserInfoDto | None = None
    fund_id: UUID | None = None
    fund_org_id: str | None = None
    fund: Any | None = None
    shelving_shelve_id: UUID | None = None
    shelving_shelve_org_id: str | None = None
    shelving_shelve: Any | None = None
    information: str | None = None
    case_file_cover_id: UUID | None = None
    certification_sheet_id: UUID | None = None
    destruction_acts: list[Any] | None = None
    document_size: int = 0
    hash_ed: Annotated[str | None, Field(alias="hashED")] = None
    signs: list[AttachmentSignature] | None = None
    completed: bool = False


class DocumentFormField(EdmsBaseDto):
    id: str | None = None
    payload: dict[str, dict[str, Any]] | None = None


class DocumentAppealDto(EdmsBaseDto):
    country_appeal_id: Annotated[UUID | None, Field(description="Идентификатор страны заявителя")] = None
    country_appeal_name: Annotated[str | None, Field(description="Наименование страны заявителя")] = None
    receipt_date: Annotated[datetime | None, Field(description="Содержит дату поступления")] = None
    declarant_type: Annotated[DeclarantType | None, Field(description="Тип заявителя")] = None
    citizen_type_id: Annotated[UUID | None, Field(description="Идентификатор вида обращения")] = None
    citizen_type: CitizenTypeDto | None = None
    collective: Annotated[bool | None, Field(description="Признак коллективного обращения")] = None
    anonymous: Annotated[bool | None, Field(description="Признак анонимного обращения")] = None
    fio_applicant: Annotated[str | None, Field(description="Содержит Ф.И.О. заявителя")] = None
    city_id: Annotated[UUID | None, Field(description="Идентификатор города")] = None
    city_name: Annotated[str | None, Field(description="Город")] = None
    region_id: Annotated[UUID | None, Field(description="Идентификатор области")] = None
    region_name: Annotated[str | None, Field(description="Область")] = None
    district_id: Annotated[UUID | None, Field(description="Идентификатор района")] = None
    district_name: Annotated[str | None, Field(description="Район")] = None
    index: Annotated[str | None, Field(description="Индекс")] = None
    full_address: Annotated[str | None, Field(description="Содержит улицу, дом, корпус, квартиру")] = None
    phone: Annotated[str | None, Field(description="Контактный телефон")] = None
    email: Annotated[str | None, Field(description="e-mail заявителя")] = None
    organization_name: Annotated[str | None, Field(description="Наименование организации-заявителя")] = None
    signed: Annotated[str | None, Field(description="Ф.И.О. лица, подписавшего документ")] = None
    correspondent_org_number: Annotated[str | None, Field(description="Исходящий рег. номер организации-корреспондента")] = None
    date_doc_correspondent_org: Annotated[datetime | None, Field(description="Дата рег. номера организации-корреспондента")] = None
    subject_id: Annotated[UUID | None, Field(description="Идентификатор тематики")] = None
    subject: SubjectDto | None = None
    correspondent_appeal: Annotated[str | None, Field(description="Организация, переславшая обращение")] = None
    correspondent_appeal_id: UUID | None = None
    index_date_cover_letter: Annotated[str | None, Field(description="Дата и индекс сопроводительного письма")] = None
    repeat_identical_appeals: Annotated[list[RepeatIdenticalAppealDto] | None, Field(description="Повторные и идентичные обращения")] = None
    review_progress: Annotated[str | None, Field(description="Ход рассмотрения")] = None
    solution_result_id: Annotated[UUID | None, Field(description="Идентификатор результата решения")] = None
    solution_result: SolutionResultDto | None = None
    nomenclature_affair_id: Annotated[UUID | None, Field(description="Идентификатор номенклатуры дел")] = None
    nomenclature_affair: NomenclatureAffairDto | None = None
    reasonably: bool | None = None
    submission_form: SubmissionFormAppeal | None = None


class DocumentProcessDto(EdmsBaseDto):
    id: UUID | None = None
    current_id: UUID | None = None
    current: DocumentProcessItemDto | None = None
    next_id: UUID | None = None
    document_id: UUID | None = None
    items: list[DocumentProcessItemDto] | None = None
    completed: bool | None = None
    started: bool | None = None


class DocumentProcessItemDto(EdmsBaseDto):
    id: UUID | None = None
    process_id: UUID | None = None
    name: str | None = None
    type: DocumentProcessType | None = None
    order: int | None = None
    completed: bool | None = None


class AutoControl(EdmsBaseDto):
    auto_control: bool | None = None
    control_days: int | None = None
    control_type_id: UUID | None = None
    control_type_org_id: str | None = None


class DocumentResponsibleExecutorDto(EdmsBaseDto):
    id: UUID | None = None
    document_id: UUID | None = None
    document: Any | None = None
    executor: UserInfoDto | None = None


class IntroductionDto(EdmsBaseDto):
    id: UUID | None = None
    create_date: datetime | None = None
    author: UserInfoDto | None = None
    document_id: UUID | None = None
    introduction_date: datetime | None = None
    introduction_stamp: UserInfoDto | None = None
    comment: str | None = None


DocumentTypeDto.model_rebuild()
AttachmentDocumentDto.model_rebuild()
TemporaryAttachmentDto.model_rebuild()
DocumentProfileDto.model_rebuild()
RoleMergeDto.model_rebuild()
TaskDto.model_rebuild()
TaskExecutorsDto.model_rebuild()
TaskProjectDto.model_rebuild()
ChildTaskInfo.model_rebuild()
TasksAndProjectsDto.model_rebuild()
KanbanBoard.model_rebuild()
PermissionDto.model_rebuild()
PermissionRoleDto.model_rebuild()
DocumentDto.model_rebuild()
DocumentAppealDto.model_rebuild()
DocumentProcessDto.model_rebuild()
DocumentProcessItemDto.model_rebuild()
DocumentResponsibleExecutorDto.model_rebuild()
NomenclatureAffairDto.model_rebuild()
ContractControlPointDto.model_rebuild()
ContractControlPointAttachmentDto.model_rebuild()
ContractControlPointLinkDto.model_rebuild()
ContractControlPointResponsibleDto.model_rebuild()
ControlPointWithPermission.model_rebuild()

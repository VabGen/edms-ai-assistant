from __future__ import annotations

from typing import Any, TYPE_CHECKING

from pydantic import Field

from edms_ai_assistant.domain.base import EdmsBaseDto

if TYPE_CHECKING:
    from edms_ai_assistant.domain.enums import (
        GroupByStoragePeriod,
        ReminderType,
        StoragePeriodType,
        YearPostfix,
    )
    from uuid import UUID
    from datetime import datetime


class ReferenceItemDto(EdmsBaseDto):
    id: UUID | int | None = None
    name: str | None = None
    active: bool | None = None
    create_date: datetime | None = None


class CityHierarchyDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = None
    district_id: UUID | None = None
    district_name: str | None = None
    region_id: UUID | None = None
    region_name: str | None = None


class SubjectDto(EdmsBaseDto):
    id: UUID | None = Field(None, description="Идентификатор тематики")
    name: str | None = Field(None, description="Наименование тематики")
    code: int | None = Field(None, description="Код тематики")
    parent_subject_id: UUID | None = Field(None, description="Идентификатор родительской тематики")
    parent_subject: Any | None = Field(None, description="Родительская тематика")
    active: bool | None = None
    deleted: bool | None = None
    create_date: datetime | None = None


class CountryDto(EdmsBaseDto):
    id: UUID | None = Field(None, description="Идентификатор страны")
    code: int | None = Field(None, ge=1, le=999, description="Цифровой код страны")
    short_name: str | None = Field(None, description="Краткое наименование страны")
    full_name: str | None = Field(None, description="Полное наименование страны")
    alpha_code_two: str | None = None
    alpha_code_three: str | None = None
    active: bool | None = None
    create_date: datetime | None = None


class RegionDto(EdmsBaseDto):
    id: UUID | None = None
    name_region: str | None = None
    code_region: int | None = None
    active: bool | None = None
    create_date: datetime | None = None


class CurrencyDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = None
    active: bool | None = None
    deleted: bool | None = None
    create_date: datetime | None = None
    code: str | None = None


class DeliveryMethodDto(EdmsBaseDto):
    id: int | None = None
    delivery_name: str | None = None
    aismv: bool | None = None
    kancler_next: bool | None = None
    active: bool | None = None
    system: bool | None = None
    create_date: datetime | None = None


class ArchiveFundDto(EdmsBaseDto):
    id: UUID | None = None
    code: str | None = None
    name: str | None = None
    active: bool | None = None


class NomenclatureShelvingShelveDto(EdmsBaseDto):
    id: UUID | None = None
    fund_id: UUID | None = None
    fund: ArchiveFundDto | None = None
    shelving_name: str | None = None
    shelve_name: str | None = None
    active: bool | None = None


class StoragePeriodDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = None
    group_by_storage_period: GroupByStoragePeriod | None = None
    active: bool | None = None
    create_date: datetime | None = None
    by_staff: bool | None = None
    storage_years: int | None = None
    year_postfix: YearPostfix | None = None
    storage_period_type: StoragePeriodType | None = None


class RegistrationJournalDto(EdmsBaseDto):
    id: UUID | None = None
    journal_name: str | None = None
    counter_value: int | None = None
    active: bool | None = None
    create_date: datetime | None = None


class DocumentLanguageDto(EdmsBaseDto):
    id: UUID | None = None
    language_code: str | None = None
    active: bool | None = None


class CitizenTypeDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = None
    active: bool | None = None
    report: bool | None = None
    create_date: datetime | None = None


class AdditionalDocumentTypeDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = None
    active: bool | None = None
    deleted: bool | None = None
    create_date: datetime | None = None


class SolutionResultDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = None
    active: bool | None = None
    create_date: datetime | None = None


class UnifiedDocumentationSystemDto(EdmsBaseDto):
    id: UUID | None = None
    code: str | None = None
    name: str | None = None
    active: bool | None = None


class OrgDto(EdmsBaseDto):
    """Organization model."""
    id: str | None = Field(None, description="Идентификатор организации")
    name: str | None = Field(None, description="Наименование организации")
    current_count: int | None = Field(None, description="Текущее кол-во активных учетных записей в СЭД")
    max_count: int | None = Field(None, description="Максимально кол-во активных учетных записей")
    active: bool | None = Field(None, description="Признак активена ли организация")


class InvestmentProgramDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = None
    active: bool | None = None
    create_date: datetime | None = None


class SubscriberDto(EdmsBaseDto):
    id: UUID | None = None
    smdo_code: str | None = None
    end_date: str | None = None
    connection_date: str | None = None
    eas_id: str | None = None
    oais_abonent: bool | None = None
    start_date: str | None = None
    status: str | None = None
    fax: str | None = None
    home: str | None = None
    phone: str | None = None
    subscriber_status: str | None = None
    unp: str | None = None
    corpus: str | None = None
    email: str | None = None
    post_index: str | None = None
    soato: str | None = None
    street: str | None = None
    objid: UUID | None = None
    row_id: str | None = None
    created_on: datetime | None = None
    updated_on: datetime | None = None
    oais_org_id: UUID | None = None
    name: str | None = None
    short_name: str | None = None
    sed_type_id: UUID | None = None
    sed_type_object_id: UUID | None = None
    typesed_row_id: str | None = None
    typesed_value: str | None = None
    create_date: datetime | None = None
    updated_date: datetime | None = None


class TypicalSummaryDto(EdmsBaseDto):
    id: UUID | None = None
    summary_text: str | None = Field(None, max_length=255)
    active: bool | None = None
    create_date: datetime | None = None


class TypicalSummaryTaskDto(EdmsBaseDto):
    id: UUID | None = None
    summary_task_text: str | None = None
    active: bool | None = None
    create_date: datetime | None = None


class TypicalControlPointDto(EdmsBaseDto):
    id: UUID | None = None
    name: str | None = None
    active: bool | None = None
    create_date: datetime | None = None


class RemindersRulesDto(EdmsBaseDto):
    id: ReminderType | None = None
    organization_id: str | None = None
    cron: str | None = None
    plain_date: datetime | None = None
    days: int | None = None
    active: bool | None = None
    subject: str | None = None
    text: str | None = None
    ignore_weekends: bool | None = None

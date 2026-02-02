# edms_ai_assistant/models/appeal_fields.py
import re
from datetime import datetime
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class DeclarantType(StrEnum):
    """Тип заявителя: физическое или юридическое лицо"""
    PHYSICAL = "PHYSICAL"
    LEGAL = "LEGAL"


class AppealFields(BaseModel):
    """
    Модель данных для автозаполнения полей документа категории APPEAL.
    Соответствует Java-классу DocumentAppealFields.
    """

    # === ОСНОВНЫЕ ПОЛЯ ===
    deliveryMethod: Optional[str] = Field(
        None,
        description="Способ доставки обращения (например, 'Почта России', 'Электронная почта', 'Курьер')"
    )

    shortSummary: Optional[str] = Field(
        None,
        max_length=200,
        description="Краткое содержание обращения (максимум 200 символов)"
    )

    # === ИНФОРМАЦИЯ О ЗАЯВИТЕЛЕ ===
    fioApplicant: Optional[str] = Field(
        None,
        description="Ф.И.О. заявителя в формате 'Фамилия Имя Отчество'"
    )

    receiptDate: Optional[datetime] = Field(
        None,
        description="Дата поступления обращения"
    )

    declarantType: Optional[DeclarantType] = Field(
        None,
        description="Признак, является заявитель физическим или юридическим лицом"
    )

    # === ПРИЗНАКИ ОБРАЩЕНИЯ ===
    collective: Optional[bool] = Field(
        None,
        description="Признак коллективного обращения"
    )

    anonymous: Optional[bool] = Field(
        None,
        description="Признак анонимного обращения"
    )

    reasonably: Optional[bool] = Field(
        None,
        description="Признак обоснованности обращения"
    )

    # === КАТЕГОРИЗАЦИЯ ===
    citizenType: Optional[str] = Field(
        None,
        description="Вид обращения (Жалоба, Благодарность, Предложение, Заявление и т.д.)"
    )

    themeName: Optional[str] = Field(
        None,
        description="Тематика обращения"
    )

    subThemeName: Optional[str] = Field(
        None,
        description="Подтематика обращения"
    )

    # === ГЕОГРАФИЧЕСКИЕ ДАННЫЕ ===
    country: Optional[str] = Field(
        None,
        description="Страна заявителя"
    )

    regionName: Optional[str] = Field(
        None,
        description="Область проживания заявителя"
    )

    districtName: Optional[str] = Field(
        None,
        description="Район проживания заявителя"
    )

    cityName: Optional[str] = Field(
        None,
        description="Город проживания заявителя"
    )

    index: Optional[str] = Field(
        None,
        description="Почтовый индекс (6 цифр)"
    )

    fullAddress: Optional[str] = Field(
        None,
        description="Полный адрес: улица, дом, корпус, квартира"
    )

    # === КОНТАКТНЫЕ ДАННЫЕ ===
    phone: Optional[str] = Field(
        None,
        description="Контактный телефон заявителя"
    )

    email: Optional[str] = Field(
        None,
        description="Электронная почта заявителя"
    )

    # === ОРГАНИЗАЦИЯ (для юридических лиц) ===
    organizationName: Optional[str] = Field(
        None,
        description="Наименование организации-заявителя"
    )

    signed: Optional[str] = Field(
        None,
        description="Ф.И.О. лица, подписавшего документ"
    )

    # === КОРРЕСПОНДЕНТ ===
    correspondentAppeal: Optional[str] = Field(
        None,
        description="Организация, которая приняла обращение или в которую подается обращение"
    )

    correspondentOrgNumber: Optional[str] = Field(
        None,
        description="Исходящий регистрационный индекс, присвоенный документу организацией-корреспондентом"
    )

    dateDocCorrespondentOrg: Optional[datetime] = Field(
        None,
        description="Дата присвоения организацией-корреспондентом документу исходящего регистрационного номера"
    )

    # === ДОПОЛНИТЕЛЬНЫЕ ПОЛЯ ===
    indexDateCoverLetter: Optional[str] = Field(
        None,
        description="Дата и индекс сопроводительного письма"
    )

    reviewProgress: Optional[str] = Field(
        None,
        description="Ход рассмотрения обращения"
    )

    # === ВАЛИДАТОРЫ ===

    @field_validator('shortSummary')
    @classmethod
    def truncate_summary(cls, v: Optional[str]) -> Optional[str]:
        """Обрезает краткое содержание до 200 символов"""
        if v and len(v) > 200:
            return v[:197] + "..."
        return v

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        """Проверяет формат email"""
        if v and v != "Неизвестно":
            if '@' not in v or '.' not in v.split('@')[-1]:
                return None
        return v

    @field_validator('phone')
    @classmethod
    def normalize_phone(cls, v: Optional[str]) -> Optional[str]:
        """Нормализует номер телефона"""
        if v and v != "Неизвестно":
            cleaned = re.sub(r'[^\d+]', '', v)
            if len(cleaned) >= 10:  # Минимальная длина телефона
                return cleaned
        return v

    @field_validator('index')
    @classmethod
    def validate_index(cls, v: Optional[str]) -> Optional[str]:
        """Проверяет формат почтового индекса (6 цифр)"""
        if v and v != "Неизвестно":
            cleaned = re.sub(r'\D', '', v)
            if len(cleaned) == 6:
                return cleaned
            return None
        return v

    @model_validator(mode='after')
    def replace_unknown_with_none(self):
        """Заменяет значения 'Неизвестно' на None для всех строковых полей"""
        for field_name, field_value in self.model_dump().items():
            if isinstance(field_value, str) and field_value.strip().lower() in ['неизвестно', 'unknown', 'н/д', 'n/a']:
                setattr(self, field_name, None)
        return self

    @model_validator(mode='after')
    def validate_applicant_info(self):
        """Проверяет базовую консистентность данных заявителя"""
        if self.anonymous and self.fioApplicant:
            # Анонимное обращение не должно содержать ФИО
            self.fioApplicant = None

        if self.declarantType == DeclarantType.LEGAL and not self.organizationName:
            # Для юридических лиц желательно указывать название организации
            pass

        return self

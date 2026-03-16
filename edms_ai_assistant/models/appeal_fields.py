# edms_ai_assistant/models/appeal_fields.py
"""
Модели данных для автоматического заполнения карточек обращений граждан (APPEAL).
"""

import logging
import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class DeclarantType(StrEnum):
    """
    Тип заявителя согласно справочнику СЭД.

    Attributes:
        INDIVIDUAL: Физическое лицо (гражданин)
        ENTITY: Юридическое лицо (организация, госорган)
    """

    INDIVIDUAL = "INDIVIDUAL"
    ENTITY = "ENTITY"


class AppealFields(BaseModel):
    """
    Структура данных для извлечения информации из текста обращения с помощью LLM.

    Attributes:
        deliveryMethod: Способ доставки обращения (например, "Почта", "Email", "Курьер")
        shortSummary: Краткое содержание обращения (до 200 символов)
        receiptDate: Дата поступления/написания обращения
        declarantType: Тип заявителя (INDIVIDUAL или ENTITY)
        citizenType: Категория/вид обращения (например, "Жалоба", "Заявление")
        collective: Признак коллективного обращения (подписано группой лиц)
        anonymous: Признак анонимного обращения (без указания ФИО)
        reasonably: Признак обоснованности (наличие фактов, ссылок на законы)
        fioApplicant: ФИО заявителя (для INDIVIDUAL)
        organizationName: Наименование организации-заявителя (для ENTITY)
        signed: ФИО лица, подписавшего документ (для ENTITY)
        correspondentOrgNumber: Исходящий номер документа (для ENTITY)
        dateDocCorrespondentOrg: Дата исходящего номера (для ENTITY)
        country: Название страны
        regionName: Название региона/области
        districtName: Название района
        cityName: Название города
        index: Почтовый индекс (6 цифр)
        fullAddress: Полный почтовый адрес
        phone: Контактный телефон
        email: Email для связи
        correspondentAppeal: Название организации, которая переслала обращение
        indexDateCoverLetter: Индекс и дата сопроводительного письма
        reviewProgress: Информация о ходе рассмотрения
    """

    # --- Основные поля ---
    deliveryMethod: str | None = Field(None, description="Способ доставки обращения")
    shortSummary: str | None = Field(
        None, max_length=200, description="Краткое содержание (до 200 символов)"
    )
    receiptDate: datetime | None = Field(None, description="Дата поступления обращения")
    declarantType: DeclarantType | None = Field(
        None, description="Тип заявителя: INDIVIDUAL (физлицо) или ENTITY (юрлицо)"
    )
    citizenType: str | None = Field(
        None, description="Категория обращения (Жалоба, Заявление, Предложение)"
    )

    # --- Логические признаки ---
    collective: bool | None = Field(
        None, description="Коллективное обращение (true/false)"
    )
    anonymous: bool | None = Field(None, description="Анонимное обращение (true/false)")
    reasonably: bool | None = Field(
        None, description="Обоснованность обращения (есть факты/аргументы)"
    )

    # --- Заявитель (Физическое лицо) ---
    fioApplicant: str | None = Field(None, description="ФИО заявителя")

    # --- Заявитель (Юридическое лицо) ---
    organizationName: str | None = Field(
        None, description="Наименование организации-заявителя"
    )
    signed: str | None = Field(None, description="ФИО лица, подписавшего документ")
    correspondentOrgNumber: str | None = Field(
        None, description="Исходящий номер документа организации"
    )
    dateDocCorrespondentOrg: datetime | None = Field(
        None, description="Дата исходящего документа организации"
    )

    # --- География ---
    country: str | None = Field(None, description="Страна заявителя")
    regionName: str | None = Field(None, description="Регион/Область")
    districtName: str | None = Field(None, description="Район")
    cityName: str | None = Field(None, description="Город/Населенный пункт")
    index: str | None = Field(None, description="Почтовый индекс (6 цифр)")
    fullAddress: str | None = Field(
        None, description="Полный почтовый адрес (улица, дом, квартира)"
    )

    # --- Контактная информация ---
    phone: str | None = Field(None, description="Контактный телефон")
    email: str | None = Field(None, description="Email для связи")

    # --- Дополнительные сведения ---
    correspondentAppeal: str | None = Field(
        None, description="Организация, переславшая обращение (если применимо)"
    )
    indexDateCoverLetter: str | None = Field(
        None, description="Индекс и дата сопроводительного письма"
    )
    reviewProgress: str | None = Field(
        None, description="Информация о ходе рассмотрения"
    )

    @field_validator("shortSummary")
    @classmethod
    def truncate_summary(cls, v: str | None) -> str | None:
        if v and len(v) > 200:
            logger.debug(f"Краткое содержание обрезано: {len(v)} → 200 символов")
            return v[:197] + "..."
        return v

    @field_validator("index")
    @classmethod
    def validate_index(cls, v: str | None) -> str | None:
        if v:
            cleaned = re.sub(r"\D", "", str(v))
            if len(cleaned) != 6:
                return None
            return cleaned
        return v

    @model_validator(mode="after")
    def clean_placeholders(self) -> "AppealFields":
        placeholders = {
            "none",
            "null",
            "nil",
            "unknown",
            "n/a",
            "na",
            "no",
            "not specified",
            "not available",
            "неизвестно",
            "н/д",
            "нет данных",
            "отсутствует",
            "не указано",
            "нет",
            "—",
            "-",
            "–",
            "...",
            "___",
        }

        for field_name in self.model_fields:
            value = getattr(self, field_name)

            if isinstance(value, str):
                trimmed_value = value.strip()

                if trimmed_value.lower() in placeholders or not trimmed_value:
                    setattr(self, field_name, None)

        return self

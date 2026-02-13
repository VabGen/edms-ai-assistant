"""
EDMS AI Assistant - Appeal Extraction Service
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.models.appeal_fields import AppealFields

logger = logging.getLogger(__name__)


class AppealExtractionService:
    """
    Сервис для извлечения структурированных данных из обращений граждан.
    """

    MIN_TEXT_LENGTH = 30
    MAX_TEXT_LENGTH = 12000
    DEFAULT_MAX_RETRIES = 3
    BASE_RETRY_DELAY = 2

    def __init__(self):
        base_llm = get_chat_model()
        self.extraction_llm = base_llm.with_config({"temperature": 0.0})
        logger.info("AppealExtractionService initialized with temperature=0.0")

    async def extract_appeal_fields(self, text: str) -> AppealFields:
        if not self._validate_text_length(text):
            return AppealFields()

        try:
            parser = JsonOutputParser(pydantic_object=AppealFields)
            prompt = self._build_extraction_prompt()
            chain = prompt | self.extraction_llm | parser

            truncated_text = self._truncate_text(text)

            logger.debug(
                "Invoking LLM for extraction",
                extra={"text_length": len(truncated_text)},
            )

            result = await chain.ainvoke(
                {
                    "text": truncated_text,
                    "format_instructions": parser.get_format_instructions(),
                }
            )

            appeal_data = AppealFields.model_validate(result)

            appeal_data = self._post_process_fields(appeal_data, truncated_text)

            logger.info(
                "Appeal data extracted successfully",
                extra={
                    "has_fio": bool(appeal_data.fioApplicant),
                    "has_org": bool(appeal_data.organizationName),
                    "declarant_type": appeal_data.declarantType,
                    "citizen_type": appeal_data.citizenType,
                    "has_city": bool(appeal_data.cityName),
                },
            )

            return appeal_data

        except Exception as e:
            logger.error(
                f"LLM extraction failed: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return AppealFields()

    def _post_process_fields(self, fields: AppealFields, text: str) -> AppealFields:
        """
        Post-processing для извлечения данных, которые LLM мог пропустить.

        Fixes:
        1. Парсинг даты из correspondentOrgNumber если dateDocCorrespondentOrg пуст
        2. Извлечение города из fullAddress если cityName пуст
        """
        if fields.declarantType == "ENTITY":
            if not fields.dateDocCorrespondentOrg and fields.correspondentOrgNumber:
                parsed_date = self._parse_date_from_number(
                    fields.correspondentOrgNumber
                )
                if parsed_date:
                    fields.dateDocCorrespondentOrg = parsed_date
                    logger.info(
                        f"Parsed date from correspondentOrgNumber: {parsed_date}"
                    )

        if not fields.cityName and fields.fullAddress:
            extracted_city = self._extract_city_from_address(fields.fullAddress)
            if extracted_city:
                fields.cityName = extracted_city
                logger.info(f"Extracted city from address: {extracted_city}")

        return fields

    def _parse_date_from_number(self, number: str) -> Optional[str]:
        """
        Парсит дату из исходящего номера документа.

        Patterns:
        - "№ 123 от 15.01.2025"
        - "№ 01-01/26" (26 = 2026 год, 01 = январь)
        - "исх. № 45/2 от 12.02.2024"
        """
        date_patterns = [
            r"от\s+(\d{1,2})\.(\d{1,2})\.(\d{4})",
            r"от\s+(\d{1,2})\s+([а-яА-Я]+)\s+(\d{4})",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, number)
            if match:
                try:
                    day, month, year = match.groups()

                    if month.isdigit():
                        month_num = int(month)
                    else:
                        month_map = {
                            "января": 1,
                            "февраля": 2,
                            "марта": 3,
                            "апреля": 4,
                            "мая": 5,
                            "июня": 6,
                            "июля": 7,
                            "августа": 8,
                            "сентября": 9,
                            "октября": 10,
                            "ноября": 11,
                            "декабря": 12,
                        }
                        month_num = month_map.get(month.lower(), 1)

                    dt = datetime(int(year), month_num, int(day))
                    return dt.isoformat() + "Z"
                except (ValueError, KeyError):
                    pass

        slash_pattern = r"(\d{1,2})-(\d{1,2})/(\d{2})"
        match = re.search(slash_pattern, number)
        if match:
            try:
                month_num, day_num, year_short = match.groups()
                year_full = 2000 + int(year_short)
                dt = datetime(year_full, int(month_num), int(day_num))
                return dt.isoformat() + "Z"
            except ValueError:
                pass

        return None

    def _extract_city_from_address(self, address: str) -> Optional[str]:
        """
        Извлекает город из адреса.

        Patterns:
        - "г. Минск"
        - "Минск, ул. ..."
        - "220004, Минск"
        """
        city_patterns = [
            r"г\.\s*([А-ЯЁ][а-яё]+)",
            r"(\d{6}),?\s*([А-ЯЁ][а-яё]+)",
            r"([А-ЯЁ][а-яё]+),\s*ул\.",
        ]

        for pattern in city_patterns:
            match = re.search(pattern, address)
            if match:
                if len(match.groups()) == 2:
                    return match.group(2).strip()
                return match.group(1).strip()

        return None

    async def extract_with_retry(
        self,
        text: str,
        max_attempts: Optional[int] = None,
    ) -> AppealFields:
        max_attempts = max_attempts or self.DEFAULT_MAX_RETRIES

        for attempt in range(1, max_attempts + 1):
            try:
                result = await self.extract_appeal_fields(text)

                if self._is_valid_extraction(result):
                    logger.info(
                        f"Extraction successful on attempt {attempt}/{max_attempts}",
                        extra={"attempt": attempt, "max_attempts": max_attempts},
                    )
                    return result

                logger.warning(
                    f"Attempt {attempt}/{max_attempts}: LLM returned insufficient data",
                    extra={"attempt": attempt},
                )

            except Exception as e:
                logger.error(
                    f"Attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {e}",
                    extra={"attempt": attempt, "error": str(e)},
                )

            if attempt < max_attempts:
                wait_time = self._calculate_retry_delay(attempt)
                logger.info(
                    f"Waiting {wait_time}s before retry...",
                    extra={"wait_time": wait_time, "attempt": attempt},
                )
                await asyncio.sleep(wait_time)

        logger.error(
            f"Extraction failed after {max_attempts} attempts. Returning empty object.",
            extra={"max_attempts": max_attempts},
        )
        return AppealFields()

    def _validate_text_length(self, text: str) -> bool:
        if not text:
            logger.warning("Empty text provided for extraction")
            return False

        text_length = len(text.strip())

        if text_length < self.MIN_TEXT_LENGTH:
            logger.warning(
                f"Text too short for analysis (min: {self.MIN_TEXT_LENGTH}, got: {text_length})"
            )
            return False

        return True

    def _truncate_text(self, text: str) -> str:
        if len(text) <= self.MAX_TEXT_LENGTH:
            return text

        logger.debug(
            f"Truncating text for LLM: {len(text)} → {self.MAX_TEXT_LENGTH} chars"
        )
        return text[: self.MAX_TEXT_LENGTH]

    @staticmethod
    def _is_valid_extraction(result: AppealFields) -> bool:
        return any(
            [
                result.fioApplicant,
                result.organizationName,
                result.shortSummary,
            ]
        )

    @classmethod
    def _calculate_retry_delay(cls, attempt: int) -> int:
        return cls.BASE_RETRY_DELAY**attempt

    def _build_extraction_prompt(self) -> ChatPromptTemplate:
        system_message = """Ты — эксперт-аналитик системы электронного документооборота (СЭД).
Твоя задача: проанализировать текст официального обращения и извлечь факты для заполнения регистрационной карточки.

═══════════════════════════════════════════════════════════════════════════════
ПРАВИЛА ИЗВЛЕЧЕНИЯ И КЛАССИФИКАЦИИ
═══════════════════════════════════════════════════════════════════════════════

1️⃣ **declarantType (Тип заявителя)** — КРИТИЧЕСКИ ВАЖНО:

   ✅ INDIVIDUAL (Физическое лицо):
   - Обращение написано от имени гражданина
   - Подпись в формате "ФИО" без должности
   - Личные вопросы (ЖКХ, пенсия, медицина)
   - Примеры: "Иванов И.И.", "гражданин Петров", "житель г. Минска"

   ✅ ENTITY (Юридическое лицо):
   - Обращение от организации, предприятия, госоргана
   - Указана должность подписавшего (Директор, Председатель)
   - Есть исходящий номер документа
   - Примеры: "ОАО 'Завод'", "Минздрав РБ", "Администрация района", "МБОО"

2️⃣ **Персональные данные**:

   - **fioApplicant**: ФИО заявителя
     * Для INDIVIDUAL: автор обращения
     * Для ENTITY: подписант (если указан)

   - **organizationName**: Полное/краткое название организации
     * ТОЛЬКО для ENTITY
     * Примеры: "МБОО 'Доброе дело'", "ООО 'Стройком'"

3️⃣ **Даты и Номера** (для ENTITY):

   - **correspondentOrgNumber**: Исходящий номер документа
     * Примеры: "№ 123 от 15.01.2025", "исх. № 45/2", "№ 01-01/26"
     * ВАЖНО: Извлекай ПОЛНОСТЬЮ "как есть" вместе с датой!

   - **dateDocCorrespondentOrg**: Дата исходящего номера
     * Формат ISO 8601: "2025-01-15T00:00:00Z"
     * Извлекается из:
       1. Явной даты в тексте: "26 марта 2018 года" → "2018-03-26T00:00:00Z"
       2. Даты в номере: "№ 123 от 15.01.2025" → "2025-01-15T00:00:00Z"
       3. Закодированной даты: "№ 01-01/26" → месяц=01, день=01, год=2026 → "2026-01-01T00:00:00Z"

   - **receiptDate**: Дата написания обращения
     * Для всех типов заявителей

4️⃣ **Логические признаки**:

   - **reasonably (Обоснованность)**: true, если есть:
     * Ссылки на законы/нормативные акты
     * Конкретные факты, даты, суммы
     * Расчеты или доказательства

   - **collective**: true, если подписано группой лиц
     * Примеры: "жители дома", "группа родителей", "Мы, родители детей"

   - **anonymous**: true, если автор скрыт
     * Нет подписи или "Аноним"

5️⃣ **Контактная информация**:

   - Извлекай fullAddress, phone, email, если явно указаны
   - Проверяй формат email (должен содержать '@')

6️⃣ **Краткое содержание (shortSummary)**:

   - Сформулируй СУТЬ обращения (до 200 символов)
   - Примеры:
     * "Обращение о праве детей с аутистическими нарушениями на образование"
     * "Жалоба на некачественное оказание услуг ЖКХ"

7️⃣ **Пересылка (correspondentAppeal)**:

   - Заполняй, если обращение ПЕРЕСЛАНО из другого органа
   - Ищи паттерны:
     * "Копии:" / "Копия:" + название органа
     * "Переслано из" + название органа
     * Явное указание органа-отправителя в шапке документа
   - Извлекай ПОЛНОЕ название органа
   - Примеры:
     * "Копии: Министерство труда и социальной защиты Республики Беларусь" → "Министерство труда и социальной защиты Республики Беларусь"
     * "Администрация Президента РБ" → "Администрация Президента РБ"
     * "Минздрав РБ" → "Минздрав РБ"
   - Если в "Копии:" указано НЕСКОЛЬКО органов → выбирай ПЕРВЫЙ (главный получатель)

8️⃣ **citizenType (Вид обращения)** — КРИТИЧЕСКИ ВАЖНО:

   ✅ "Жалоба" если: "жалоб", "претензи", "недовольн", "нарушен"
   ✅ "Заявление" если: "заявлен", "прошу", "просьба", "рассмотреть"
   ✅ "Предложение" если: "предлага", "инициатив", "улучшить"
   ✅ "Запрос" если: "запрос информаци", "просим предоставить"
   ✅ "Благодарность" если: "благодар", "спасибо"

9️⃣ **География (country, regionName, districtName, cityName)** — КРИТИЧЕСКИ ВАЖНО:

   ⚠️ ИЗМЕНЕННАЯ СТРАТЕГИЯ:
   
   **cityName** (ВЫСШИЙ ПРИОРИТЕТ):
   - Извлекай ВСЕГДА из любого упоминания населенного пункта
   - Ищи в:
     1. Явном указании: "г. Минск", "город Гомель"
     2. Адресе: "220004, Минск, ул. Ленина" → "Минск"
     3. Индексе + город: "220004, Минск" → "Минск"
     4. После улицы: "ул. Ленина, Минск" → "Минск"
   - Убирай префиксы: "г. Минск" → "Минск"
   - ПРИМЕРЫ:
     * "пр. Победителей, 23, к.2, 220004, Минск" → cityName="Минск"
     * "г. Молодечно, ул. Ленина 5" → cityName="Молодечно"

   **regionName** (область):
   - Извлекай ТОЛЬКО если ЯВНО упомянута:
     * "Минская область" → regionName="Минская область"
     * "Гомельская обл." → regionName="Гомельская область"
   - ❌ Если написано только город без области → regionName=null
   - Система сама определит область по городу через справочник!

   **districtName** (район):
   - Извлекай ТОЛЬКО если ЯВНО упомянут:
     * "Октябрьский район" → districtName="Октябрьский район"
     * "Молодечненский р-н" → districtName="Молодечненский р-н"
   - ❌ Если написано только город → districtName=null
   - Система сама определит район через справочник!

   **country** (страна):
   - Извлекай ТОЛЬКО если явно упомянута:
     * "Беларусь", "Россия", "Республика Беларусь"

   ПРИМЕРЫ ПРАВИЛЬНОГО ИЗВЛЕЧЕНИЯ:

   📝 Текст: "пр. Победителей, 23, к.2, 220004, Минск"
   → cityName="Минск", regionName=null, districtName=null, country=null
   (Система автоматически определит "Минская область" и "г. Минск" через API)

   📝 Текст: "г. Молодечно, Минская область"
   → cityName="Молодечно", regionName="Минская область", districtName=null

   📝 Текст: "деревня Малые Ляды, Молодечненский район, Минская область"
   → cityName="деревня Малые Ляды", regionName="Минская область", districtName="Молодечненский район"

═══════════════════════════════════════════════════════════════════════════════
ВАЖНО
═══════════════════════════════════════════════════════════════════════════════

❗ Если информации по полю НЕТ — оставляй **null** (НЕ строку "None", "N/A")
❗ НЕ ПРИДУМЫВАЙ данные — извлекай ТОЛЬКО то, что явно написано
❗ Ответ ТОЛЬКО в формате JSON (без текста, комментариев, markdown)
❗ Все даты в формате ISO 8601 с суффиксом Z
❗ Для ENTITY обязательно пытайся извлечь дату из correspondentOrgNumber!

ПРИМЕРЫ ПРАВИЛЬНОГО ЗАПОЛНЕНИЯ:
- "fioApplicant": null (НЕ "None")
- "cityName": "Минск" (извлечено из адреса "220004, Минск")
- "regionName": null (будет определено системой)
- "dateDocCorrespondentOrg": "2026-01-01T00:00:00Z" (извлечено из "№ 01-01/26")
- "correspondentOrgNumber": "№ 01-01/26" (полностью с датой)
"""

        user_message = """Текст обращения для анализа:
───────────────────────────────────────────────────────────────────────────────
{text}
───────────────────────────────────────────────────────────────────────────────

Инструкции по формату JSON-ответа:
{format_instructions}
"""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("user", user_message),
            ]
        )

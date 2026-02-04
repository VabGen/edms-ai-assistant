# edms_ai_assistant/services/appeal_extraction_service.py
"""
Сервис для извлечения структурированных данных из неструктурированного текста обращений
с использованием LLM (Large Language Models).
"""
import logging
import asyncio
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.models.appeal_fields import AppealFields

logger = logging.getLogger(__name__)


class AppealExtractionService:

    def __init__(self):
        base_llm = get_chat_model()
        self.extraction_llm = base_llm.with_config({"temperature": 0.0})

    async def extract_appeal_fields(self, text: str) -> AppealFields:
        if not text or len(text.strip()) < 30:
            logger.warning(
                "Текст обращения слишком короткий для анализа "
                f"(минимум 30 символов, получено: {len(text.strip()) if text else 0})"
            )
            return AppealFields()

        try:
            parser = JsonOutputParser(pydantic_object=AppealFields)
            prompt = self._build_extraction_prompt()

            chain = prompt | self.extraction_llm | parser

            truncated_text = text[:12000]
            if len(text) > 12000:
                logger.debug(f"Текст обрезан для LLM: {len(text)} → 12000 символов")

            result = await chain.ainvoke(
                {
                    "text": truncated_text,
                    "format_instructions": parser.get_format_instructions(),
                }
            )

            appeal_data = AppealFields.model_validate(result)

            # logger.info(f" Данные успешно извлечены: {appeal_data}")

            return appeal_data

        except Exception as e:
            logger.error(
                f" Критическая ошибка при LLM-анализе: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return AppealFields()

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
         * Примеры: "№ 123 от 15.01.2025", "исх. № 45/2", "№ 01"

       - **dateDocCorrespondentOrg**: Дата исходящего номера
         * Формат ISO 8601: "2025-01-15T00:00:00Z"
         * Извлекается из текста вида "26 марта 2018 года"

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
         * "Запрос информации о ходе строительства школы"

    7️⃣ **Пересылка (correspondentAppeal)**:

       - Заполняй, если обращение ПЕРЕСЛАНО из другого органа
       - Примеры: "Администрация Президента", "Минздрав РБ"

    8️⃣ **citizenType (Вид обращения)** — КРИТИЧЕСКИ ВАЖНО:

       Определяется по ключевым словам в тексте:

       ✅ "Жалоба" если:
       - Слова: "жалоб", "претензи", "недовольн", "complaint"
       - Негативный тон: "нарушен", "незаконн", "ущерб", "бездействи"

       ✅ "Заявление" если:
       - Слова: "заявлен", "прошу", "просьба", "обращаемся с просьбой"
       - Конструктивный тон: "рассмотреть", "принять меры"

       ✅ "Предложение" если:
       - Слова: "предлага", "инициатив", "proposal"
       - Позитивный тон: "улучшить", "внедрить"

       ✅ "Запрос" если:
       - Слова: "запрос информаци", "просим предоставить", "разъяснить"
       - Информационная направленность

       ✅ "Благодарность" если:
       - Слова: "благодар", "спасибо", "признательн"

    9️⃣ **География (country, regionName, districtName, cityName)** — ВАЖНО:

       ПРАВИЛО: Извлекай ТОЛЬКО ТО, ЧТО ЯВНО НАПИСАНО В ТЕКСТЕ!

       - **country**: Страна
         * Примеры: "Беларусь", "Россия", "Республика Беларусь"

       - **regionName**: Область/Регион
         * Извлекай ТОЛЬКО если ЯВНО упомянута в тексте
         * Примеры: "Минская область", "Гомельская обл."
         * ❌ Если написано только "г. Минск" → ты ОБЯЗАН определить область (regionName) на основании своих знаний.
         * Примеры:
            - "г. Молодечно" -> cityName="Молодечно", regionName="Минская область"
            - "г. Гомель" -> cityName="Гомель", regionName="Гомельская область"

       - **districtName**: Район
         * Извлекай ТОЛЬКО если ЯВНО упомянут в тексте
         * Примеры: "Октябрьский район", "Первомайский р-н"
         * ❌ Если написано только "г. Минск" → ты ОБЯЗАН определить район (districtName) на основании своих знаний.
         * Примеры:
            - "г. Молодечно" -> cityName="Молодечно", districtName="Молодечненский район"
            - "г. Гомель" -> cityName="Гомель", districtName=null (областной центр)

       - **cityName**: Город/Населенный пункт
         * Примеры: "Минск", "г. Гомель", "Брест"

       ⚠️ КРИТИЧЕСКИ ВАЖНО:
       - Если в тексте: "г. Минск, Минская область" → country="Беларусь", cityName="Минск", regionName="Минская область"
       - Если в тексте: "г. Минск" → country="Беларусь", cityName="Минск", regionName=null, districtName=null
       - НЕ ПЫТАЙСЯ определить область/район по городу! Справочник это сделает сам.

    ═══════════════════════════════════════════════════════════════════════════════
    ВАЖНО
    ═══════════════════════════════════════════════════════════════════════════════

    ❗ Если информации по полю НЕТ — оставляй **null** (НЕ строку "None", "N/A" или "Unknown")
    ❗ НЕ ПРИДУМЫВАЙ данные
    ❗ НЕ ВЫВОДИ информацию логически (например, область из города)
    ❗ Ответ ТОЛЬКО в формате JSON (без текста, комментариев, markdown)
    ❗ Все даты должны строго соответствовать формату ISO 8601 с указанием времени и миллисекунд (UTC).
    ❗ Если поле пустое/неизвестно → используй null, а НЕ "None", "Unknown", "N/A", "Неизвестно"

    ПРИМЕРЫ ПРАВИЛЬНОГО ЗАПОЛНЕНИЯ:
    - "fioApplicant": null (НЕ "None")
    - "regionName": null (если не указана в тексте явно)
    - "organizationName": "МБОО \\"Доброе дело. Помощь людям с аутизмом\\""
    - "receiptDate, dateDocCorrespondentOrg: 26 марта 2018 года" -> "2018-03-26T00:00:00.000000Z"

    ПРИМЕРЫ НЕПРАВИЛЬНОГО ЗАПОЛНЕНИЯ (НЕ ДЕЛАЙ ТАК!):
    - "fioApplicant": "None" (НЕПРАВИЛЬНО!)
    - "regionName": "Минская область" (если в тексте НЕТ упоминания - НЕПРАВИЛЬНО!)
    - "email": "N/A" (НЕПРАВИЛЬНО!)
    """

        user_message = """Текст обращения для анализа:
    ───────────────────────────────────────────────────────────────────────────────
    {text}
    ───────────────────────────────────────────────────────────────────────────────

    Инструкции по формату JSON-ответа:
    {format_instructions}
    """

        return ChatPromptTemplate.from_messages(
            [("system", system_message), ("user", user_message)]
        )

    async def extract_with_retry(
        self, text: str, max_attempts: int = 3
    ) -> AppealFields:
        for attempt in range(1, max_attempts + 1):
            try:
                result = await self.extract_appeal_fields(text)

                if any(
                    [result.fioApplicant, result.organizationName, result.shortSummary]
                ):
                    # logger.info(f" Данные успешно извлечены на попытке {attempt}/{max_attempts}")
                    return result

                logger.warning(
                    f" Попытка {attempt}/{max_attempts}: "
                    "LLM не нашла значимой информации в тексте"
                )

            except Exception as e:
                logger.error(
                    f" Попытка {attempt}/{max_attempts} завершилась ошибкой: "
                    f"{type(e).__name__}: {e}"
                )

            if attempt < max_attempts:
                wait_time = 2**attempt
                logger.info(f"⏳ Ожидание {wait_time}с перед следующей попыткой...")
                await asyncio.sleep(wait_time)

        logger.error(
            f" Не удалось извлечь данные после {max_attempts} попыток. "
            "Возвращаем пустой объект."
        )
        return AppealFields()

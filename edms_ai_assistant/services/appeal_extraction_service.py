# edms_ai_assistant/services/appeal_extraction_service.py
"""
Сервис для извлечения структурированных данных из неструктурированного текста обращений
с использованием LLM (Large Language Models).

Применяется в инструменте автоматического заполнения карточек обращений граждан.
"""
import logging
import asyncio
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.models.appeal_fields import AppealFields

logger = logging.getLogger(__name__)


class AppealExtractionService:
    """
    Сервис для анализа текста обращений с помощью LLM и извлечения структурированных данных.

    Использует специализированный промпт, адаптированный под требования СЭД,
    для точной классификации типа заявителя и извлечения релевантной информации.

    Attributes:
        extraction_llm: Настроенная LLM-модель с температурой 0.0 для детерминированности
    """

    def __init__(self):
        """
        Инициализирует сервис с базовой LLM-моделью.

        Температура установлена в 0.0 для максимальной фактологической точности
        и воспроизводимости результатов.
        """
        base_llm = get_chat_model()
        self.extraction_llm = base_llm.with_config({"temperature": 0.0})

    async def extract_appeal_fields(self, text: str) -> AppealFields:
        """
        Анализирует неструктурированный текст обращения и возвращает структурированные данные.

        Метод выполняет:
        1. Валидацию длины входного текста
        2. Формирование промпта с инструкциями для LLM
        3. Запрос к LLM через LangChain
        4. Парсинг JSON-ответа в объект AppealFields
        5. Валидацию и очистку извлеченных данных

        Args:
            text: Неструктурированный текст обращения (письмо, заявление, жалоба)

        Returns:
            AppealFields: Объект с извлеченными данными. При ошибке возвращает пустой объект.

        Examples:
             service = AppealExtractionService()
             text = "Прошу рассмотреть мое заявление... Иванов И.И."
             fields = await service.extract_appeal_fields(text)
             print(fields.fioApplicant)
             "Иванов И.И."
        """
        if not text or len(text.strip()) < 30:
            logger.warning(
                "Текст обращения слишком короткий для анализа "
                f"(минимум 30 символов, получено: {len(text.strip()) if text else 0})"
            )
            return AppealFields()

        logger.info(
            f"Запуск LLM-анализа текста обращения (длина: {len(text)} символов)"
        )

        try:
            parser = JsonOutputParser(pydantic_object=AppealFields)
            prompt = self._build_extraction_prompt()

            chain = prompt | self.extraction_llm | parser

            truncated_text = text[:12000]
            if len(text) > 12000:
                logger.debug(f"Текст обрезан для LLM: {len(text)} → 12000 символов")

            # Выполнение запроса к LLM
            result = await chain.ainvoke(
                {
                    "text": truncated_text,
                    "format_instructions": parser.get_format_instructions(),
                }
            )

            # Валидация и создание объекта AppealFields
            appeal_data = AppealFields.model_validate(result)

            logger.info(
                f"✅ Данные успешно извлечены. "
                f"Тип заявителя: {appeal_data.declarantType or 'не определен'}. "
                f"ФИО/Организация: {appeal_data.fioApplicant or appeal_data.organizationName or 'н/д'}"
            )

            return appeal_data

        except Exception as e:
            logger.error(
                f"❌ Критическая ошибка при LLM-анализе: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return AppealFields()

    def _build_extraction_prompt(self) -> ChatPromptTemplate:
        """
        Формирует промпт с детальными инструкциями для LLM.

        Промпт содержит:
        - Описание роли LLM (эксперт СЭД)
        - Правила классификации типа заявителя (INDIVIDUAL vs ENTITY)
        - Инструкции по извлечению персональных данных
        - Правила определения логических признаков (обоснованность, коллективность)
        - Требования к формату ответа (только JSON, без текста)

        Returns:
            ChatPromptTemplate: Готовый промпт для использования в LangChain цепочке
        """
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
   - Примеры: "ОАО 'Завод'", "Минздрав РБ", "Администрация района"

2️⃣ **Персональные данные**:

   - **fioApplicant**: ФИО заявителя
     * Для INDIVIDUAL: автор обращения
     * Для ENTITY: подписант (если указан)

   - **organizationName**: Полное/краткое название организации
     * ТОЛЬКО для ENTITY
     * Примеры: "МБОО 'Доброе дело'", "ООО 'Стройком'"

3️⃣ **Даты и Номера** (для ENTITY):

   - **correspondentOrgNumber**: Исходящий номер документа
     * Примеры: "№ 123 от 15.01.2025", "исх. № 45/2"

   - **dateDocCorrespondentOrg**: Дата исходящего номера
     * Формат ISO 8601: "2025-01-15T00:00:00Z"

   - **receiptDate**: Дата написания обращения
     * Для всех типов заявителей

4️⃣ **Логические признаки**:

   - **reasonably (Обоснованность)**: true, если есть:
     * Ссылки на законы/нормативные акты
     * Конкретные факты, даты, суммы
     * Расчеты или доказательства

   - **collective**: true, если подписано группой лиц
     * Примеры: "жители дома", "группа родителей"

   - **anonymous**: true, если автор скрыт
     * Нет подписи или "Аноним"

5️⃣ **Контактная информация**:

   - Извлекай fullAddress, phone, email, если явно указаны
   - Проверяй формат email (должен содержать '@')

6️⃣ **Краткое содержание (shortSummary)**:

   - Сформулируй СУТЬ обращения (до 200 символов)
   - Примеры:
     * "Жалоба на некачественное оказание услуг ЖКХ по адресу..."
     * "Запрос информации о ходе строительства школы"
     * "Предложение по благоустройству придомовой территории"

7️⃣ **Пересылка (correspondentAppeal)**:

   - Заполняй, если обращение ПЕРЕСЛАНО из другого органа
   - Примеры: "Администрация Президента", "Минздрав РБ"

═══════════════════════════════════════════════════════════════════════════════
ВАЖНО
═══════════════════════════════════════════════════════════════════════════════

❗ Если информации по полю НЕТ — оставляй null
❗ НЕ ПРИДУМЫВАЙ данные
❗ Ответ ТОЛЬКО в формате JSON (без текста, комментариев, markdown)
❗ Все даты в формате ISO 8601
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
        """
        Метод-обертка с логикой повторных попыток при сбоях.

        Используется для повышения надежности при нестабильном соединении с LLM API
        или при получении пустых результатов.

        Args:
            text: Текст обращения для анализа
            max_attempts: Максимальное количество попыток (по умолчанию 3)

        Returns:
            AppealFields: Извлеченные данные или пустой объект после всех попыток

        Note:
            Использует экспоненциальную задержку между попытками: 2^attempt секунд
        """
        for attempt in range(1, max_attempts + 1):
            try:
                result = await self.extract_appeal_fields(text)

                if any(
                    [result.fioApplicant, result.organizationName, result.shortSummary]
                ):
                    logger.info(
                        f"✅ Данные успешно извлечены на попытке {attempt}/{max_attempts}"
                    )
                    return result

                logger.warning(
                    f"⚠️ Попытка {attempt}/{max_attempts}: "
                    "LLM не нашла значимой информации в тексте"
                )

            except Exception as e:
                logger.error(
                    f"❌ Попытка {attempt}/{max_attempts} завершилась ошибкой: "
                    f"{type(e).__name__}: {e}"
                )

            if attempt < max_attempts:
                wait_time = 2**attempt
                logger.info(f"⏳ Ожидание {wait_time}с перед следующей попыткой...")
                await asyncio.sleep(wait_time)

        logger.error(
            f"❌ Не удалось извлечь данные после {max_attempts} попыток. "
            "Возвращаем пустой объект."
        )
        return AppealFields()

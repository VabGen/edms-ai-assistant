# edms_ai_assistant/services/appeal_extraction_service.py
import json
import logging
from typing import Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.models.appeal_fields import AppealFields

logger = logging.getLogger(__name__)


class AppealExtractionService:
    """
    Сервис для извлечения структурированных данных из текста обращения граждан
    с использованием LLM.
    """

    def __init__(self):
        self.llm = get_chat_model()
        # Понижаем температуру для более консистентных результатов
        self.extraction_llm = self.llm.copy(update={"temperature": 0.1})

    async def extract_appeal_fields(self, text: str) -> AppealFields:
        """
        Извлекает структурированные данные обращения из неструктурированного текста.

        Args:
            text: Текст обращения (из .docx, .pdf или .txt файла)

        Returns:
            AppealFields с заполненными полями

        Raises:
            ValueError: Если LLM не смог извлечь данные или вернул некорректный JSON
        """
        if not text or len(text.strip()) < 50:
            logger.warning("Текст обращения слишком короткий для анализа")
            return AppealFields()

        logger.info(f"Начало извлечения данных из текста длиной {len(text)} символов")

        try:
            # промпт для извлечения данных
            extraction_prompt = self._build_extraction_prompt()

            parser = JsonOutputParser(pydantic_object=AppealFields)

            chain = extraction_prompt | self.extraction_llm | parser

            result = await chain.ainvoke({
                "text": text[:8000],
                "format_instructions": parser.get_format_instructions()
            })

            appeal_fields = AppealFields.model_validate(result)

            logger.info(
                f"Успешно извлечены данные обращения. "
                f"Заявитель: {appeal_fields.fioApplicant or 'Не указан'}, "
                f"Тип: {appeal_fields.citizenType or 'Не определен'}"
            )

            return appeal_fields

        except json.JSONDecodeError as e:
            logger.error(f"LLM вернула некорректный JSON: {e}")
            raise ValueError(f"Не удалось распарсить ответ LLM: {e}")

        except Exception as e:
            logger.error(f"Ошибка извлечения данных из обращения: {e}", exc_info=True)
            raise ValueError(f"Ошибка анализа текста обращения: {str(e)}")

    def _build_extraction_prompt(self) -> ChatPromptTemplate:
        """ промпт для извлечения данных из текста обращения"""

        system_message = """Ты — эксперт по анализу обращений граждан в государственные органы.

Твоя задача: извлечь структурированные данные из текста обращения и вернуть их в формате JSON.

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:

1. **Формат данных**:
   - Верни ТОЛЬКО валидный JSON без дополнительного текста
   - Используй точно такие же ключи, как в схеме
   - Если информация не найдена → используй null

2. **Даты**:
   - Формат: ISO 8601 (YYYY-MM-DDTHH:MM:SS)
   - Примеры: "2024-03-15T10:30:00", "2024-12-01T00:00:00"

3. **ФИО**:
   - Формат: "Фамилия Имя Отчество"
   - Пример: "Иванов Иван Иванович"

4. **Адреса**:
   - Указывай полностью, без сокращений
   - Пример: "улица Ленина, дом 25, квартира 10"

5. **Тип заявителя (declarantType)**:
   - "PHYSICAL" — для физических лиц
   - "LEGAL" — для юридических лиц (организаций)

6. **Виды обращений (citizenType)**:
   - Жалоба, Заявление, Предложение, Благодарность, Запрос

7. **Boolean-поля**:
   - Коллективное (collective): true если от нескольких лиц
   - Анонимное (anonymous): true если автор не указан
   - Обоснованное (reasonably): true если есть факты/доказательства

8. **Краткое содержание (shortSummary)**:
   - Максимум 200 символов
   - Суть проблемы одним предложением

ПРИМЕРЫ ОБРАЩЕНИЙ:

**Пример 1 (жалоба физического лица):**
```
От: Петрова Мария Сергеевна
Адрес: г. Минск, ул. Победы, д. 15, кв. 42
Индекс: 220030
Телефон: +375291234567
Email: petrov@mail.ru

Прошу рассмотреть жалобу на плохое состояние дорожного покрытия во дворе дома.
Дата подачи: 15.03.2024
```

**Извлеченные данные:**
```json
{{
  "fioApplicant": "Петрова Мария Сергеевна",
  "cityName": "Минск",
  "fullAddress": "улица Победы, дом 15, квартира 42",
  "index": "220030",
  "phone": "+375291234567",
  "email": "petrov@mail.ru",
  "shortSummary": "Жалоба на плохое состояние дорожного покрытия во дворе",
  "citizenType": "Жалоба",
  "receiptDate": "2024-03-15T00:00:00",
  "declarantType": "PHYSICAL",
  "collective": false,
  "anonymous": false
}}
```

**Пример 2 (заявление от организации):**
```
ООО "Строй-Мастер"
220015, г. Минск, пр. Независимости, 100
ИНН: 1234567890
Генеральный директор: Сидоров А.В.

Заявление о согласовании проектной документации
Дата: 01.12.2024
```

**Извлеченные данные:**
```json
{{
  "organizationName": "ООО Строй-Мастер",
  "signed": "Сидоров А.В.",
  "cityName": "Минск",
  "fullAddress": "проспект Независимости, дом 100",
  "index": "220015",
  "shortSummary": "Заявление о согласовании проектной документации",
  "citizenType": "Заявление",
  "receiptDate": "2024-12-01T00:00:00",
  "declarantType": "LEGAL",
  "collective": false,
  "anonymous": false
}}
```

ВАЖНО:
- Анализируй весь текст внимательно
- Ищи неявные указания (например, "мы, жители дома..." → collective: true)
- Если сомневаешься → лучше null, чем неверные данные
- Верни строго JSON без markdown-блоков и комментариев
"""

        user_message = """Извлеки данные из следующего текста обращения:

{text}

Схема данных:
{format_instructions}

Верни ТОЛЬКО JSON:"""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", user_message)
        ])

    async def extract_with_retry(
        self,
        text: str,
        max_attempts: int = 3
    ) -> Optional[AppealFields]:
        """
        Извлекает данные с повторными попытками при ошибках.

        Args:
            text: Текст обращения
            max_attempts: Максимальное количество попыток

        Returns:
            AppealFields или None при неудаче
        """
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Попытка извлечения данных #{attempt}/{max_attempts}")
                return await self.extract_appeal_fields(text)

            except Exception as e:
                logger.warning(f"Попытка #{attempt} неудачна: {e}")

                if attempt == max_attempts:
                    logger.error(
                        f"Все {max_attempts} попытки исчерпаны. "
                        f"Не удалось извлечь данные."
                    )
                    return None

                # задержка перед следующей попыткой
                import asyncio
                await asyncio.sleep(1)

        return None

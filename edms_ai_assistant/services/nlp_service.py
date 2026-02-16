# edms_ai_assistant/services/nlp_service.py
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter

logger = logging.getLogger(__name__)


# ========================================
# ENUMS & DATA CLASSES
# ========================================


class UserIntent(Enum):
    """Типы намерений пользователя."""

    SUMMARIZE = "summarize"
    COMPARE = "compare"
    SEARCH = "search"
    CREATE_INTRODUCTION = "create_introduction"
    CREATE_TASK = "create_task"
    QUESTION = "question"
    ANALYZE = "analyze"
    EXTRACT = "extract"
    UPDATE = "update"
    DELETE = "delete"
    COMPOSITE = "composite"
    UNKNOWN = "unknown"


class QueryComplexity(Enum):
    """Сложность запроса."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class EntityType(Enum):
    """Типы извлекаемых сущностей."""

    DATE = "date"
    DATETIME = "datetime"
    PERSON = "person"
    NUMBER = "number"
    MONEY = "money"
    DOCUMENT_ID = "document_id"
    DEPARTMENT = "department"
    DURATION = "duration"


@dataclass
class Entity:
    """Структура именованной сущности."""

    type: EntityType
    value: Any
    raw_text: str
    confidence: float = 1.0
    normalized_value: Optional[Any] = None


@dataclass
class UserQuery:
    """Структура пользовательского запроса."""

    original: str
    refined: str
    intent: UserIntent
    secondary_intents: List[UserIntent] = field(default_factory=list)
    complexity: QueryComplexity = QueryComplexity.SIMPLE
    entities: Dict[str, List[Entity]] = field(default_factory=dict)
    keywords: Set[str] = field(default_factory=set)
    confidence: float = 1.0


@dataclass
class SemanticContext:
    """Семантический контекст для обработки запроса."""

    query: UserQuery
    document: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ========================================
# ENTITY EXTRACTOR
# ========================================


class EntityExtractor:
    """
    экстрактор именованных сущностей.

    Функции:
    - Извлечение дат с нормализацией
    - Распознавание ФИО с контекстом
    - Парсинг денежных сумм
    - Извлечение UUID документов
    - Определение временных промежутков
    """

    DATE_PATTERNS = [
        # DD.MM.YYYY
        (
            r"(\d{1,2})\.(\d{1,2})\.(\d{4})",
            lambda m: f"{m[2]}-{int(m[1]):02d}-{int(m[0]):02d}",
        ),
        # DD месяц YYYY
        (
            r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)(?:\s+(\d{4}))?",
            "month_name",
        ),
        # "сегодня", "завтра", "вчера"
        (r"\b(сегодня|завтра|вчера|послезавтра)\b", "relative_day"),
        # "через N дней/недель"
        (
            r"через\s+(\d+)\s+(день|дня|дней|неделю|недели|недель|месяц|месяца|месяцев)",
            "duration",
        ),
        # "до DD.MM"
        (r"до\s+(\d{1,2})\.(\d{1,2})", "deadline"),
    ]

    MONTH_NAMES = {
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

    def extract_dates(
        self, text: str, base_date: Optional[datetime] = None
    ) -> List[Entity]:
        """
        Извлекает и нормализует даты из текста.

        Args:
            text: Исходный текст
            base_date: Базовая дата для относительных выражений

        Returns:
            Список извлеченных дат
        """
        if base_date is None:
            base_date = datetime.now()

        dates = []

        # Обработка паттернов
        for pattern, handler in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                raw = match.group(0)

                try:
                    if handler == "month_name":
                        day = int(match.group(1))
                        month = self.MONTH_NAMES[match.group(2)]
                        year = int(match.group(3)) if match.group(3) else base_date.year
                        normalized = datetime(year, month, day)

                    elif handler == "relative_day":
                        delta_map = {
                            "сегодня": 0,
                            "завтра": 1,
                            "послезавтра": 2,
                            "вчера": -1,
                        }
                        delta = delta_map[match.group(1)]
                        normalized = base_date + timedelta(days=delta)

                    elif handler == "duration":
                        count = int(match.group(1))
                        unit = match.group(2)

                        if "день" in unit or "дня" in unit or "дней" in unit:
                            delta = timedelta(days=count)
                        elif "недел" in unit:
                            delta = timedelta(weeks=count)
                        elif "месяц" in unit:
                            delta = timedelta(days=count * 30)
                        else:
                            continue

                        normalized = base_date + delta

                    elif handler == "deadline":
                        day = int(match.group(1))
                        month = int(match.group(2))
                        year = base_date.year
                        normalized = datetime(year, month, day)

                        if normalized < base_date:
                            normalized = datetime(year + 1, month, day)

                    elif handler == "deadline":
                        try:
                            day = int(match.group(1))
                            month = int(match.group(2))
                            year = base_date.year

                            if month < 1 or month > 12 or day < 1 or day > 31:
                                logger.debug(f"Invalid date: {day}.{month}")
                                continue
                            normalized = datetime(year, month, day)

                            if normalized < base_date:
                                normalized = datetime(year + 1, month, day)
                        except ValueError as e:
                            logger.debug(f"Failed to parse deadline date: {e}")
                            continue

                    elif callable(handler):
                        iso_date = handler(match.groups())
                        normalized = datetime.fromisoformat(iso_date)

                    else:
                        continue

                    dates.append(
                        Entity(
                            type=EntityType.DATE,
                            value=normalized,
                            raw_text=raw,
                            normalized_value=normalized.isoformat(),
                        )
                    )

                except (ValueError, KeyError) as e:
                    logger.debug(f"Failed to parse date '{raw}': {e}")
                    continue

        return dates

    def extract_persons(self, text: str) -> List[Entity]:
        """
        Извлекает ФИО из текста.

        Args:
            text: Исходный текст

        Returns:
            Список извлеченных ФИО
        """
        persons = []

        # Паттерн: Фамилия Имя Отчество
        pattern = r"\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)(?:\s+([А-ЯЁ][а-яё]+))?\b"

        for match in re.finditer(pattern, text):
            full_match = match.group(0)
            last_name = match.group(1)
            first_name = match.group(2)
            middle_name = match.group(3) if match.group(3) else None

            # Исключаем служебные слова
            if last_name.lower() in ["через", "после", "перед", "около"]:
                continue

            normalized = {
                "lastName": last_name,
                "firstName": first_name,
                "middleName": middle_name,
            }

            persons.append(
                Entity(
                    type=EntityType.PERSON,
                    value=full_match,
                    raw_text=full_match,
                    normalized_value=normalized,
                    confidence=0.8 if middle_name else 0.6,
                )
            )

        return persons

    def extract_numbers(self, text: str) -> List[Entity]:
        """Извлекает числовые значения."""
        numbers = []

        # Целые числа и десятичные
        for match in re.finditer(r"\b(\d+(?:[.,]\d+)?)\b", text):
            raw = match.group(0)
            value = float(raw.replace(",", "."))

            numbers.append(
                Entity(
                    type=EntityType.NUMBER,
                    value=value,
                    raw_text=raw,
                    normalized_value=value,
                )
            )

        return numbers

    def extract_money(self, text: str) -> List[Entity]:
        """Извлекает денежные суммы с валютой."""
        money = []

        # Паттерны валют
        currency_patterns = [
            (r"(\d+(?:[.,]\d+)?)\s*(руб|₽|rub|бел\.руб)", "BYN"),
            (r"(\d+(?:[.,]\d+)?)\s*(\$|usd|долл)", "USD"),
            (r"(\d+(?:[.,]\d+)?)\s*(€|eur|евро)", "EUR"),
            (r"(\d+(?:[.,]\d+)?)\s*(руб|rub)", "RUB"),
        ]

        for pattern, currency in currency_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                raw = match.group(0)
                amount = float(match.group(1).replace(",", "."))

                money.append(
                    Entity(
                        type=EntityType.MONEY,
                        value=amount,
                        raw_text=raw,
                        normalized_value={"amount": amount, "currency": currency},
                    )
                )

        return money

    def extract_document_ids(self, text: str) -> List[Entity]:
        """Извлекает UUID документов."""
        doc_ids = []

        uuid_pattern = (
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
        )

        for match in re.finditer(uuid_pattern, text, re.IGNORECASE):
            doc_ids.append(
                Entity(
                    type=EntityType.DOCUMENT_ID,
                    value=match.group(0),
                    raw_text=match.group(0),
                    normalized_value=match.group(0).lower(),
                )
            )

        return doc_ids

    def extract_all(
        self, text: str, base_date: Optional[datetime] = None
    ) -> Dict[str, List[Entity]]:
        """
        Извлекает все типы сущностей из текста.

        Args:
            text: Исходный текст
            base_date: Базовая дата для относительных выражений

        Returns:
            Словарь сущностей по типам
        """
        entities = {}

        # Даты
        dates = self.extract_dates(text, base_date)
        if dates:
            entities["dates"] = dates

        # Персоны
        persons = self.extract_persons(text)
        if persons:
            entities["persons"] = persons

        # Числа
        numbers = self.extract_numbers(text)
        if numbers:
            entities["numbers"] = numbers

        # Деньги
        money = self.extract_money(text)
        if money:
            entities["money"] = money

        # UUID документов
        doc_ids = self.extract_document_ids(text)
        if doc_ids:
            entities["document_ids"] = doc_ids

        return entities


# ========================================
# QUERY REFINER
# ========================================


class QueryRefiner:
    """
    Улучшение и нормализация запросов.

    Функции:
    - Исправление опечаток
    - Расширение аббревиатур
    - Нормализация формулировок
    - Добавление контекста
    """

    # Словарь аббревиатур EDMS
    ABBREVIATIONS = {
        "док": "документ",
        "ознак": "ознакомление",
        "пор": "поручение",
        "исп": "исполнитель",
        "отв": "ответственный",
        "сов": "совещание",
        "дог": "договор",
        "сэд": "СЭД",
    }

    # Синонимы действий
    ACTION_SYNONYMS = {
        "покажи": "найди",
        "выведи": "найди",
        "дай": "найди",
        "скажи": "опиши",
        "расскажи": "опиши",
        "объясни": "опиши",
    }

    def expand_abbreviations(self, text: str) -> str:
        """Расширяет аббревиатуры до полных слов."""
        words = text.split()
        expanded = []

        for word in words:
            word_lower = word.lower()
            if word_lower in self.ABBREVIATIONS:
                expanded.append(self.ABBREVIATIONS[word_lower])
            else:
                expanded.append(word)

        return " ".join(expanded)

    def normalize_actions(self, text: str) -> str:
        """Нормализует глаголы действий."""
        text_lower = text.lower()

        for synonym, canonical in self.ACTION_SYNONYMS.items():
            text_lower = re.sub(r"\b" + synonym + r"\b", canonical, text_lower)

        return text_lower

    def add_context(
        self, text: str, intent: UserIntent, entities: Dict[str, List[Entity]]
    ) -> str:
        """
        Добавляет контекст в запрос на основе намерения и сущностей.

        Args:
            text: Исходный текст
            intent: Определенное намерение
            entities: Извлеченные сущности

        Returns:
            Обогащенный запрос
        """
        refined = text

        if intent == UserIntent.SEARCH and entities:
            if "persons" in entities:
                person = entities["persons"][0]
                refined = f"Найти документы, связанные с {person.value}"
            elif "dates" in entities:
                date = entities["dates"][0]
                refined = f"Найти документы за {date.raw_text}"

        # Для создания поручения: добавляем срок, если не указан
        elif intent == UserIntent.CREATE_TASK:
            if "dates" not in entities or not entities["dates"]:
                refined += " (срок: +7 дней)"

        # Для сравнения: уточняем, что сравнивать
        elif intent == UserIntent.COMPARE:
            if (
                "версия" in text.lower()
                and "document_ids" in entities
                and len(entities["document_ids"]) < 2
            ):
                refined += " (требуется указать версии для сравнения)"

        return refined

    def refine(
        self, text: str, intent: UserIntent, entities: Dict[str, List[Entity]]
    ) -> str:
        """
        Полное уточнение запроса.

        Args:
            text: Исходный текст
            intent: Определенное намерение
            entities: Извлеченные сущности

        Returns:
            Уточненный запрос
        """
        # 1. Базовая очистка
        refined = text.strip()

        # 2. Расширение аббревиатур
        refined = self.expand_abbreviations(refined)

        # 3. Нормализация действий
        refined = self.normalize_actions(refined)

        # 4. Добавление контекста
        refined = self.add_context(refined, intent, entities)

        # 5. Удаление лишних пробелов
        refined = re.sub(r"\s+", " ", refined).strip()

        return refined


# ========================================
# SEMANTIC DISPATCHER
# ========================================


class SemanticDispatcher:
    """
    Профессиональный семантический диспетчер запросов.

    Возможности:
    - Многоуровневая классификация намерений
    - Определение сложности с учетом контекста документа
    - Интеллектуальное извлечение сущностей
    - Улучшение формулировок запросов
    """

    INTENT_KEYWORDS = {
        UserIntent.CREATE_INTRODUCTION: {
            "primary": [
                "ознакомление",
                "ознакомь",
                "список ознакомления",
                "добавь в ознакомление",
            ],
            "secondary": ["виза", "визирование", "согласование"],
        },
        UserIntent.CREATE_TASK: {
            "primary": ["поручение", "создай задачу", "задание", "поручи"],
            "secondary": ["исполнитель", "срок исполнения", "контроль"],
        },
        UserIntent.SUMMARIZE: {
            "primary": ["суммаризуй", "кратко", "резюме", "опиши", "сводка"],
            "secondary": ["анализ", "выжимка", "основное"],
        },
        UserIntent.COMPARE: {
            "primary": ["сравни", "отличия", "разница", "версия"],
            "secondary": ["изменения", "что изменилось"],
        },
        UserIntent.SEARCH: {
            "primary": ["найди", "поиск", "покажи", "где"],
            "secondary": ["выведи", "список", "реестр"],
        },
        UserIntent.ANALYZE: {
            "primary": ["проанализируй", "подробно", "детали"],
            "secondary": ["разбор", "структура"],
        },
        UserIntent.QUESTION: {
            "primary": ["какая", "какой", "сколько", "когда", "почему"],
            "secondary": ["расскажи", "объясни", "что"],
        },
    }

    def __init__(self):
        """Инициализация диспетчера с подкомпонентами."""
        self.entity_extractor = EntityExtractor()
        self.query_refiner = QueryRefiner()
        logger.info("SemanticDispatcher initialized (Professional Edition)")

    def detect_intent(self, message: str) -> Tuple[UserIntent, List[UserIntent], float]:
        """
        Определяет основное и дополнительные намерения.

        Args:
            message: Сообщение пользователя

        Returns:
            (основное_намерение, дополнительные_намерения, уверенность)
        """
        message_lower = message.lower()
        scores = Counter()

        # Подсчет совпадений с ключевыми словами
        for intent, keywords in self.INTENT_KEYWORDS.items():
            primary_count = sum(1 for kw in keywords["primary"] if kw in message_lower)
            secondary_count = sum(
                1 for kw in keywords["secondary"] if kw in message_lower
            )

            # Первичные ключевые слова дают 2 балла, вторичные — 1
            scores[intent] = primary_count * 2 + secondary_count

        # Дополнительные эвристики
        if message.strip().endswith("?"):
            scores[UserIntent.QUESTION] += 1

        if not scores:
            return UserIntent.UNKNOWN, [], 0.0

        # Сортируем по убыванию score
        sorted_intents = scores.most_common()

        primary_intent = sorted_intents[0][0]
        primary_score = sorted_intents[0][1]

        # Определяем дополнительные намерения (score >= 50% от основного)
        threshold = primary_score * 0.5
        secondary_intents = [
            intent for intent, score in sorted_intents[1:] if score >= threshold
        ]

        # Уверенность: нормализуем от 0 до 1
        intent_keywords = self.INTENT_KEYWORDS.get(primary_intent, {})
        max_possible_score = len(intent_keywords.get("primary", [])) * 2 + len(
            intent_keywords.get("secondary", [])
        )
        confidence = min(primary_score / max(max_possible_score, 1), 1.0)

        # Если несколько намерений с высоким score — композитное
        if len(secondary_intents) > 1:
            return UserIntent.COMPOSITE, secondary_intents, confidence

        return primary_intent, secondary_intents, confidence

    def estimate_complexity(
        self, message: str, document: Optional[Any] = None
    ) -> QueryComplexity:
        """
        Оценивает сложность запроса с учетом документа.

        Args:
            message: Сообщение пользователя
            document: Документ (если доступен)

        Returns:
            Уровень сложности
        """
        word_count = len(message.split())
        has_conditions = any(
            word in message.lower() for word in ["если", "когда", "где", "как", "при"]
        )
        has_multiple_entities = (
            len(re.findall(r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\b", message)) > 2
        )

        # Базовая оценка
        complexity_score = 0

        if word_count > 20:
            complexity_score += 3
        elif word_count > 10:
            complexity_score += 2
        elif word_count > 5:
            complexity_score += 1

        if has_conditions:
            complexity_score += 2

        if has_multiple_entities:
            complexity_score += 1

        # Учет документа
        if document:
            # Безопасное получение атрибутов с обработкой None
            attachments = getattr(document, "attachmentDocument", None) or []
            tasks = getattr(document, "taskList", None) or []
            process = getattr(document, "process", None)

            # Проверяем сложность документа
            has_attachments = bool(attachments)
            has_tasks = bool(tasks)
            has_process = bool(process)
            is_contract = getattr(document, "docCategoryConstant", None) == "CONTRACT"

            if has_attachments and len(attachments) > 3:
                complexity_score += 1

            if has_tasks and len(tasks) > 5:
                complexity_score += 1

            if has_process and not getattr(process, "completed", True):
                complexity_score += 1

            if is_contract:
                complexity_score += 1

        # Маппинг score на уровни
        if complexity_score >= 8:
            return QueryComplexity.VERY_COMPLEX
        elif complexity_score >= 5:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 2:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.SIMPLE

    def extract_keywords(self, message: str) -> Set[str]:
        """Извлекает ключевые слова из запроса."""
        # Удаляем стоп-слова
        stop_words = {
            "а",
            "в",
            "и",
            "на",
            "с",
            "о",
            "к",
            "по",
            "для",
            "из",
            "у",
            "от",
            "до",
            "это",
            "как",
            "что",
            "где",
            "когда",
            "мне",
            "меня",
            "мой",
            "моя",
            "мое",
            "можно",
            "нужно",
            "надо",
            "пожалуйста",
        }

        words = re.findall(r"\b[а-яёА-ЯЁ]{3,}\b", message.lower())
        keywords = {word for word in words if word not in stop_words}

        return keywords

    def build_context(
        self, message: str, document: Optional[Any] = None
    ) -> SemanticContext:
        """
        Строит полный семантический контекст для запроса.

        Args:
            message: Сообщение пользователя
            document: Документ (если доступен)

        Returns:
            Семантический контекст с метаданными и предложениями
        """
        # 1. Классификация намерений
        primary_intent, secondary_intents, confidence = self.detect_intent(message)

        # 2. Извлечение сущностей
        entities_dict = self.entity_extractor.extract_all(message)

        # 3. Оценка сложности (с учетом документа!)
        complexity = self.estimate_complexity(message, document)

        # 4. Извлечение ключевых слов
        keywords = self.extract_keywords(message)

        # 5. Уточнение запроса (ИСПОЛЬЗУЕМ ВСЕ ПАРАМЕТРЫ!)
        refined_message = self.query_refiner.refine(
            message, primary_intent, entities_dict
        )

        # 6. Построение UserQuery
        query = UserQuery(
            original=message,
            refined=refined_message,
            intent=primary_intent,
            secondary_intents=secondary_intents,
            complexity=complexity,
            entities=entities_dict,
            keywords=keywords,
            confidence=confidence,
        )

        # 7. Метаданные
        metadata = {
            "word_count": len(message.split()),
            "char_count": len(message),
            "has_question_mark": message.strip().endswith("?"),
        }

        if document:
            attachments = getattr(document, "attachmentDocument", None) or []
            tasks = getattr(document, "taskList", None) or []

            metadata.update(
                {
                    "has_document": True,
                    "document_id": str(getattr(document, "id", "")),
                    "document_category": str(
                        getattr(document, "docCategoryConstant", "")
                    ),
                    "document_status": str(getattr(document, "status", "")),
                    "attachments_count": len(attachments),
                    "tasks_count": len(tasks),
                }
            )

        # 8. Предложения и предупреждения
        suggestions = []
        warnings = []

        # Проверка на отсутствие сущностей для определенных намерений
        if primary_intent == UserIntent.CREATE_TASK:
            if "dates" not in entities_dict:
                suggestions.append("Рекомендуется указать срок выполнения поручения")
            if "persons" not in entities_dict:
                warnings.append(
                    "Не указан исполнитель. Будет использован ответственный по умолчанию"
                )

        if primary_intent == UserIntent.COMPARE:
            if (
                "document_ids" not in entities_dict
                or len(entities_dict.get("document_ids", [])) < 2
            ):
                warnings.append(
                    "Для сравнения требуется указать два документа или версии"
                )

        # Проверка сложности
        if complexity == QueryComplexity.VERY_COMPLEX and confidence < 0.7:
            suggestions.append(
                "Запрос очень сложный. Рекомендуется разбить на несколько более простых"
            )

        # 9. Сборка контекста
        return SemanticContext(
            query=query,
            document=document,
            metadata=metadata,
            suggestions=suggestions,
            warnings=warnings,
        )


# ========================================
# EDMS NATURAL LANGUAGE SERVICE
# ========================================


class EDMSNaturalLanguageService:
    """
    Сервис для семантического анализа данных EDMS.
    """

    @staticmethod
    def format_user(user: Any) -> Optional[str]:
        """Универсальное форматирование UserInfoDto или EmployeeDto."""
        if not user:
            return None

        try:
            ln = getattr(user, "lastName", "") or ""
            fn = getattr(user, "firstName", "") or ""
            mn = getattr(user, "middleName", "") or ""
            post = getattr(user, "authorPost", "") or getattr(
                getattr(user, "post", None), "name", ""
            )
            name = f"{ln} {fn} {mn}".strip()
            return f"{name} ({post})" if post else name if name else None
        except Exception as e:
            logger.debug(f"Error formatting user: {e}")
            return None

    @staticmethod
    def format_date(instant: Any) -> Optional[str]:
        """Форматирование Instant в читаемую дату (DD.MM.YYYY)."""
        if not instant:
            return None

        try:
            if hasattr(instant, "strftime"):
                return instant.strftime("%d.%m.%Y")

            str_instant = str(instant)
            if len(str_instant) >= 10:
                parts = str_instant[:10].split("-")
                if len(parts) == 3:
                    return f"{parts[2]}.{parts[1]}.{parts[0]}"

            return str_instant[:10] if len(str_instant) >= 10 else str_instant
        except Exception as e:
            logger.debug(f"Error formatting date: {e}")
            return None

    @staticmethod
    def format_datetime(instant: Any) -> Optional[str]:
        """Форматирование Instant в дату и время (DD.MM.YYYY HH:MM)."""
        if not instant:
            return None

        try:
            if hasattr(instant, "strftime"):
                return instant.strftime("%d.%m.%Y %H:%M")

            str_instant = str(instant)
            if len(str_instant) >= 16:
                date_part = str_instant[:10].split("-")
                time_part = str_instant[11:16]
                if len(date_part) == 3:
                    return f"{date_part[2]}.{date_part[1]}.{date_part[0]} {time_part}"

            return str_instant[:16] if len(str_instant) >= 16 else str_instant
        except Exception as e:
            logger.debug(f"Error formatting datetime: {e}")
            return None

    def get_safe(self, obj: Any, path: str, default: Any = None) -> Any:
        """
        Безопасное извлечение вложенного значения по пути с обработкой всех edge cases.

        Args:
            obj: Объект для извлечения
            path: Путь к значению (разделенный точками)
            default: Значение по умолчанию

        Returns:
            Извлеченное значение или default
        """
        if obj is None:
            return default

        val = obj
        try:
            for part in path.split("."):
                if val is None:
                    return default

                if isinstance(val, dict):
                    val = val.get(part, default)
                else:
                    val = getattr(val, part, default)

            if hasattr(val, "value"):
                return val.value

            return val if val is not None else default

        except (AttributeError, KeyError, TypeError) as e:
            logger.debug(f"Error accessing path '{path}': {e}")
            return default

    def process_document(self, doc: Any) -> Dict[str, Any]:
        """
        Полный анализ документа.

        Args:
            doc: Объект DocumentDto

        Returns:
            Структурированный словарь с данными документа
        """
        if not doc:
            logger.warning("Attempted to process None document")
            return {}

        try:
            category = self.get_safe(doc, "docCategoryConstant")

            # ========== 1. БАЗОВАЯ ИДЕНТИФИКАЦИЯ ==========
            base_info = {
                "id": str(doc.id) if getattr(doc, "id", None) else None,
                "категория": category,
                "профиль": getattr(doc, "profileName", None),
                "тип_документа": self.get_safe(doc, "documentType.name"),
                "тип_создания": self.get_safe(doc, "createType"),
                "краткое_содержание": getattr(doc, "shortSummary", None),
                "полный_текст": getattr(doc, "summary", None),
                "примечание": getattr(doc, "note", None),
            }

            # ========== 2. РЕГИСТРАЦИЯ И МЕТАДАННЫЕ ==========
            registration = {
                "рег_номер": getattr(doc, "regNumber", None)
                or getattr(doc, "reservedRegNumber", None),
                "дата_регистрации": self.format_date(getattr(doc, "regDate", None)),
                "дата_создания": self.format_datetime(getattr(doc, "createDate", None)),
                "зарезервированный_номер": getattr(doc, "reservedRegNumber", None),
                "дата_резервирования": self.format_datetime(
                    getattr(doc, "reservedRegDate", None)
                ),
                "исходящий_номер": getattr(doc, "outRegNumber", None),
                "исходящая_дата": self.format_date(getattr(doc, "outRegDate", None)),
                "журнал_регистрации": (
                    {
                        "id": (
                            str(doc.journalId)
                            if getattr(doc, "journalId", None)
                            else None
                        ),
                        "номер": getattr(doc, "journalNumber", None),
                        "название": self.get_safe(doc, "registrationJournal.name"),
                    }
                    if (
                        getattr(doc, "journalId", None)
                        or getattr(doc, "journalNumber", None)
                    )
                    else None
                ),
                "формула_номера": getattr(doc, "formula", None),
                "пропуск_регистрации": getattr(doc, "skipRegistration", None),
            }

            # ========== 3. УЧАСТНИКИ ДОКУМЕНТООБОРОТА ==========
            participants = {
                "автор": self.format_user(getattr(doc, "author", None)),
                "инициатор": self.format_user(getattr(doc, "initiator", None)),
                "ответственный_исполнитель": self.format_user(
                    getattr(doc, "responsibleExecutor", None)
                ),
                "кем_подписан": self.format_user(getattr(doc, "whoSigned", None)),
                "подписанты": [
                    self.format_user(u)
                    for u in (getattr(doc, "whoAddressed", None) or [])
                    if self.format_user(u)
                ],
                "внутренние_подписанты": getattr(doc, "inDocSigners", None),
                "корреспондент": (
                    {
                        "название": getattr(doc, "correspondentName", None),
                        "id": (
                            str(doc.correspondentId)
                            if getattr(doc, "correspondentId", None)
                            else None
                        ),
                        "детали": self.get_safe(doc, "correspondent.name"),
                    }
                    if (
                        getattr(doc, "correspondentName", None)
                        or getattr(doc, "correspondentId", None)
                    )
                    else None
                ),
                "адресаты": (
                    [
                        {
                            "название": self.get_safe(r, "name"),
                            "тип": self.get_safe(r, "recipientType"),
                        }
                        for r in (getattr(doc, "recipientList", None) or [])
                    ]
                    if getattr(doc, "recipientList", None)
                    else None
                ),
                "есть_адресаты": getattr(doc, "recipients", None),
                "ответственные_за_подготовку": (
                    [
                        {
                            "исполнитель": self.format_user(
                                self.get_safe(re, "responsibleExecutor")
                            ),
                            "срок": self.format_date(self.get_safe(re, "deadline")),
                        }
                        for re in (getattr(doc, "responsibleExecutors", None) or [])
                    ]
                    if getattr(doc, "responsibleExecutors", None)
                    else None
                ),
                "количество_ответственных": getattr(
                    doc, "responsibleExecutorsCount", None
                ),
                "есть_ответственные": getattr(doc, "hasResponsibleExecutor", None),
            }

            # ========== 4. ЖИЗНЕННЫЙ ЦИКЛ И ПРОЦЕССЫ ==========
            lifecycle = {
                "текущий_статус": self.get_safe(doc, "status"),
                "предыдущий_статус": self.get_safe(doc, "prevStatus"),
                "текущий_этап_БП": getattr(doc, "currentBpmnTaskName", None),
                "процесс": (
                    {
                        "id": (
                            str(doc.processId)
                            if getattr(doc, "processId", None)
                            else None
                        ),
                        "завершен": self.get_safe(doc, "process.completed"),
                        "этапы": [
                            {
                                "название": getattr(item, "name", None),
                                "статус": (
                                    "Выполнен"
                                    if getattr(item, "completed", False)
                                    else "В работе"
                                ),
                                "дата_начала": self.format_datetime(
                                    self.get_safe(item, "startDate")
                                ),
                                "дата_окончания": self.format_datetime(
                                    self.get_safe(item, "endDate")
                                ),
                            }
                            for item in (self.get_safe(doc, "process.items") or [])
                        ],
                    }
                    if getattr(doc, "process", None)
                    else None
                ),
                "автомаршрутизация": getattr(doc, "autoRouting", None),
            }

            # ========== 5. КОНТРОЛЬ И СРОКИ ==========
            control_info = {
                "на_контроле": getattr(doc, "controlFlag", None),
                "снят_с_контроля": getattr(doc, "removeControl", None),
                "дней_на_исполнение": getattr(doc, "daysExecution", None),
                "автоконтроль": self.get_safe(doc, "autoControl"),
                "контролер": (
                    {
                        "сотрудник": self.format_user(
                            self.get_safe(doc, "control.controlEmployee")
                        ),
                        "срок": self.format_date(
                            self.get_safe(doc, "control.controlDate")
                        ),
                        "дата_постановки": self.format_date(
                            self.get_safe(doc, "control.dateControl")
                        ),
                        "снят": self.get_safe(doc, "control.removeControl"),
                    }
                    if getattr(doc, "control", None)
                    else None
                ),
            }

            # ========== 6. КАТЕГОРИАЛЬНАЯ СПЕЦИФИКАЦИЯ ==========
            specialized = {}

            # 6.1 ДОГОВОРЫ
            if getattr(doc, "contractNumber", None) or category == "CONTRACT":
                currency_code = self.get_safe(doc, "currency.code", "BYN")
                specialized["договор"] = {
                    "номер": getattr(doc, "contractNumber", None),
                    "дата": self.format_date(getattr(doc, "contractDate", None)),
                    "сумма": (
                        f"{doc.contractSum} {currency_code}"
                        if getattr(doc, "contractSum", None) is not None
                        else None
                    ),
                    "валюта_id": (
                        str(doc.currencyId)
                        if getattr(doc, "currencyId", None) is not None
                        else None
                    ),
                    "дата_подписания": self.format_date(
                        getattr(doc, "contractSigningDate", None)
                    ),
                    "дата_начала_действия": self.format_date(
                        getattr(doc, "contractStartDate", None)
                    ),
                    "начало_срока": self.format_date(
                        getattr(doc, "contractDurationStart", None)
                    ),
                    "конец_срока": self.format_date(
                        getattr(doc, "contractDurationEnd", None)
                    ),
                    "пролонгация": getattr(doc, "contractAutoProlongation", None),
                    "согласование": getattr(doc, "contractAgreement", None),
                    "типовой": getattr(doc, "contractTypical", None),
                }

            # 6.2 СОВЕЩАНИЯ
            if getattr(doc, "dateMeeting", None) or category in ["MEETING", "QUESTION"]:
                specialized["совещание"] = {
                    "дата": self.format_date(getattr(doc, "dateMeeting", None)),
                    "дата_заседания": self.format_date(
                        getattr(doc, "dateMeetingQuestion", None)
                    ),
                    "время_начала": self.format_datetime(
                        getattr(doc, "startMeeting", None)
                    ),
                    "время_окончания": self.format_datetime(
                        getattr(doc, "endMeeting", None)
                    ),
                    "место": getattr(doc, "placeMeeting", None),
                    "председатель": self.format_user(getattr(doc, "chairperson", None)),
                    "секретарь": self.format_user(getattr(doc, "secretary", None)),
                    "внешние_приглашенные": getattr(doc, "externalInvitees", None),
                    "количество_приглашенных": getattr(doc, "inviteesCount", None),
                    "количество_для_оповещения": getattr(
                        doc, "meetingQuestionNotifyCount", None
                    ),
                    "форма_проведения": self.get_safe(doc, "formMeetingType"),
                    "номер_вопроса_в_повестке": getattr(doc, "numberQuestion", None),
                    "есть_вопросы": getattr(doc, "hasQuestion", None),
                    "дополнение_к_повестке": getattr(doc, "addition", None),
                    "вопросы_повестки": [
                        {
                            "вопрос": getattr(q, "question", None),
                            "докладчик": self.format_user(self.get_safe(q, "reporter")),
                            "порядковый_номер": self.get_safe(q, "questionNumber"),
                        }
                        for q in (getattr(doc, "documentQuestions", None) or [])
                    ],
                    "родительское_заседание_id": (
                        str(doc.documentMeetingQuestionId)
                        if getattr(doc, "documentMeetingQuestionId", None)
                        else None
                    ),
                    "дополнительное_заседание_id": (
                        str(doc.additionMeetingQuestionId)
                        if getattr(doc, "additionMeetingQuestionId", None)
                        else None
                    ),
                    "дата_проведения_по_вопросу": self.format_date(
                        getattr(doc, "dateQuestion", None)
                    ),
                    "комментарий_руководителя": getattr(doc, "commentQuestion", None),
                }

            # 6.3 ОБРАЩЕНИЯ
            if category == "APPEAL":
                app = getattr(doc, "documentAppeal", None)
                if app and (
                    getattr(app, "fioApplicant", None)
                    or getattr(app, "organizationName", None)
                ):
                    specialized["обращение"] = {
                        "заявитель": getattr(app, "fioApplicant", None),
                        "организация": getattr(app, "organizationName", None),
                        "тип": (
                            "Коллективное"
                            if getattr(app, "collective", False)
                            else "Индивидуальное"
                        ),
                        "адрес": f"{getattr(app, 'regionName', None) or ''}, {getattr(app, 'cityName', None) or ''}, {getattr(app, 'fullAddress', None) or ''}".strip(
                            ", "
                        ),
                        "тематика": self.get_safe(app, "subject.name"),
                        "результат_решения": self.get_safe(app, "solutionResult.name"),
                        "дата_поступления": self.format_date(
                            self.get_safe(app, "receiptDate")
                        ),
                        "срок_рассмотрения": self.format_date(
                            self.get_safe(app, "considerationDeadline")
                        ),
                    }

            # ========== 7. ПОРУЧЕНИЯ И ЗАДАЧИ ==========
            tasks_info = {
                "общее_количество": getattr(doc, "countTask", None),
                "проектных": getattr(doc, "taskProjectCount", None),
                "завершенных": getattr(doc, "completedTaskCount", None),
                "список": (
                    [
                        {
                            "номер": getattr(t, "taskNumber", None),
                            "текст": getattr(t, "taskText", None),
                            "исполнитель": self.format_user(getattr(t, "author", None)),
                            "срок": (
                                self.format_date(getattr(t, "planedDateEnd", None))
                                if getattr(t, "planedDateEnd", None)
                                else "Бессрочно"
                            ),
                            "статус": self.get_safe(t, "taskStatus"),
                            "на_контроле": getattr(t, "onControl", None),
                            "создано": self.format_datetime(
                                self.get_safe(t, "createDate")
                            ),
                            "завершено": self.format_datetime(
                                self.get_safe(t, "factDateEnd")
                            ),
                        }
                        for t in (getattr(doc, "taskList", None) or [])
                    ]
                    if getattr(doc, "taskList", None)
                    else []
                ),
            }

            # ========== 8. БЕЗОПАСНОСТЬ И ДОСТУП ==========
            security = {
                "гриф_ДСП": getattr(doc, "dspFlag", None),
                "гриф_доступа_включен": getattr(doc, "enableAccessGrief", None),
                "гриф_доступа": (
                    {
                        "id": (
                            str(doc.accessGriefId)
                            if getattr(doc, "accessGriefId", None)
                            else None
                        ),
                        "название": self.get_safe(doc, "accessGrief.name"),
                        "уровень": self.get_safe(doc, "accessGrief.level"),
                    }
                    if getattr(doc, "accessGrief", None)
                    else None
                ),
            }

            # ========== 9. СВЯЗИ И ВЛОЖЕНИЯ ==========
            relations = {
                "вложения": [
                    {
                        "название": getattr(a, "name", None),
                        "id": str(a.id) if getattr(a, "id", None) else None,
                        "тип": self.get_safe(a, "attachmentDocumentType.name"),
                        "размер_байт": getattr(a, "size", None),
                        "дата_загрузки": self.format_datetime(
                            getattr(a, "uploadDate", None)
                        ),
                        "есть_ЭЦП": bool(
                            getattr(a, "signs", None) and len(a.signs) > 0
                        ),
                    }
                    for a in (getattr(doc, "attachmentDocument", None) or [])
                ],
                "количество_связей": getattr(doc, "documentLinksCount", None),
                "связанные_документы": {
                    "в_ответ_на": (
                        str(doc.answerDocId)
                        if getattr(doc, "answerDocId", None)
                        else None
                    ),
                    "получен_в_ответ": (
                        str(doc.receivedDocId)
                        if getattr(doc, "receivedDocId", None)
                        else None
                    ),
                    "ссылочный_документ": (
                        str(doc.refDocId) if getattr(doc, "refDocId", None) else None
                    ),
                },
                "дополнительные_документы": (
                    [
                        {
                            "название": self.get_safe(ad, "name"),
                            "тип": self.get_safe(ad, "type"),
                        }
                        for ad in (getattr(doc, "additionalDocuments", None) or [])
                    ]
                    if getattr(doc, "additionalDocuments", None)
                    else None
                ),
            }

            # ========== 10. ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ==========
            additional = {
                "страниц": getattr(doc, "pages", None),
                "листов_приложений": getattr(doc, "additionalPages", None),
                "номер_экземпляра": getattr(doc, "exemplarNumber", None),
                "количество_экземпляров": getattr(doc, "exemplarCount", None),
                "способ_получения": (
                    {
                        "id": getattr(doc, "deliveryMethodId", None),
                        "название": self.get_safe(doc, "deliveryMethod.name"),
                    }
                    if (
                        getattr(doc, "deliveryMethod", None)
                        or getattr(doc, "deliveryMethodId", None)
                    )
                    else None
                ),
                "страна": (
                    {
                        "id": (
                            str(doc.countryId)
                            if getattr(doc, "countryId", None)
                            else None
                        ),
                        "название": getattr(doc, "countryName", None),
                    }
                    if (
                        getattr(doc, "countryName", None)
                        or getattr(doc, "countryId", None)
                    )
                    else None
                ),
                "инвестиционная_программа": (
                    {
                        "id": (
                            str(doc.investProgramId)
                            if getattr(doc, "investProgramId", None)
                            else None
                        ),
                        "название": self.get_safe(doc, "investmentProgram.name"),
                    }
                    if getattr(doc, "investProgramId", None)
                    else None
                ),
                "версионность": {
                    "включена": getattr(doc, "versionFlag", None),
                    "id_версии": (
                        str(doc.documentVersionId)
                        if getattr(doc, "documentVersionId", None)
                        else None
                    ),
                    "информация": (
                        {
                            "номер": self.get_safe(doc, "version.versionNumber"),
                            "дата_создания": self.format_datetime(
                                self.get_safe(doc, "version.createDate")
                            ),
                            "автор": self.format_user(
                                self.get_safe(doc, "version.author")
                            ),
                        }
                        if getattr(doc, "version", None)
                        else None
                    ),
                },
                "ознакомление": (
                    {
                        "количество_визирующих": getattr(
                            doc, "introductionCount", None
                        ),
                        "количество_завизировавших": getattr(
                            doc, "introductionCompleteCount", None
                        ),
                        "список": [
                            {
                                "сотрудник": self.format_user(
                                    self.get_safe(i, "employee")
                                ),
                                "дата": self.format_datetime(
                                    self.get_safe(i, "introductionDate")
                                ),
                                "завизировано": self.get_safe(i, "completed"),
                            }
                            for i in (getattr(doc, "introduction", None) or [])
                        ],
                    }
                    if getattr(doc, "introduction", None)
                    else None
                ),
                "номенклатура": (
                    {
                        "дела_для_списания": getattr(doc, "writeOffAffairCount", None),
                        "предварительных_дел": getattr(doc, "preAffairCount", None),
                        "список_предварительных": [
                            {
                                "название": self.get_safe(
                                    pn, "nomenclatureAffair.name"
                                ),
                                "индекс": self.get_safe(pn, "nomenclatureAffair.index"),
                            }
                            for pn in (
                                getattr(doc, "preNomenclatureAffairs", None) or []
                            )
                        ],
                    }
                    if (
                        getattr(doc, "writeOffAffairCount", None)
                        or getattr(doc, "preAffairCount", None)
                        or getattr(doc, "preNomenclatureAffairs", None)
                    )
                    else None
                ),
                "опись": self.get_safe(doc, "documentInventoryData"),
                "форма_документа": (
                    {
                        "id": (
                            str(doc.documentFormId)
                            if getattr(doc, "documentFormId", None)
                            else None
                        ),
                        "определение": self.get_safe(
                            doc, "documentFormDefinition.name"
                        ),
                    }
                    if (
                        getattr(doc, "documentFormId", None)
                        or getattr(doc, "documentFormDefinition", None)
                    )
                    else None
                ),
                "свойства_пользователя": (
                    {
                        "цвет": self.get_safe(doc, "color.colorCode"),
                        "прочие": self.get_safe(doc, "userProps"),
                    }
                    if (getattr(doc, "color", None) or getattr(doc, "userProps", None))
                    else None
                ),
            }

            # ========== 11. ПОЛЬЗОВАТЕЛЬСКИЕ ПОЛЯ ==========
            custom_fields = {}
            if getattr(doc, "customFields", None):
                custom_fields = {
                    "данные": doc.customFields,
                    "количество_полей": len(doc.customFields),
                }

            # ========== СБОРКА ФИНАЛЬНОГО РЕЗУЛЬТАТА ==========
            result = {
                "базовая_информация": self._clean_dict(base_info),
                "регистрация": self._clean_dict(registration),
                "участники": self._clean_dict(participants),
                "жизненный_цикл": self._clean_dict(lifecycle),
                "контроль": self._clean_dict(control_info),
                "задачи": self._clean_dict(tasks_info),
                "безопасность": self._clean_dict(security),
                "связи_и_вложения": self._clean_dict(relations),
                "дополнительная_информация": self._clean_dict(additional),
            }

            if specialized:
                result["специализированная_информация"] = self._clean_dict(specialized)

            if custom_fields:
                result["пользовательские_поля"] = custom_fields

            return result

        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            return {"error": "Ошибка обработки документа", "details": str(e)}

    def _clean_dict(self, d: Any) -> Any:
        """
        Рекурсивная очистка словаря от None и пустых значений.

        Args:
            d: Словарь или список для очистки

        Returns:
            Очищенная структура данных
        """
        if isinstance(d, dict):
            cleaned = {}
            for k, v in d.items():
                cleaned_value = self._clean_dict(v)
                if cleaned_value not in [None, [], {}, ""]:
                    cleaned[k] = cleaned_value
            return cleaned if cleaned else None

        elif isinstance(d, list):
            cleaned = [self._clean_dict(i) for i in d]
            cleaned = [item for item in cleaned if item not in [None, [], {}, ""]]
            return cleaned if cleaned else None

        else:
            return d

    # ========== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ==========

    def analyze_attachment_meta(self, attachment: Any) -> Dict[str, Any]:
        """
        Анализирует метаданные конкретного файла.

        Args:
            attachment: Объект вложения

        Returns:
            Словарь с метаданными файла
        """
        if not attachment:
            return {}

        try:
            return {
                "название": getattr(attachment, "name", None),
                "тип_вложения": self.get_safe(
                    attachment, "attachmentDocumentType.name"
                ),
                "размер_кб": (
                    round(attachment.size / 1024, 2)
                    if getattr(attachment, "size", None)
                    else 0
                ),
                "дата_загрузки": self.format_datetime(
                    getattr(attachment, "uploadDate", None)
                ),
                "есть_эцп": bool(
                    getattr(attachment, "signs", None) and len(attachment.signs) > 0
                ),
                "автор_id": (
                    str(attachment.authorId)
                    if getattr(attachment, "authorId", None)
                    else None
                ),
            }
        except Exception as e:
            logger.error(f"Error analyzing attachment: {e}")
            return {"error": str(e)}

    def analyze_local_file(self, file_path: str) -> Dict[str, Any]:
        """
        Анализирует параметры локального файла перед чтением.

        Args:
            file_path: Путь к файлу

        Returns:
            Словарь с параметрами файла
        """
        try:
            if not os.path.exists(file_path):
                return {"error": "Файл не найден"}

            stats = os.stat(file_path)
            ext = os.path.splitext(file_path)[1].lower()

            return {
                "имя_файла": os.path.basename(file_path),
                "расширение": ext,
                "размер_мб": round(stats.st_size / (1024 * 1024), 2),
                "путь": file_path,
                "тип_контента": (
                    "Документ" if ext in [".pdf", ".docx", ".doc"] else "Текст/Лог"
                ),
            }
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return {"error": str(e)}

    def suggest_summarize_format(self, text: str) -> Dict[str, Any]:
        """
        Определяет структуру текста и рекомендует формат анализа.

        Args:
            text: Текст для анализа

        Returns:
            Рекомендация по формату суммаризации
        """
        if not text:
            return {
                "recommended": "abstractive",
                "reason": "Текст пуст",
                "stats": {"chars": 0, "lines": 0},
            }

        length = len(text)
        lines = text.count("\n")
        has_many_digits = len(re.findall(r"\d+", text)) > 20

        if length > 5000 or has_many_digits:
            recommendation = "thesis"
            reason = (
                "Текст объемный или содержит много данных, тезисный план будет удобнее."
            )
        elif lines < 5:
            recommendation = "abstractive"
            reason = "Текст компактный, лучше всего подойдет краткий пересказ сути."
        else:
            recommendation = "extractive"
            reason = "В тексте много конкретики, выделим ключевые факты."

        return {
            "recommended": recommendation,
            "reason": reason,
            "stats": {"chars": length, "lines": lines},
        }

    def process_employee_info(self, emp: Any) -> Dict[str, Any]:
        """
        Формирует расширенную аналитическую карточку сотрудника.

        Args:
            emp: Объект сотрудника

        Returns:
            Структурированная карточка сотрудника
        """
        if not emp:
            return {}

        try:
            full_name = self.format_user(emp)

            return {
                "основное": {
                    "фио": full_name,
                    "должность": self.get_safe(emp, "post.postName"),
                    "департамент": self.get_safe(emp, "department.name"),
                    "статус": "Уволен" if getattr(emp, "fired", False) else "Активен",
                    "является_ио": getattr(emp, "io", None),
                },
                "контакты": {
                    "email": getattr(emp, "email", None),
                    "телефон": getattr(emp, "phone", None),
                    "адрес": getattr(emp, "address", None),
                    "площадка": getattr(emp, "place", None),
                },
                "структура": {
                    "руководитель_подразделения": getattr(
                        emp, "currentUserLeader", None
                    ),
                    "код_департамента": self.get_safe(emp, "department.departmentCode"),
                    "id_департамента": (
                        str(emp.departmentId)
                        if getattr(emp, "departmentId", None)
                        else None
                    ),
                },
            }
        except Exception as e:
            logger.error(f"Error processing employee info: {e}")
            return {"error": str(e)}

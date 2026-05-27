# edms_ai_assistant/services/entity_extractor.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from edms_ai_assistant.utils.datetime_utils import LOCAL_TZ, now_local, to_local_timezone
from edms_ai_assistant.utils.regex_utils import UUID_RE

logger = logging.getLogger(__name__)


class EntityType(Enum):
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
    type: EntityType
    value: Any
    raw_text: str
    confidence: float = 1.0
    normalized_value: Any | None = None


class EntityExtractor:
    """Rule-based named entity extractor for EDMS domain text."""

    DATE_PATTERNS: list[tuple[str, Any]] = [
        (r"(\d{1,2})\.(\d{1,2})\.(\d{4})", lambda m: f"{m[2]}-{int(m[1]):02d}-{int(m[0]):02d}"),
        (r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)(?:\s+(\d{4}))?",
         "month_name"),
        (r"\b(сегодня|завтра|вчера|послезавтра)\b", "relative_day"),
        (r"через\s+(\d+)\s+(день|дня|дней|неделю|недели|недель|месяц|месяца|месяцев)", "duration"),
        (r"до\s+(\d{1,2})\.(\d{1,2})", "deadline"),
    ]

    MONTH_NAMES: dict[str, int] = {
        "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
        "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
    }

    _PERSON_STOP_WORDS = {
        "через", "после", "перед", "около", "между", "вокруг", "создай", "создать",
        "добавь", "добавить", "найди", "найти", "покажи", "показать", "сделай", "сделать",
        "поручи", "поручить", "отправь", "отправить", "напиши", "написать", "проверь",
        "проверить", "оформи", "оформить", "зарегистрируй", "зарегистрировать", "сравни",
        "сравнить", "проанализируй", "проанализировать", "удали", "удалить", "измени",
        "изменить", "поставь", "поставить", "сними", "снять", "завизируй", "завизировать",
        "документ", "поручение", "ознакомление", "задача", "задачу", "контроль",
        "уведомление", "резолюция", "обращение", "договор", "совещание", "исполнитель",
        "контролёр", "автор", "инициатор",
    }

    def extract_dates(self, text: str, base_date: datetime | None = None) -> list[Entity]:
        if base_date is None: base_date = now_local()
        dates: list[Entity] = []
        for pattern, handler in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                raw = match.group(0)
                try:
                    normalized: datetime
                    if handler == "month_name":
                        day, month = int(match.group(1)), self.MONTH_NAMES[match.group(2)]
                        year = int(match.group(3)) if match.group(3) else base_date.year
                        normalized = datetime(year, month, day, tzinfo=LOCAL_TZ)
                    elif handler == "relative_day":
                        delta_map = {"сегодня": 0, "завтра": 1, "послезавтра": 2, "вчера": -1}
                        normalized = base_date + timedelta(days=delta_map[match.group(1)])
                    elif handler == "duration":
                        count, unit = int(match.group(1)), match.group(2)
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
                        day, month = int(match.group(1)), int(match.group(2))
                        if not (1 <= month <= 12 and 1 <= day <= 31): continue
                        normalized = datetime(base_date.year, month, day, tzinfo=LOCAL_TZ)
                        if normalized < base_date: normalized = datetime(base_date.year + 1, month, day,
                                                                         tzinfo=LOCAL_TZ)
                    elif callable(handler):
                        parsed = datetime.fromisoformat(handler(match.groups()))
                        normalized = parsed.replace(tzinfo=LOCAL_TZ) if parsed.tzinfo is None else parsed.astimezone(
                            LOCAL_TZ)
                    else:
                        continue
                    dates.append(Entity(type=EntityType.DATE, value=normalized, raw_text=raw,
                                        normalized_value=to_local_timezone(normalized)))
                except (ValueError, KeyError) as exc:
                    logger.debug("Failed to parse date '%s': %s", raw, exc)
        return dates

    def extract_persons(self, text: str) -> list[Entity]:
        persons = []
        for match in re.finditer(r"\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)(?:\s+([А-ЯЁ][а-яё]+))?\b", text):
            last_name, first_name = match.group(1), match.group(2)
            if last_name.lower() in self._PERSON_STOP_WORDS or first_name.lower() in self._PERSON_STOP_WORDS: continue
            persons.append(Entity(type=EntityType.PERSON, value=match.group(0), raw_text=match.group(0),
                                  normalized_value={"lastName": last_name, "firstName": first_name,
                                                    "middleName": match.group(3)},
                                  confidence=0.8 if match.group(3) else 0.6))
        return persons

    @staticmethod
    def extract_numbers(text: str) -> list[Entity]:
        return [Entity(type=EntityType.NUMBER, value=float(m.group(0).replace(",", ".")), raw_text=m.group(0),
                       normalized_value=float(m.group(0).replace(",", "."))) for m in
                re.finditer(r"\b(\d+(?:[.,]\d+)?)\b", text)]

    @staticmethod
    def extract_money(text: str) -> list[Entity]:
        money = []
        for pattern, currency in [(r"(\d+(?:[.,]\d+)?)\s*(руб|₽|rub|бел\.руб)", "BYN"),
                                  (r"(\d+(?:[.,]\d+)?)\s*(\$|usd|долл)", "USD"),
                                  (r"(\d+(?:[.,]\d+)?)\s*(€|eur|евро)", "EUR"),
                                  (r"(\d+(?:[.,]\d+)?)\s*(руб|rub)", "RUB")]:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                amount = float(match.group(1).replace(",", "."))
                money.append(Entity(type=EntityType.MONEY, value=amount, raw_text=match.group(0),
                                    normalized_value={"amount": amount, "currency": currency}))
        return money

    @staticmethod
    def extract_document_ids(text: str) -> list[Entity]:
        return [Entity(type=EntityType.DOCUMENT_ID, value=m.group(0), raw_text=m.group(0),
                       normalized_value=m.group(0).lower()) for m in UUID_RE.finditer(text)]

    def extract_all(self, text: str, base_date: datetime | None = None) -> dict[str, list[Entity]]:
        entities: dict[str, list[Entity]] = {}

        def _get_dates() -> list[Entity]:
            return self.extract_dates(text, base_date)

        def _get_persons() -> list[Entity]:
            return self.extract_persons(text)

        def _get_numbers() -> list[Entity]:
            return self.extract_numbers(text)

        def _get_money() -> list[Entity]:
            return self.extract_money(text)

        def _get_document_ids() -> list[Entity]:
            return self.extract_document_ids(text)

        extractors = [
            ("dates", _get_dates),
            ("persons", _get_persons),
            ("numbers", _get_numbers),
            ("money", _get_money),
            ("document_ids", _get_document_ids),
        ]

        for key, extractor in extractors:
            result = extractor()
            if result:
                entities[key] = result
        return entities

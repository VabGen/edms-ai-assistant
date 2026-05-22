# edms_ai_assistant/services/query_refiner.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from edms_ai_assistant.services.entity_extractor import Entity

logger = logging.getLogger(__name__)


class UserIntent(Enum):
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
    FILE_ANALYSIS = "file_analysis"
    CREATE_DOCUMENT = "create_document"
    COMPLIANCE_CHECK = "compliance_check"
    CONTROL = "control"
    ACCESS_GRIEF = "access_grief"
    APPEAL_AUTOFILL = "appeal_autofill"


class QueryComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class UserQuery:
    original: str
    refined: str
    intent: UserIntent
    secondary_intents: list[UserIntent] = field(default_factory=list)
    complexity: QueryComplexity = QueryComplexity.SIMPLE
    entities: dict[str, list[Entity]] = field(default_factory=dict)
    keywords: set[str] = field(default_factory=set)
    confidence: float = 1.0


@dataclass
class SemanticContext:
    query: UserQuery
    document: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class QueryRefiner:
    """Normalises and enriches raw user queries before LLM dispatch."""

    ABBREVIATIONS: dict[str, str] = {
        "doc": "документ", "dok": "документ", "доки": "документы", "сэд": "СЭД",
        "ознак": "ознакомление", "пор": "поручение", "исп": "исполнитель",
        "отв": "ответственный", "сов": "совещание", "дог": "договор",
        "рег": "регистрационный", "нотиф": "уведомление", "резол": "резолюция"
    }

    ACTION_SYNONYMS: dict[str, str] = {
        "покажи": "найди", "выведи": "найди", "дай": "найди",
        "скажи": "опиши", "расскажи": "опиши", "объясни": "опиши"
    }

    CONTROL_DOMAIN_SYNONYMS = {
        "поставь контроль": "управление контролем документа поставить",
        "поставить контроль": "управление контролем документа поставить",
        "сними с контроля": "управление контролем документа снять",
        "снять с контроля": "управление контролем документа снять",
        "измени контроль": "управление контролем документа изменить",
        "изменить контроль": "управление контролем документа изменить",
        "удали контроль": "управление контролем документа удалить",
        "удалить контроль": "управление контролем документа удалить",
        "кто контролёр": "управление контролем документа посмотреть",
        "статус контроля": "управление контролем документа посмотреть",
    }

    EDMS_DOMAIN_SYNONYMS: dict[str, str] = {
        "история документа": "сравни версии документа", "историю документа": "сравни версии документа",
        "что менялось": "сравни версии документа", "покажи изменения": "сравни версии документа",
        "наложить резолюцию": "добавь резолюцию", "поставь визу": "добавь резолюцию",
        "дай задачу": "создай поручение", "поставь задачу": "создай поручение",
        "виза": "ознакомление", "завизировать": "добавить в ознакомление",
        "напомни о документе": "отправь уведомление", "уведомить": "отправь уведомление",
        "вкратце": "кратко опиши", "о чём документ": "кратко опиши документ",
        "найти документ": "найди документ", "поиск документов": "найди документы",
        "инфо": "информация о документе", "реквизиты": "информация о документе",
        "создай документ": "создай документ из файла", "оформи": "создай документ из файла",
    }

    def normalize_domain_synonyms(self, text: str) -> str:
        text_lower = text.lower()
        sorted_pairs = sorted(self.EDMS_DOMAIN_SYNONYMS.items(), key=lambda x: -len(x[0]))
        if not sorted_pairs: return text_lower
        pattern = re.compile("|".join(re.escape(jargon) for jargon, _ in sorted_pairs), flags=re.IGNORECASE)
        canonical_map = {jargon.lower(): canonical for jargon, canonical in sorted_pairs}
        return pattern.sub(lambda m: canonical_map.get(m.group(0).lower(), m.group(0)), text_lower)

    def expand_abbreviations(self, text: str) -> str:
        return " ".join(self.ABBREVIATIONS.get(w.lower(), w) for w in text.split())

    def normalize_actions(self, text: str) -> str:
        text_lower = text.lower()
        for synonym, canonical in self.ACTION_SYNONYMS.items():
            text_lower = re.sub(r"\b" + synonym + r"\b", canonical, text_lower)
        return text_lower

    def add_context(self, text: str, intent: UserIntent, entities: dict[str, list[Entity]]) -> str:
        hints: list[str] = []
        if intent == UserIntent.SEARCH:
            if "persons" in entities: hints.append(f"исполнитель: {entities['persons'][0].value}")
            if "dates" in entities: hints.append(
                f"дата: {entities['dates'][0].normalized_value or entities['dates'][0].raw_text}")
        elif intent == UserIntent.CREATE_TASK:
            hints.append(
                f"срок: {entities['dates'][0].normalized_value}" if "dates" in entities else "срок: +7 дней (не указан)")
            if "persons" in entities: hints.append(f"исполнитель: {entities['persons'][0].value}")
        elif intent == UserIntent.COMPARE:
            ids = entities.get("document_ids", [])
            if len(ids) >= 2:
                hints.extend([f"версия_1: {ids[0].value}", f"версия_2: {ids[1].value}"])
            else:
                hints.append("версии: авто")
        elif intent == UserIntent.SUMMARIZE:
            if "numbers" in entities: hints.append(f"объём: {entities['numbers'][0].value} слов")
        elif intent == UserIntent.COMPOSITE:
            hints.append("тип: составной запрос")

        return text + " [" + "; ".join(hints) + "]" if hints else text

    def refine(self, text: str, intent: UserIntent, entities: dict[str, list[Entity]]) -> str:
        refined = text.strip()
        refined = self.normalize_domain_synonyms(refined)
        refined = self.expand_abbreviations(refined)
        refined = self.normalize_actions(refined)
        refined = self.add_context(refined, intent, entities)
        return re.sub(r"\s+", " ", refined).strip()

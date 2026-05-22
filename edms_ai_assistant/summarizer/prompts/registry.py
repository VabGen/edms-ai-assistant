"""
Typed Prompt Registry — version-controlled prompts.
"""

from __future__ import annotations

import json
from typing import Final

from pydantic import BaseModel

from edms_ai_assistant.summarizer.structured.models import (
    MODE_OUTPUT_MODEL,
    SummaryMode,
)

PROMPT_REGISTRY_VERSION: Final[str] = "2025.06.002"


def _schema_hint(mode: SummaryMode) -> str:
    """Compact JSON Schema hint for the mode's output model."""
    model_cls = MODE_OUTPUT_MODEL[mode]
    schema = model_cls.model_json_schema()
    required = schema.get("required", [])
    props = {
        k: v.get("description", v.get("type", "string"))
        for k, v in schema.get("properties", {}).items()
        if k in required
    }
    return json.dumps(props, ensure_ascii=False, separators=(",", ":"))


class PromptTemplate(BaseModel):
    model_config = {"frozen": True}

    name: str
    mode: SummaryMode
    version: str
    system: str
    user_template: str

    def render(
        self,
        text: str,
        *,
        language: str = "ru",
        extra: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        schema = _schema_hint(self.mode)
        system = self.system.replace("{language}", language)
        user = self.user_template.format(
            text=text,
            language=language,
            schema=schema,
            **(extra or {}),
        )
        return system, user


# ---------------------------------------------------------------------------
# Системная инструкция — основа для всех промптов
# ---------------------------------------------------------------------------

_SYSTEM_BASE = """Ты — эксперт по анализу документов системы электронного документооборота (СЭД/EDMS).

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА (нарушение недопустимо):
1. ЯЗЫК ОТВЕТА: Отвечай СТРОГО на языке "{language}". Если язык "ru" — весь текст только на русском. Никакого английского.
2. ФОРМАТ: Верни ТОЛЬКО валидный JSON. Без markdown-блоков (```json), без комментариев, без пояснений до/после JSON.
3. ЭМОДЗИ: ЗАПРЕЩЕНО использовать эмодзи в тексте. Тон должен быть строго профессиональным и деловым.
4. ФАКТЫ: Извлекай только то, что явно написано в документе. Не придумывай.
5. ТЕХНИЧЕСКИЕ ДАННЫЕ: Игнорируй UUID, системные идентификаторы, пути к файлам.
6. ДЛИНА: Не обрезай значения строк. Завершай JSON полностью."""

# ---------------------------------------------------------------------------
# Промпты для каждого режима
# ---------------------------------------------------------------------------

PROMPT_EXECUTIVE = PromptTemplate(
    name="executive_v2",
    mode=SummaryMode.EXECUTIVE,
    version="2.0.0",
    system=_SYSTEM_BASE,
    user_template="""Создай краткую выжимку документа для руководителя.

Язык ответа: {language}

Верни JSON строго в этом формате:
{{
  "headline": "одно предложение — главная мысль (не более 200 символов)",
  "bullets": ["тезис 1", "тезис 2", "тезис 3"],
  "recommendation": "рекомендуемое действие или null"
}}

Требования:
- headline: одно законченное предложение, суть документа
- bullets: 3-5 ключевых пункта, каждый не более 20 слов
- recommendation: только если документ требует решения, иначе null

Текст документа:
{text}""",
)

PROMPT_DETAILED_NOTES = PromptTemplate(
    name="detailed_notes_v2",
    mode=SummaryMode.DETAILED_NOTES,
    version="2.0.0",
    system=_SYSTEM_BASE,
    user_template="""Создай структурированный конспект документа.

Язык ответа: {language}

Верни JSON строго в этом формате:
{{
  "document_type": "тип документа (ДОГОВОР/ПИСЬМО/ПРИКАЗ/ПРОТОКОЛ/РЕГЛАМЕНТ/ИНОЕ)",
  "sections": [
    {{
      "title": "название раздела",
      "content": "содержание раздела",
      "subsections": ["подпункт 1", "подпункт 2"]
    }}
  ],
  "key_entities": ["организация 1", "персона 1", "документ №..."],
  "date_range": "период дат или null"
}}

Требования:
- sections: следуй структуре самого документа, не более 15 разделов
- key_entities: все упомянутые организации, люди, ссылки на документы
- date_range: если есть даты — укажи диапазон в формате "ДД.ММ.ГГГГ — ДД.ММ.ГГГГ"

Текст документа:
{text}""",
)

PROMPT_ACTION_ITEMS = PromptTemplate(
    name="action_items_v2",
    mode=SummaryMode.ACTION_ITEMS,
    version="2.0.0",
    system=_SYSTEM_BASE,
    user_template="""Извлеки все задачи, поручения и обязательства из документа.

Язык ответа: {language}

Верни JSON строго в этом формате:
{{
  "action_items": [
    {{
      "task": "описание задачи",
      "owner": "ответственный или null",
      "deadline": "ГГГГ-ММ-ДД или null",
      "priority": "high/medium/low",
      "source_fragment": "цитата из документа",
      "confidence": 0.9
    }}
  ],
  "document_context": "краткий контекст"
}}

Правила приоритетов:
- high: "срочно", "немедленно", "обязательно", ближайший дедлайн
- medium: обычные поручения со сроками
- low: рекомендации, пожелания

Если задач нет: верни {{"action_items": [], "document_context": "краткий контекст документа"}}

Текст документа:
{text}""",
)

PROMPT_THESIS = PromptTemplate(
    name="thesis_v2",
    mode=SummaryMode.THESIS,
    version="2.0.0",
    system=_SYSTEM_BASE,
    user_template="""Составь тезисный план документа.

Язык ответа: {language}

Верни JSON строго в этом формате:
{{
  "main_argument": "главный тезис/цель документа (одно предложение)",
  "sections": [
    {{
      "title": "название раздела",
      "thesis": "ключевой тезис раздела",
      "points": [
        {{
          "claim": "утверждение",
          "evidence": "обоснование или null",
          "sub_points": []
        }}
      ]
    }}
  ],
  "conclusion": "вывод или ожидаемый результат"
}}

Требования:
- main_argument: центральная идея всего документа
- sections: не более 6, следуй логике документа
- conclusion: итог или ожидаемый результат

Текст документа:
{text}""",
)

PROMPT_EXTRACTIVE = PromptTemplate(
    name="extractive_v3",
    mode=SummaryMode.EXTRACTIVE,
    version="3.0.0",
    system=_SYSTEM_BASE,
    user_template="""Извлеки ключевые факты из документа как структурированные данные.

Язык ответа: {language}

Верни JSON строго в этом формате:
{{
  "facts": [
    {{
      "category": "категория факта",
      "label": "краткое название (до 80 символов)",
      "value": "значение факта (до 300 символов)"
    }}
  ],
  "document_summary": "одно предложение — суть документа"
}}

Категории фактов:
- ДАТА — конкретные даты и дедлайны
- ПЕРСОНА — имена людей и их роли
- ОРГАНИЗАЦИЯ — названия организаций
- СУММА — денежные суммы, количества, объёмы
- ТРЕБОВАНИЕ — обязательные условия и ограничения
- СРОК — временные рамки и периоды
- ПРОЧЕЕ — другие важные факты

Правила:
- Максимум 20 фактов, приоритет по важности
- Извлекай только явно указанное в тексте

Текст документа:
{text}""",
)

PROMPT_ABSTRACTIVE = PromptTemplate(
    name="abstractive_v3",
    mode=SummaryMode.ABSTRACTIVE,
    version="3.0.0",
    system=_SYSTEM_BASE,
    user_template="""Напиши краткое изложение документа своими словами.

Язык ответа: {language}

Верни JSON строго в этом формате:
{{
  "summary": "краткое изложение в 2-4 абзацах",
  "key_themes": ["тема 1", "тема 2", "тема 3"]
}}

Требования:
- summary: профессиональный стиль, 2-4 абзаца, не более 2000 символов
- summary: перескажи своими словами, сохраняя все важные детали
- key_themes: 2-5 главных тем, кратко (1-4 слова каждая)

Текст документа:
{text}""",
)

PROMPT_MULTILINGUAL = PromptTemplate(
    name="multilingual_v2",
    mode=SummaryMode.MULTILINGUAL,
    version="2.0.0",
    system=_SYSTEM_BASE,
    user_template="""Проанализируй и изложи документ.

Язык вывода: {language}

Верни JSON строго в этом формате:
{{
  "detected_language": "BCP-47 код языка источника (ru/be/en/kk)",
  "summary_language": "{language}",
  "summary": "краткое изложение на языке {language}",
  "translation_notes": "примечания по переводу или null"
}}

Требования:
- detected_language: определи язык исходного документа
- summary: полное изложение на языке {language}, 2-4 абзаца
- translation_notes: только если есть термины, которые сложно перевести

Текст документа:
{text}""",
)

# --- Map-stage промпты для Map-Reduce ---

PROMPT_MAP_GENERIC = PromptTemplate(
    name="map_generic_v2",
    mode=SummaryMode.ABSTRACTIVE,
    version="2.0.0",
    system="""Ты обрабатываешь ФРАГМЕНТ большого документа.
Твоя задача — извлечь ключевую информацию из этого фрагмента.
ОБЯЗАТЕЛЬНО: отвечай на русском языке. Только текст, без JSON.""",
    user_template="""Изложи ключевое содержание этого фрагмента в 2-4 предложениях на русском языке.
Не используй JSON. Только текст.

Фрагмент:
{text}""",
)

PROMPT_MAP_EXTRACTIVE = PromptTemplate(
    name="map_extractive_v2",
    mode=SummaryMode.EXTRACTIVE,
    version="2.0.0",
    system="""Ты обрабатываешь ФРАГМЕНТ большого документа.
Извлеки ключевые факты из этого фрагмента.
ОБЯЗАТЕЛЬНО: отвечай на русском языке. Только текст, без JSON.""",
    user_template="""Перечисли 3-5 ключевых фактов из этого фрагмента на русском языке.
Формат: "- Факт: значение"
Только текст, без JSON.

Фрагмент:
{text}""",
)

PROMPT_QUALITY_JUDGE = PromptTemplate(
    name="quality_judge_v1",
    mode=SummaryMode.ABSTRACTIVE,
    version="1.0.0",
    system="""Ты — строгий редактор-эксперт. Твоя задача — оценить качество
суммаризации документа по шкале от 0.0 до 1.0.

Критерии оценки:
- Точность фактов (нет ли искажений / выдумок)
- Полнота (покрыты ли ключевые моменты документа)
- Связность изложения
- Соответствие запрошенному формату

Отвечай ТОЛЬКО валидным JSON, без markdown-блоков и пояснений.""",
    user_template="""Оцени качество следующей суммаризации.

Исходный документ (фрагмент):
{text}

Сгенерированная суммаризация:
{summary}

Верни JSON:
{{
  "score": 0.85,
  "critique": "краткое объяснение оценки на русском (2-3 предложения)"
}}

Шкала:
- 0.9-1.0: образцово точно и полно
- 0.7-0.9: хорошо, минорные недочёты
- 0.5-0.7: средне, есть пропуски или неточности
- 0.0-0.5: плохо, серьёзные проблемы""",
)


PROMPT_REDUCE_EXECUTIVE = PromptTemplate(
    name="reduce_executive_v2",
    mode=SummaryMode.EXECUTIVE,
    version="2.0.0",
    system=_SYSTEM_BASE,
    user_template="""Объедини эти частичные изложения в итоговую выжимку для руководителя.

Язык ответа: {language}

Верни JSON строго в этом формате:
{{
  "headline": "главная мысль (одно предложение)",
  "bullets": ["тезис 1", "тезис 2", "тезис 3"],
  "recommendation": "рекомендуемое действие или null"
}}

Частичные изложения:
{text}""",
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PromptRegistry:
    _DIRECT: dict[SummaryMode, PromptTemplate] = {
        SummaryMode.EXECUTIVE: PROMPT_EXECUTIVE,
        SummaryMode.DETAILED_NOTES: PROMPT_DETAILED_NOTES,
        SummaryMode.ACTION_ITEMS: PROMPT_ACTION_ITEMS,
        SummaryMode.THESIS: PROMPT_THESIS,
        SummaryMode.EXTRACTIVE: PROMPT_EXTRACTIVE,
        SummaryMode.ABSTRACTIVE: PROMPT_ABSTRACTIVE,
        SummaryMode.MULTILINGUAL: PROMPT_MULTILINGUAL,
    }

    _MAP: dict[SummaryMode, PromptTemplate] = {
        SummaryMode.EXTRACTIVE: PROMPT_MAP_EXTRACTIVE,
        SummaryMode.ABSTRACTIVE: PROMPT_MAP_GENERIC,
        SummaryMode.EXECUTIVE: PROMPT_MAP_GENERIC,
        SummaryMode.DETAILED_NOTES: PROMPT_MAP_GENERIC,
        SummaryMode.ACTION_ITEMS: PROMPT_MAP_GENERIC,
        SummaryMode.THESIS: PROMPT_MAP_GENERIC,
        SummaryMode.MULTILINGUAL: PROMPT_MAP_GENERIC,
    }

    _REDUCE: dict[SummaryMode, PromptTemplate] = {
        SummaryMode.EXECUTIVE: PROMPT_REDUCE_EXECUTIVE,
    }

    def get(self, mode: SummaryMode) -> PromptTemplate:
        return self._DIRECT[mode]

    def get_map(self, mode: SummaryMode) -> PromptTemplate:
        return self._MAP.get(mode, PROMPT_MAP_GENERIC)

    def get_reduce(self, mode: SummaryMode) -> PromptTemplate:
        return self._REDUCE.get(mode, self._DIRECT[mode])

    @staticmethod
    def get_judge() -> PromptTemplate:
        """Промпт для LLM-as-judge — оценки качества вывода."""
        return PROMPT_QUALITY_JUDGE

    @staticmethod
    def version() -> str:
        return PROMPT_REGISTRY_VERSION

    @staticmethod
    def cache_version_tag() -> str:
        return f"prompts:{PROMPT_REGISTRY_VERSION}"


_registry: PromptRegistry | None = None


def get_prompt_registry() -> PromptRegistry:
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry

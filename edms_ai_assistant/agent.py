# edms_ai_assistant/agent.py
"""
EDMS AI Assistant — Core Agent Module.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import UUID

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, field_validator, model_validator

from edms_ai_assistant.clients.document_client import DocumentClient
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.model import AgentState
from edms_ai_assistant.services.nlp_service import (
    SemanticDispatcher,
    UserIntent,
)
from edms_ai_assistant.tools import all_tools
from edms_ai_assistant.utils.regex_utils import UUID_RE

logger = logging.getLogger(__name__)

# ─── Фразы успешных мутирующих операций для requires_reload ──────────────────
# При совпадении любой из них — фронтенд должен перезагрузить страницу EDMS
_MUTATION_SUCCESS_PHRASES: tuple[str, ...] = (
    "успешно добавлен",
    "успешно создан",
    "список ознакомления",
    "поручение создано",
    "поручение успешно",
    "обращение заполнено",
    "обращение успешно",
    "карточка заполнена",
    "добавлено в список",
    "добавлен в список",
    "ознакомление создано",
    "задача создана",
    # Резолюции
    "резолюция успешно добавлена",
    "резолюция добавлена",
    "резолюция создана",
    # Уведомления
    "уведомление отправлено",
    "напоминание отправлено",
    "уведомлен",
)


def _is_valid_uuid(value: str) -> bool:
    """Returns True if *value* matches the canonical UUID4 pattern."""
    return bool(UUID_RE.match(value.strip()))


# ─── Инструменты требующие обязательного document_id из контекста ─────────────
_TOOLS_REQUIRING_DOCUMENT_ID: frozenset[str] = frozenset(
    {
        "doc_get_details",
        "doc_get_versions",
        "doc_compare",
        "doc_get_file_content",
        "doc_compare_with_local",
        "doc_summarize_text",
        "doc_search_tool",
        "introduction_create_tool",
        "task_create_tool",
        "doc_get_resolutions",
        "doc_create_resolution",
        "doc_send_notification",
    }
)

# ─── Placeholder-значения local_file_path для doc_compare_with_local ──────────
_COMPARE_LOCAL_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "",
        "local_file",
        "local_file_path",
        "/path/to/file",
        "path/to/file",
        "none",
        "null",
        "<local_file_path>",
        "<path>",
    }
)

# ─── Инструменты с disambiguation которые обрабатываются в _handle_human_choice
_DISAMBIGUATION_TOOLS: frozenset[str] = frozenset(
    {
        "introduction_create_tool",
        "task_create_tool",
        "doc_create_resolution",
        "doc_send_notification",
    }
)


def _is_mutation_response(content: Optional[str]) -> bool:
    """
    Returns True if the agent response describes a successful mutating EDMS operation.

    Used to signal the frontend to reload the page so that the EDMS SPA
    reflects the newly created/updated data without a manual refresh.

    Args:
        content: Final agent response text.

    Returns:
        True if the response contains a mutation success phrase.
    """
    if not content:
        return False
    lower = content.lower()
    return any(phrase in lower for phrase in _MUTATION_SUCCESS_PHRASES)


# ─────────────────────────────────────────────────────────────────────────────
# Domain value objects & enumerations
# ─────────────────────────────────────────────────────────────────────────────


class AgentStatus(str, Enum):
    """Agent execution result statuses."""

    SUCCESS = "success"
    ERROR = "error"
    REQUIRES_ACTION = "requires_action"
    PROCESSING = "processing"


class ActionType(str, Enum):
    """Types of interactive actions that require user participation."""

    SUMMARIZE_SELECTION = "summarize_selection"
    DISAMBIGUATION = "requires_disambiguation"
    CONFIRMATION = "requires_confirmation"


@dataclass
class ContextParams:
    """
    Immutable execution context passed through the entire agent lifecycle.

    Не использует frozen=True чтобы избежать проблем с field(default_factory)
    в Python 3.12. Иммутабельность обеспечивается соглашением — поля не
    изменяются после __post_init__.

    Attributes:
        user_token: JWT authorization token.
        document_id: UUID of the active EDMS document.
        file_path: UUID of an EDMS attachment or local filesystem path.
        thread_id: LangGraph conversation thread identifier.
        user_name: Display name for the system prompt.
        user_first_name: First name for personalized greetings.
        current_date: Formatted date string injected into the prompt.
    """

    user_token: str
    document_id: Optional[str] = None
    file_path: Optional[str] = None
    thread_id: str = "default"
    user_name: str = "пользователь"
    user_first_name: Optional[str] = None
    user_last_name: Optional[str] = None
    user_full_name: Optional[str] = None
    user_id: Optional[str] = None
    current_date: str = field(
        default_factory=lambda: datetime.now().strftime("%d.%m.%Y")
    )
    current_year: str = field(default_factory=lambda: str(datetime.now().year))
    uploaded_file_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.user_token or not isinstance(self.user_token, str):
            raise ValueError("user_token must be a non-empty string")
        if self.file_path and not self.uploaded_file_name:
            fp = str(self.file_path).strip()
            if not _is_valid_uuid(fp):
                self.uploaded_file_name = Path(fp).name
        if not self.user_full_name:
            parts = [p for p in (self.user_last_name, self.user_first_name) if p]
            if parts:
                self.user_full_name = " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Service layer request / response models (Pydantic v2)
# ─────────────────────────────────────────────────────────────────────────────


class AgentRequest(BaseModel):
    """Validated incoming request to the agent (Service Layer boundary)."""

    message: str = Field(default="", max_length=8000)
    user_token: str = Field(..., min_length=10)
    context_ui_id: Optional[str] = Field(
        None,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$|^$",
    )
    thread_id: Optional[str] = Field(None, max_length=255)
    user_context: Dict[str, Any] = Field(default_factory=dict)
    file_path: Optional[str] = Field(None, max_length=500)
    human_choice: Optional[str] = Field(None, max_length=200)

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        """Strips surrounding whitespace."""
        return v.strip()

    @model_validator(mode="after")
    def validate_message_or_choice(self) -> "AgentRequest":
        """
        Ensures the request has either a non-empty message or a human_choice.

        Human-in-the-Loop choice flows (summarize type selection, disambiguation)
        send human_choice as the primary payload — message may be empty or
        equal to the choice label. Both cases are valid.

        Raises:
            ValueError: If both message and human_choice are empty.
        """
        has_message = bool(self.message and self.message.strip())
        has_choice = bool(self.human_choice and self.human_choice.strip())
        if not has_message and not has_choice:
            raise ValueError("Either message or human_choice must be provided")
        if not has_message and has_choice:
            self.message = self.human_choice
        return self

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: Optional[str]) -> Optional[str]:
        """
        Validates file_path as UUID or filesystem path.

        Accepted formats:
        - UUID v4: ``550e8400-e29b-41d4-a716-446655440000``
        - Unix absolute path: ``/tmp/file.docx``
        - Windows absolute path: ``C:\\Users\\...\\file.docx``

        Args:
            v: Raw file path value.

        Returns:
            Cleaned value or None.

        Raises:
            ValueError: If the format is unrecognized.
        """
        if not v:
            return None
        stripped = v.strip()
        if _is_valid_uuid(stripped):
            return stripped
        if len(stripped) < 500:
            if stripped.startswith("/"):
                return stripped
            if re.match(r"^[A-Za-z]:\\", stripped):
                return stripped
            if re.match(r"^[^/\\]+[\\/]", stripped):
                return stripped
        raise ValueError(f"Invalid file_path format: {v!r}")


class AgentResponse(BaseModel):
    """Standardized agent execution result (internal, not exposed via HTTP)."""

    status: AgentStatus
    content: Optional[str] = None
    message: Optional[str] = None
    action_type: Optional[ActionType] = None
    requires_reload: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Repository interface & implementation (Dependency Inversion)
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class IDocumentRepository(Protocol):
    """
    Document repository interface.

    Decorated with @runtime_checkable so isinstance() checks work correctly
    in dependency injection and testing scenarios.
    """

    async def get_document(self, token: str, doc_id: str) -> Optional[DocumentDto]:
        """Fetches document metadata by ID."""
        ...


class DocumentRepository:
    """Production implementation of IDocumentRepository."""

    async def get_document(self, token: str, doc_id: str) -> Optional[DocumentDto]:
        """
        Fetches and validates document metadata from the EDMS REST API.

        Args:
            token: JWT authorization token.
            doc_id: UUID of the document to fetch.

        Returns:
            Validated DocumentDto or None on any error.
        """
        try:
            async with DocumentClient() as client:
                raw_data = await client.get_document_metadata(token, doc_id)
                doc = DocumentDto.model_validate(raw_data)
                logger.info("Document fetched", extra={"doc_id": doc_id})
                return doc
        except Exception as exc:
            logger.error(
                "Failed to fetch document",
                exc_info=True,
                extra={"doc_id": doc_id, "error": str(exc)},
            )
            return None


# ─────────────────────────────────────────────────────────────────────────────
# PromptBuilder — Strategy pattern для системных промптов
# ─────────────────────────────────────────────────────────────────────────────


class PromptBuilder:
    """
    Strategy for building system prompts with dynamic context injection.

    Принципы:
    - Базовый шаблон CORE_TEMPLATE обязателен для всех запросов
    - Дополнительные снипеты добавляются по интенту
    - Промпт НЕ содержит реальных значений token/document_id —
      они инжектируются в _orchestrate через args патчинг
    """

    CORE_TEMPLATE = """<role>
Ты — экспертный ИИ-помощник системы электронного документооборота (EDMS/СЭД).
Специализация: анализ документов, управление персоналом, автоматизация рутинных задач.
</role>

<context>
- Пользователь (имя): {user_name}
- Пользователь (фамилия): {user_last_name}
- Пользователь (полное имя): {user_full_name}
- Текущая дата: {current_date} (год: {current_year})
- Активный документ в EDMS: {context_ui_id}
- Загруженный файл/вложение: {local_file}
- Имя загруженного файла: {uploaded_file_name}
</context>

<current_user_rules>
Когда пользователь говорит "добавь меня", "я", "моя фамилия" и т.п.:
- Его фамилия: {user_last_name}
- Его полное имя: {user_full_name}
- Используй эти данные напрямую — НЕ спрашивай фамилию у пользователя.
- Передавай фамилию в инструменты поиска сотрудников автоматически.
</current_user_rules>

<critical_rules>
1. **Автоинъекция параметров**: `token` и `document_id` добавляются системой АВТОМАТИЧЕСКИ.
   Не указывай эти параметры явно при вызове инструментов.

2. **Работа с файлом/вложением**:
   - Если "Загруженный файл" — путь (/tmp/...): **ПРИОРИТЕТ 1** — используй ЭТОТ файл.
     Вызови `read_local_file_content(file_path=<путь>)` для анализа или суммаризации.
     **Не спрашивай пользователя — просто сделай это автоматически.**
   - Если "Загруженный файл" — UUID (0c2216e1-...): вызови `doc_get_file_content(attachment_id=<UUID>)`
   - Если "Загруженный файл" — "Не загружен": вызови `doc_get_details()` для поиска вложений в документе
   - **ЗАПРЕЩЕНО**: если указан "Загруженный файл" (путь или UUID) — НИКОГДА не вызывай `doc_get_file_content` с UUID из вложений документа. Работай ТОЛЬКО с указанным файлом.

3. **Строгая последовательность**:
   - Вызывай СТРОГО ОДИН инструмент за раз
   - Дождись результата инструмента, затем вызывай следующий
   - НИКОГДА не вызывай `doc_summarize_text` одновременно с `doc_get_file_content`
   - Правильно: получи текст → получи результат → передай текст в суммаризацию

4. **Disambiguation (requires_disambiguation)**:
   - При получении статуса "requires_disambiguation" — ПОКАЖИ пользователю список вариантов
   - Попроси выбрать конкретную позицию
   - Дождись ответа пользователя ПЕРЕД повторным вызовом инструмента

5. **Финальный ответ**:
   - ВСЕГДА формулируй итоговый ответ на РУССКОМ языке
   - Обращайся к пользователю по имени: {user_name}
   - Ответ должен быть понятен пользователю, без технических деталей API
   - Структурируй ответ: заголовок → ключевые факты → вывод

6. **Язык**: Только русский. Никаких английских терминов в ответе пользователю.

7. **ЗАПРЕТ технических данных в ответах**:
   - НИКОГДА не показывай UUID (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx) пользователю.
   - Вместо UUID сотрудника → используй его ФИО.
   - Вместо UUID вложения → используй имя файла.
   - Вместо UUID документа → используй название или номер документа.
   - Технические данные (ID, пути, токены) — только во внутренних вызовах инструментов.
</critical_rules>

<available_tools_guide>
| Сценарий                          | Последовательность инструментов                              |
|-----------------------------------|--------------------------------------------------------------|
| Анализ документа целиком          | doc_get_details → doc_get_file_content → doc_summarize_text  |
| Анализ конкретного вложения (UUID)| doc_get_file_content → doc_summarize_text                    |
| Анализ загруженного файла         | read_local_file_content → doc_summarize_text                 |
| Сравнение файла с вложением [ЕСТЬ файл]  | doc_compare_with_local (приоритет всегда)                     |
| Вопрос о документе                | doc_get_details                                              |
| Сравнение версий документа [НЕТ файла]   | doc_get_versions (возвращает все сравнения) |
| Поиск документов в базе EDMS      | doc_search_tool                                              |
| Поиск сотрудника                  | employee_search_tool                                         |
| Добавление в лист ознакомления    | introduction_create_tool                                     |
| Создание поручения                | task_create_tool                                             |
| Автозаполнение обращения          | autofill_appeal_document                                     |
| Просмотр резолюций                | doc_get_resolutions                                          |
| Создание резолюции                | [employee_search_tool →] doc_create_resolution               |
| Уведомление / напоминание         | employee_search_tool → doc_send_notification                 |
| Вопрос без документа              | Ответь напрямую из контекста                                 |
</available_tools_guide>

<response_format>
✅ Структурировано, кратко, информативно
✅ Маркированные списки для перечислений
✅ Выделение ключевых данных (суммы, даты, имена)
❌ Технические детали HTTP/API
❌ JSON-структуры в ответе пользователю
❌ Фразы "как ИИ я не могу..." — просто помогай
</response_format>"""

    _SNIPPETS: Dict[UserIntent, str] = {
        UserIntent.CREATE_INTRODUCTION: """
<introduction_workflow>
Workflow создания листа ознакомления:
1. Вызови introduction_create_tool с last_names сотрудников
2. Если вернулся "requires_disambiguation" → покажи список найденных сотрудников пользователю
3. Попроси пользователя указать ID нужного сотрудника
4. Повторный вызов: introduction_create_tool(selected_employee_ids=["uuid1", "uuid2"])
5. Сообщи пользователю об успехе с именами добавленных сотрудников
</introduction_workflow>""",
        UserIntent.CREATE_TASK: """
<task_creation_guide>
Параметры поручения:
- task_text: текст поручения (обязательно)
- executor_last_names: фамилии исполнителей (обязательно, минимум 1)
- responsible_last_name: ответственный исполнитель (опционально; если не указан → первый из executor_last_names)
- planed_date_end: дата в ISO 8601 (опционально; если не указана → автоматически +7 дней)

КРИТИЧНО — извлечение даты из текста задачи:
Если пользователь упоминает дату или срок В ЛЮБОЙ ФОРМЕ — ОБЯЗАТЕЛЬНО передай planed_date_end.
Примеры (текущий год = {current_year}):
- "к 15 апреля" → "{current_year}-04-15T23:59:59Z"
- "до 1 мая" → "{current_year}-05-01T23:59:59Z"
- "через неделю" → текущая дата ({current_date}) + 7 дней + "T23:59:59Z"
- "до конца месяца" → последний день текущего месяца + "T23:59:59Z"
- "срочно" / без даты → НЕ передавай planed_date_end (сервис поставит +7 дней)
Всегда добавляй суффикс 'Z' (UTC). Год = {current_year} если не указан явно.

Disambiguation: если исполнитель не найден однозначно → покажи список, дождись выбора.
</task_creation_guide>""",
        UserIntent.SUMMARIZE: """
<summarize_guide>
Workflow суммаризации документа:

ШАГ 1 — Получи текст:
  - Локальный файл: read_local_file_content(file_path=<путь>)
  - Вложение EDMS (UUID): doc_get_file_content(attachment_id=<UUID>)

ШАГ 2 — Вызови суммаризацию:
  doc_summarize_text(text=<полученный текст>, summary_type=<тип или None>)
  - Если пользователь явно указал формат ("сделай выжимку фактов", "тезисно", "перескажи") →
    передай соответствующий summary_type: extractive | thesis | abstractive
  - Если формат НЕ указан → передай summary_type=None, инструмент спросит пользователя

ШАГ 3 — Обработай ответ:
  - status=requires_choice → ПОКАЖИ пользователю три варианта и жди его выбора
  - status=success → представь результат структурировано

ЗАПРЕЩЕНО: подставлять summary_type самостоятельно если пользователь не указал формат.
</summarize_guide>""",
        UserIntent.COMPARE: """
<compare_decision_tree>
⚠️ ОБЯЗАТЕЛЬНО прочитай условие ДО выбора инструмента сравнения:

УСЛОВИЕ А: В контексте есть "Загруженный файл" (путь /tmp/... или UUID)?
  → ДА: ИСПОЛЬЗУЙ ТОЛЬКО doc_compare_with_local. СТОП. doc_get_versions НЕ вызывать.
  → НЕТ: ИСПОЛЬЗУЙ doc_get_versions (сам вернёт все сравнения, doc_compare НЕ нужен).

ЗАПРЕЩЕНО при наличии загруженного файла:
  ❌ doc_get_versions
  ❌ doc_compare
  ❌ предлагать пользователю "выбрать версию"
  ❌ спрашивать "какие версии сравнить"

Если загруженный файл ЕСТЬ, а пользователь говорит "нет" или "не то" —
это значит он хочет другое вложение, а НЕ версии документа.
Покажи список вложений документа через doc_get_details и дай выбрать.
</compare_decision_tree>

<compare_with_local_guide>
ПУТЬ А: Есть загруженный файл → doc_compare_with_local

ШАГ 1 — Вызови СРАЗУ, без предварительных вызовов:
  doc_compare_with_local(
      local_file_path=<АВТОМАТИЧЕСКИ из контекста, не спрашивай>,
      attachment_id=<имя или UUID вложения — только если пользователь явно указал>,
      document_id=<АВТОМАТИЧЕСКИ из контекста>
  )
  - Пользователь написал "сравни договор" → attachment_id="договор" (инструмент найдёт)
  - Пользователь написал просто "сравни" → НЕ передавай attachment_id (инструмент найдёт по имени файла)
  - НИКОГДА не вызывай doc_get_details перед этим

ШАГ 2 — Обработай ответ:
  - status=success → покажи результат (схожесть %, различия)
  - status=requires_disambiguation → покажи список вложений, пользователь выберет
  - status=error → сообщи об ошибке, предложи повторить

ШАГ 3 — Формат ответа:
  - "Сравнение: «{имя загруженного файла}» и «{имя вложения}»"
  - Схожесть: X%
  - Различия: что добавлено / что удалено
</compare_with_local_guide>

<compare_versions_guide>
ПУТЬ Б: НЕТ загруженного файла → сравнение версий документа

ШАГ 1 — Вызови doc_get_versions. Инструмент АВТОМАТИЧЕСКИ:
  - Получает ВСЕ N версий документа
  - Сравнивает КАЖДУЮ соседнюю пару: v1↔v2, v2↔v3, ..., v(N-1)↔vN
  - Возвращает поле "comparisons" с результатами всех пар

ШАГ 2 — Ответь пользователю, используя поле "comparisons" из ответа:
  - Для каждой пары: что изменилось в метаданных и вложениях
  - Если "has_any_changes" = false → версии идентичны
  - Если "comparison_complete" = true → НЕ вызывай doc_compare, данные уже есть

⚠️ ЗАПРЕЩЕНО:
  - Спрашивать "какие версии сравнить" — всё уже сравнено автоматически
  - Вызывать doc_compare после doc_get_versions — это дублирование
  - Вызывать doc_get_versions несколько раз

Формат ответа: по каждой паре — секция с изменениями (или "изменений нет").
</compare_versions_guide>""",
        UserIntent.SEARCH: """
<search_guide>
При поиске документов в базе EDMS:
- Поиск по тексту/номеру/категории/дате: doc_search_tool
  Параметры: search, reg_number, doc_category (INTERN/INCOMING/OUTGOING/APPEAL), date_from, date_to
- Поиск сотрудника по фамилии: employee_search_tool
- Информация о текущем документе из контекста: doc_get_details
- Если нужна информация из текста документа: doc_get_file_content → ответь на основе текста
После doc_search_tool можно передать id найденного документа в doc_get_details или doc_get_file_content.
</search_guide>""",
        UserIntent.ANALYZE: """
<analyze_guide>
Для глубокого анализа документа:
1. doc_get_details — структура, метаданные, поручения, процессы
2. doc_get_file_content — текстовое содержимое
3. doc_summarize_text с типом thesis — тезисный разбор
Обязательно укажи: тип документа, статус, ключевые участники, сроки.
</analyze_guide>""",
        UserIntent.QUESTION: """
<question_guide>
Отвечай на вопросы о документе:
- Простые вопросы о метаданных: doc_get_details
- Вопросы о содержимом: doc_get_file_content → ответ на основе текста
- Вопросы о сотрудниках: employee_search_tool
- Общие вопросы без документа: отвечай напрямую из контекста
</question_guide>""",
        UserIntent.RESOLUTION: """
<resolution_guide>
При работе с резолюциями документа:
- Просмотр резолюций: doc_get_resolutions(document_id=..., token=...)
- Создание резолюции: doc_create_resolution(document_id=..., resolution_text=..., executor_ids=[], deadline=...)
  - executor_ids — опциональный список UUID исполнителей (получи через employee_search_tool)
  - deadline — опциональная дата в ISO 8601 (например: "2026-04-01T23:59:59Z")
- Если в запросе упоминаются исполнители по фамилии → сначала employee_search_tool → затем doc_create_resolution
Резолюция — это официальное решение руководителя на документ.
</resolution_guide>""",
        UserIntent.NOTIFICATION: """
<notification_guide>
При отправке уведомлений и напоминаний:
- Инструмент: doc_send_notification(document_id=..., recipient_ids=[...], message=..., notification_type=..., deadline=...)
  - recipient_ids — UUID сотрудников (получи через employee_search_tool если не известны)
  - notification_type: REMINDER (напоминание), DEADLINE (срок), CUSTOM (произвольное)
  - deadline — опциональная дата дедлайна в ISO 8601
- Workflow: employee_search_tool → doc_send_notification
- Если сотрудник один и найден однозначно — сразу передавай его UUID.
</notification_guide>""",
        UserIntent.FILE_ANALYSIS: """
<file_analysis_guide>
При анализе загруженного файла:
- Локальный файл (/tmp/...): read_local_file_content → doc_summarize_text
- UUID вложения EDMS: doc_get_file_content → doc_summarize_text
- Сравнение файла с вложением документа: doc_compare_with_local
Путь к файлу берётся из <local_file_path> в system prompt.
</file_analysis_guide>""",
    }

    @classmethod
    def build(
        cls,
        context: ContextParams,
        intent: UserIntent,
        semantic_xml: str,
    ) -> str:
        """
        Assembles the full system prompt from context, intent snippet, and semantic XML.

        Args:
            context: Immutable execution context.
            intent: Detected primary user intent for snippet selection.
            semantic_xml: Pre-serialized semantic context XML block.

        Returns:
            Complete system prompt string ready for SystemMessage.
        """
        base = cls.CORE_TEMPLATE.format(
            user_name=context.user_first_name or context.user_name,
            user_last_name=context.user_last_name or "Не указана",
            user_full_name=context.user_full_name or context.user_name,
            current_date=context.current_date,
            current_year=context.current_year,
            context_ui_id=context.document_id or "Не указан",
            local_file=context.uploaded_file_name or context.file_path or "Не загружен",
            uploaded_file_name=context.uploaded_file_name or "Не определено",
        )
        snippet = cls._SNIPPETS.get(intent, "")
        return base + snippet + semantic_xml


# ─────────────────────────────────────────────────────────────────────────────
# ContentExtractor — извлечение финального контента из цепочки сообщений
# ─────────────────────────────────────────────────────────────────────────────


class ContentExtractor:
    """
    Extracts final human-readable content from a LangGraph message chain.

    Priority chain for extract_final_content:
    1. Last AIMessage with non-trivial text (not a tool-call marker)
    2. Last ToolMessage parsed as JSON (content/message/text fields)
    3. Fallback: any AIMessage with content
    4. Last resort: raw ToolMessage content

    The class is stateless — all methods are classmethods.
    """

    _SKIP_PATTERNS: tuple[str, ...] = (
        "вызвал инструмент",
        "tool call",
        '"name"',
        '"id"',
        '"tool_calls"',
    )
    MIN_CONTENT_LENGTH = 30
    _JSON_PRIORITY_FIELDS: tuple[str, ...] = (
        "content",
        "message",
        "text",
        "text_preview",
        "result",
    )
    _TECHNICAL_FIELDS: frozenset[str] = frozenset(
        {
            "status",
            "meta",
            "format_used",
            "was_truncated",
            "text_length",
            "suggestion",
            "action_type",
            "added_count",
            "not_found",
            "partial_success",
            "attachment_used",
            "warnings",
        }
    )

    @classmethod
    def extract_final_content(cls, messages: List[BaseMessage]) -> Optional[str]:
        """
        Extracts the final user-visible content from the message chain.

        Args:
            messages: Complete LangGraph message chain.

        Returns:
            Cleaned content string, or None if nothing found.
        """
        # Шаг 1: последнее содержательное AIMessage
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                text = str(m.content).strip()
                if not cls._is_technical(text) and len(text) >= cls.MIN_CONTENT_LENGTH:
                    logger.debug(
                        "Extracted final AIMessage", extra={"chars": len(text)}
                    )
                    return text

        # Шаг 2: последнее ToolMessage с распознаваемым JSON-полем
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                extracted = cls._parse_tool_message(m)
                if extracted:
                    logger.debug(
                        "Extracted ToolMessage JSON", extra={"chars": len(extracted)}
                    )
                    return extracted

        # Шаг 3: fallback — любое AIMessage
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                text = str(m.content).strip()
                if text:
                    logger.debug("Fallback AIMessage", extra={"chars": len(text)})
                    return text

        # Шаг 4: last resort — сырой ToolMessage
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                extracted = cls._parse_tool_message(m)
                if extracted:
                    return extracted

        return None

    @classmethod
    def extract_last_tool_text(cls, messages: List[BaseMessage]) -> Optional[str]:
        """
        Extracts substantial text content from the most recent ToolMessage.

        Used to feed file content into doc_summarize_text on the next
        orchestration iteration.

        Args:
            messages: Complete LangGraph message chain.

        Returns:
            Text content string (100+ chars) or None.
        """
        for m in reversed(messages):
            if not isinstance(m, ToolMessage):
                continue
            try:
                raw = str(m.content).strip()
                if raw.startswith("{"):
                    data: Dict[str, Any] = json.loads(raw)
                    for key in cls._JSON_PRIORITY_FIELDS:
                        val = data.get(key)
                        if val and len(str(val)) > 100:
                            return str(val)
                if len(raw) > 100:
                    return raw
            except json.JSONDecodeError:
                raw = str(m.content)
                if len(raw) > 100:
                    return raw
        return None

    @classmethod
    def clean_json_artifacts(cls, content: str) -> str:
        """
        Strips technical JSON wrappers from final content.

        Handles both clean JSON responses and mixed text with embedded JSON.

        Args:
            content: Raw content that may contain JSON envelopes.

        Returns:
            Clean human-readable text.
        """
        stripped = content.strip()

        # Случай 1: весь контент — это JSON объект
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                data = json.loads(stripped)
                for key in cls._JSON_PRIORITY_FIELDS:
                    val = data.get(key)
                    if (
                        val
                        and isinstance(val, str)
                        and len(val) >= cls.MIN_CONTENT_LENGTH
                    ):
                        return val.replace("\\n", "\n").replace('\\"', '"').strip()
            except (json.JSONDecodeError, ValueError):
                pass

        # Случай 2: JSON-обёртки встроены в текст — убираем регулярками
        # Убираем {"status": "success", "content": "..."} паттерны
        content = re.sub(
            r'\{"status"\s*:\s*"[^"]*",\s*"(?:content|message|text)"\s*:\s*"',
            "",
            content,
        )
        # Убираем хвостовые meta-поля
        content = re.sub(r'",\s*"meta"\s*:\s*\{[^}]*\}\s*\}', "", content)
        content = re.sub(
            r'",?\s*"[a-z_]+"\s*:\s*(?:true|false|null|\d+)\s*\}?\s*$', "", content
        )
        content = re.sub(r'"\s*\}$', "", content)

        return content.replace('\\"', '"').replace("\\n", "\n").strip()

    @classmethod
    def _is_technical(cls, content: str) -> bool:
        """Returns True if content is a technical marker not suitable for display."""
        lower = content.lower()
        return any(pattern in lower for pattern in cls._SKIP_PATTERNS)

    @classmethod
    def _parse_tool_message(cls, message: ToolMessage) -> Optional[str]:
        """
        Safely parses human-readable content from a ToolMessage.

        Args:
            message: LangGraph ToolMessage to parse.

        Returns:
            Extracted text or None.
        """
        try:
            raw = str(message.content).strip()
            if raw.startswith("{"):
                data: Dict[str, Any] = json.loads(raw)
                # Пропускаем технические ответы об ошибках
                if data.get("status") == "error":
                    return None
                for key in cls._JSON_PRIORITY_FIELDS:
                    val = data.get(key)
                    if (
                        val
                        and isinstance(val, str)
                        and len(val) >= cls.MIN_CONTENT_LENGTH
                    ):
                        return val
        except json.JSONDecodeError:
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# AgentStateManager — управление состоянием LangGraph
# ─────────────────────────────────────────────────────────────────────────────


class AgentStateManager:
    """
    Manages LangGraph graph state: invocation, inspection, and patching.

    Separates LangGraph API calls from orchestration logic,
    making it easier to swap checkpoint backends (MemorySaver → Postgres).
    """

    def __init__(self, graph: CompiledStateGraph, checkpointer: MemorySaver) -> None:
        """
        Initializes the state manager with a compiled graph and checkpointer.

        Args:
            graph: Compiled LangGraph state graph.
            checkpointer: Checkpoint backend (MemorySaver in dev, Postgres in prod).

        Raises:
            ValueError: If graph or checkpointer is None.
        """
        if graph is None:
            raise ValueError("Graph cannot be None")
        if checkpointer is None:
            raise ValueError("Checkpointer cannot be None")
        self.graph = graph
        self.checkpointer = checkpointer
        logger.debug(
            "AgentStateManager initialized",
            extra={
                "graph_type": type(graph).__name__,
                "checkpointer_type": type(checkpointer).__name__,
            },
        )

    def _config(self, thread_id: str) -> Dict[str, Any]:
        """Builds a LangGraph config dict for the given thread."""
        return {"configurable": {"thread_id": thread_id}}

    async def get_state(self, thread_id: str) -> Any:
        """
        Returns the current graph state snapshot for *thread_id*.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            StateSnapshot with .values (messages) and .next (pending nodes).
        """
        return await self.graph.aget_state(self._config(thread_id))

    async def update_state(
        self,
        thread_id: str,
        messages: List[BaseMessage],
        as_node: str = "agent",
    ) -> None:
        """
        Patches graph state for *thread_id* with new messages.

        Args:
            thread_id: Conversation thread identifier.
            messages: List of messages to merge into state.
            as_node: Node name to attribute the update to.
        """
        await self.graph.aupdate_state(
            self._config(thread_id),
            {"messages": messages},
            as_node=as_node,
        )

    async def invoke(
        self,
        inputs: Optional[Dict[str, Any]],
        thread_id: str,
        timeout: float = 120.0,
    ) -> None:
        """
        Invokes the graph for *thread_id* with optional inputs.

        Passing ``inputs=None`` resumes from the last interrupt point.

        Args:
            inputs: Initial graph inputs, or None to resume from interrupt.
            thread_id: Conversation thread identifier.
            timeout: Maximum wall-clock seconds to wait.

        Raises:
            asyncio.TimeoutError: If execution exceeds *timeout*.
        """
        await asyncio.wait_for(
            self.graph.ainvoke(inputs, config=self._config(thread_id)),
            timeout=timeout,
        )

    async def is_thread_broken(self, thread_id: str) -> bool:
        """Check if a thread has a dangling AIMessage with unresolved tool_calls.

        A thread is "broken" when:
        - The last message is an AIMessage with tool_calls (graph interrupted before tools)
        - AND the message before it is also an AIMessage (i.e. no ToolMessage response exists yet),
          OR the history structure violates the tool_call/tool_response pairing constraint.

        This state causes the LLM API to reject subsequent invocations with
        "An assistant message with tool_calls must be followed by tool messages".

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            True if the thread needs repair before next invocation.
        """
        try:
            state = await self.get_state(thread_id)
            messages = state.values.get("messages", [])
            if not messages:
                return False
            last = messages[-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                if len(messages) < 2 or not isinstance(messages[-2], ToolMessage):
                    return True
            return False
        except Exception:
            return False

    async def repair_thread(self, thread_id: str) -> bool:
        """Repair a broken thread by injecting synthetic ToolMessage error responses.

        For each unresolved tool_call in the last AIMessage, injects a synthetic
        ToolMessage with a graceful error payload. This satisfies the LLM API
        constraint and allows the graph to resume normally on the next user message.

        Strategy: inject synthetic ToolMessages → graph can now call agent node
        again → agent sees the errors and formulates a user-facing response.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            True if repair succeeded, False on failure.
        """
        try:
            state = await self.get_state(thread_id)
            messages = state.values.get("messages", [])
            if not messages:
                return False

            last = messages[-1]
            if not isinstance(last, AIMessage):
                return False

            tool_calls = getattr(last, "tool_calls", []) or []
            if not tool_calls:
                return False

            synthetic_tool_msgs = [
                ToolMessage(
                    content=json.dumps(
                        {
                            "status": "error",
                            "message": (
                                "Выполнение инструмента прервано: предыдущий запрос завершился "
                                "некорректно. Пожалуйста, повторите запрос."
                            ),
                        }
                    ),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
                for tc in tool_calls
            ]

            await self.update_state(
                thread_id,
                synthetic_tool_msgs,
                as_node="tools",
            )
            logger.warning(
                "Thread repaired: injected %d synthetic ToolMessage(s)",
                len(synthetic_tool_msgs),
                extra={"thread_id": thread_id},
            )
            return True

        except Exception as exc:
            logger.error(
                "Thread repair failed: %s",
                exc,
                extra={"thread_id": thread_id},
                exc_info=True,
            )
            return False


# ─────────────────────────────────────────────────────────────────────────────
# EdmsDocumentAgent — главный класс агента
# ─────────────────────────────────────────────────────────────────────────────


class EdmsDocumentAgent:
    """
    Production-ready EDMS AI agent orchestrating LangGraph tool-call workflows.

    Graph topology:
        START → agent → [tools → validator → agent]* → END

    Key design decisions:
    - ``interrupt_before=["tools"]`` enables Human-in-the-Loop (disambiguation,
      summarization type selection).
    - ``parallel_tool_calls`` is NOT passed to bind_tools — the custom model
      endpoint does not support this parameter and silently drops all tool_calls
      if it's present.
    - Parallel tool_calls are prevented at the Python layer in _orchestrate
      by keeping only the first call per turn.
    - Token and document_id are injected in _orchestrate, not in the prompt,
      to avoid prompt-injection and to keep the prompt model-agnostic.

    Attributes:
        MAX_ITERATIONS: Guard against infinite orchestration loops.
        EXECUTION_TIMEOUT: Per-invocation wall-clock timeout in seconds.
    """

    MAX_ITERATIONS: int = 10
    EXECUTION_TIMEOUT: float = 120.0

    def __init__(
        self,
        document_repo: Optional[IDocumentRepository] = None,
        semantic_dispatcher: Optional[SemanticDispatcher] = None,
    ) -> None:
        """
        Initializes the agent and compiles the LangGraph workflow.

        Args:
            document_repo: Document repository (DI; defaults to DocumentRepository).
            semantic_dispatcher: NLP dispatcher (DI; defaults to SemanticDispatcher).

        Raises:
            RuntimeError: If any component fails to initialize.
        """
        try:
            self.model = get_chat_model()
            self.tools = all_tools
            self.document_repo: IDocumentRepository = (
                document_repo or DocumentRepository()
            )
            self.dispatcher = semantic_dispatcher or SemanticDispatcher()
            self._checkpointer = MemorySaver()
            self._model_with_tools = self.model.bind_tools(self.tools)
            self._compiled_graph = self._build_graph()

            if self._compiled_graph is None:
                raise RuntimeError("Graph compilation returned None")

            self.state_manager = AgentStateManager(
                graph=self._compiled_graph,
                checkpointer=self._checkpointer,
            )

            logger.info(
                "EdmsDocumentAgent initialized",
                extra={
                    "tools_count": len(self.tools),
                    "model_type": type(self.model).__name__,
                },
            )
        except Exception as exc:
            logger.error("Failed to initialize EdmsDocumentAgent", exc_info=True)
            raise RuntimeError(f"Agent initialization failed: {exc}") from exc

    # ── Public API ────────────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, bool]:
        """
        Returns a shallow health status for each agent component.

        Returns:
            Dict mapping component names to boolean availability flags.
        """
        return {
            "model": self.model is not None,
            "tools": len(self.tools) > 0,
            "document_repo": self.document_repo is not None,
            "dispatcher": self.dispatcher is not None,
            "graph": getattr(self, "_compiled_graph", None) is not None,
            "state_manager": getattr(self, "state_manager", None) is not None,
            "checkpointer": getattr(self, "_checkpointer", None) is not None,
        }

    async def chat(
        self,
        message: str,
        user_token: str,
        context_ui_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None,
        human_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for agent interaction.

        Handles both fresh conversations and resumptions from Human-in-the-Loop
        interrupts (disambiguation, summarization type selection).

        Args:
            message: User message text.
            user_token: JWT authorization token.
            context_ui_id: UUID of the active EDMS document in the UI.
            thread_id: Conversation thread identifier (auto-generated if None).
            user_context: Optional user profile dict.
            file_path: UUID of an EDMS attachment or local filesystem path.
            human_choice: Disambiguation UUIDs (comma-separated) or summary type.

        Returns:
            Serialized AgentResponse dict suitable for the HTTP layer.
        """
        try:
            request = AgentRequest(
                message=message,
                user_token=user_token,
                context_ui_id=context_ui_id,
                thread_id=thread_id,
                user_context=user_context or {},
                file_path=file_path,
                human_choice=human_choice,
            )
            context = await self._build_context(request)

            # ── Автоматическое восстановление сломанного треда ───────────────
            if await self.state_manager.is_thread_broken(context.thread_id):
                repaired = await self.state_manager.repair_thread(context.thread_id)
                logger.warning(
                    "Broken thread detected and %s",
                    "repaired" if repaired else "repair FAILED",
                    extra={"thread_id": context.thread_id},
                )

            state = await self.state_manager.get_state(context.thread_id)

            # Если граф ждёт продолжения (interrupt_before) и пришёл human_choice
            if human_choice and state.next:
                return await self._handle_human_choice(context, human_choice)

            document: Optional[DocumentDto] = None
            if context.document_id:
                document = await self.document_repo.get_document(
                    context.user_token, context.document_id
                )

            semantic_ctx = self.dispatcher.build_context(request.message, document)
            logger.info(
                "Semantic analysis complete",
                extra={
                    "intent": semantic_ctx.query.intent.value,
                    "complexity": semantic_ctx.query.complexity.value,
                    "thread_id": context.thread_id,
                },
            )

            full_prompt = PromptBuilder.build(
                context,
                semantic_ctx.query.intent,
                self._build_semantic_xml(semantic_ctx),
            )

            inputs: Dict[str, Any] = {
                "messages": [
                    SystemMessage(content=full_prompt),
                    HumanMessage(content=semantic_ctx.query.refined),
                ]
            }

            return await self._orchestrate(
                context=context,
                inputs=inputs,
                is_choice_active=bool(human_choice),
                iteration=0,
            )

        except Exception as exc:
            logger.error(
                "Chat error",
                exc_info=True,
                extra={"user_message": message[:200]},
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка обработки запроса: {exc}",
            ).model_dump()

    # ── Human-in-the-Loop ─────────────────────────────────────────────────────

    async def _handle_human_choice(
        self, context: ContextParams, human_choice: str
    ) -> Dict[str, Any]:
        """
        Resumes a paused graph after the user resolves a disambiguation or
        selects a summarization type.

        Patches the pending AIMessage tool_calls with the user's choice,
        then resumes orchestration.

        Args:
            context: Immutable execution context.
            human_choice: Raw user choice: UUID list or summary type string.

        Returns:
            Serialized AgentResponse dict.
        """
        state = await self.state_manager.get_state(context.thread_id)
        last_msg: AIMessage = state.values["messages"][-1]
        raw_calls = getattr(last_msg, "tool_calls", [])

        patched_calls = []
        for tc in raw_calls:
            t_args = dict(tc["args"])
            t_name: str = tc["name"]

            if t_name == "doc_summarize_text":
                # Пользователь выбрал тип суммаризации
                t_args["summary_type"] = human_choice.strip()
                logger.info(
                    "Human choice: summary_type",
                    extra={"type": human_choice, "thread_id": context.thread_id},
                )

            elif t_name in _DISAMBIGUATION_TOOLS:
                # Пользователь выбрал сотрудника(ов) из disambiguation-списка.
                raw_ids = [x.strip() for x in human_choice.split(",") if x.strip()]
                valid_ids = []
                for raw_id in raw_ids:
                    try:
                        UUID(raw_id)  # валидация формата
                        valid_ids.append(raw_id)
                    except ValueError:
                        logger.warning(
                            "Invalid UUID in human_choice",
                            extra={"raw_id": raw_id},
                        )

                if valid_ids:
                    # Маппинг инструментов на их поле selected_ids
                    _TOOL_ID_FIELD: Dict[str, str] = {
                        "introduction_create_tool": "selected_employee_ids",
                        "task_create_tool": "selected_employee_ids",
                        "doc_create_resolution": "executor_ids",
                        "doc_send_notification": "recipient_ids",
                    }
                    id_field = _TOOL_ID_FIELD.get(t_name, "selected_employee_ids")
                    t_args[id_field] = valid_ids
                    t_args.pop("last_names", None)
                    t_args.pop("executor_last_names", None)
                    t_args.pop("recipient_last_names", None)
                    logger.info(
                        "Human choice: employee disambiguation resolved",
                        extra={
                            "tool": t_name,
                            "id_field": id_field,
                            "count": len(valid_ids),
                            "thread_id": context.thread_id,
                        },
                    )

            elif t_name == "doc_compare_with_local":
                # Пользователь выбрал вложение из disambiguation-списка.
                # human_choice должен содержать UUID вложения.
                choice = human_choice.strip()
                try:
                    UUID(choice)
                    t_args["attachment_id"] = choice
                    if context.document_id and not _is_valid_uuid(
                        str(t_args.get("document_id", "")).strip()
                    ):
                        t_args["document_id"] = context.document_id
                        logger.debug(
                            "Re-injected document_id for compare resume",
                            extra={"doc_id": context.document_id[:8]},
                        )
                    # Гарантируем local_file_path из контекста при resume
                    if context.file_path:
                        cur_local = str(t_args.get("local_file_path", "")).strip()
                        fp = str(context.file_path).strip()
                        if (
                            not cur_local
                            or cur_local.lower() in _COMPARE_LOCAL_PLACEHOLDERS
                        ):
                            t_args["local_file_path"] = fp
                            logger.debug(
                                "Re-injected local_file_path for compare resume",
                                extra={"path": fp[:32]},
                            )
                    # Гарантируем original_filename при resume.
                    if context.uploaded_file_name and not t_args.get(
                        "original_filename"
                    ):
                        t_args["original_filename"] = context.uploaded_file_name
                        logger.debug(
                            "Re-injected original_filename for compare resume",
                            extra={"file_name": context.uploaded_file_name},
                        )
                    logger.info(
                        "Human choice: attachment disambiguation resolved for compare",
                        extra={
                            "attachment_id": choice[:8] + "...",
                            "thread_id": context.thread_id,
                            "doc_id": str(t_args.get("document_id", "?"))[:8],
                        },
                    )
                except ValueError:
                    logger.warning(
                        "Invalid attachment UUID in human_choice for doc_compare_with_local",
                        extra={"raw_choice": choice},
                    )

            patched_calls.append({"name": t_name, "args": t_args, "id": tc["id"]})

        await self.state_manager.update_state(
            context.thread_id,
            [
                AIMessage(
                    content=last_msg.content or "",
                    tool_calls=patched_calls,
                    id=last_msg.id,
                )
            ],
            as_node="agent",
        )

        return await self._orchestrate(
            context=context,
            inputs=None,
            is_choice_active=True,
            iteration=0,
        )

    # ── Core orchestration loop ───────────────────────────────────────────────

    async def _orchestrate(
        self,
        context: ContextParams,
        inputs: Optional[Dict[str, Any]],
        is_choice_active: bool,
        iteration: int,
    ) -> Dict[str, Any]:
        """
        Core recursive orchestration loop.

        На каждой итерации:
        1. Вызывает граф (resume или fresh start)
        2. Читает последнее состояние
        3. Если граф завершён (END) → извлекает и возвращает финальный контент
        4. Если граф прерван перед tools → патчит tool_calls (инжект token, doc_id,
           attachment_id, summary_type) → сохраняет обратно → рекурсия

        Параллельные tool_calls блокируются здесь: из списка берётся только первый.
        Это безопаснее, чем полагаться на параметр модели, который может не поддерживаться.

        Args:
            context: Immutable execution context.
            inputs: Graph inputs (None = resume from interrupt).
            is_choice_active: True when resuming after human choice.
            iteration: Current recursion depth.

        Returns:
            Serialized AgentResponse dict.
        """
        if iteration > self.MAX_ITERATIONS:
            logger.error(
                "Max iterations exceeded",
                extra={"thread_id": context.thread_id, "iterations": iteration},
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышен лимит итераций обработки.",
            ).model_dump()

        try:
            await self.state_manager.invoke(
                inputs=inputs,
                thread_id=context.thread_id,
                timeout=self.EXECUTION_TIMEOUT,
            )

            state = await self.state_manager.get_state(context.thread_id)
            messages: List[BaseMessage] = state.values.get("messages", [])

            logger.debug(
                "State snapshot",
                extra={
                    "thread_id": context.thread_id,
                    "iteration": iteration,
                    "messages_count": len(messages),
                    "last_type": type(messages[-1]).__name__ if messages else "none",
                    "has_tool_calls": bool(
                        messages
                        and isinstance(messages[-1], AIMessage)
                        and getattr(messages[-1], "tool_calls", None)
                    ),
                    "state_next": list(state.next) if state.next else [],
                },
            )

            if not messages:
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Пустое состояние агента.",
                ).model_dump()

            last_msg = messages[-1]

            # ── Граф завершился (END) — нет pending nodes ──────────────────
            last_is_tool_msg = isinstance(last_msg, ToolMessage)
            last_has_tool_calls = isinstance(last_msg, AIMessage) and bool(
                getattr(last_msg, "tool_calls", None)
            )
            is_finished = (
                not state.next and not last_is_tool_msg and not last_has_tool_calls
            )
            if is_finished:
                return self._build_final_response(messages, context)

            # ── Граф прерван перед "tools" — патчим tool_calls ─────────────
            raw_calls = list(last_msg.tool_calls)

            # Защита от параллельных вызовов: берём только первый
            if len(raw_calls) > 1:
                logger.warning(
                    "Parallel tool_calls detected — keeping only the first",
                    extra={
                        "total": len(raw_calls),
                        "kept": raw_calls[0]["name"],
                        "dropped": [tc["name"] for tc in raw_calls[1:]],
                        "thread_id": context.thread_id,
                    },
                )
                raw_calls = raw_calls[:1]

            last_tool_text = ContentExtractor.extract_last_tool_text(messages)
            patched_calls = []

            for tc in raw_calls:
                t_name = tc["name"]
                t_args = dict(tc["args"])
                t_id = tc["id"]

                # ── 1. Инжект токена авторизации ──────────────────────────
                t_args["token"] = context.user_token

                # ── 1а. Инжект document_id ────────────────────────────────
                if context.document_id and t_name in _TOOLS_REQUIRING_DOCUMENT_ID:
                    cur_doc_id = str(t_args.get("document_id", "")).strip()
                    if not cur_doc_id or not _is_valid_uuid(cur_doc_id):
                        t_args["document_id"] = context.document_id
                        logger.debug(
                            "Injected document_id for tool '%s'",
                            t_name,
                            extra={"doc_id_prefix": context.document_id[:8]},
                        )

                # ── 2. Маршрутизация file_path → правильный инструмент ──────
                clean_path = str(context.file_path).strip() if context.file_path else ""
                path_is_uuid = _is_valid_uuid(clean_path)
                path_is_local = bool(clean_path) and not path_is_uuid

                if path_is_local:

                    # ── GUARD 1: doc_get_versions при наличии файла → БЛОК ────
                    if t_name == "doc_get_versions":
                        t_name = "doc_compare_with_local"
                        t_args = {
                            "local_file_path": clean_path,
                        }
                        if context.document_id:
                            t_args["document_id"] = context.document_id
                        logger.warning(
                            "GUARD: doc_get_versions blocked (local file present) "
                            "→ redirected to doc_compare_with_local",
                            extra={"path": clean_path[:32]},
                        )

                    # ── GUARD 2: doc_compare (версии) → compare_with_local ─────
                    elif t_name == "doc_compare":
                        t_name = "doc_compare_with_local"
                        t_args["local_file_path"] = clean_path
                        t_args.pop("document_id_1", None)
                        t_args.pop("document_id_2", None)
                        logger.warning(
                            "GUARD: doc_compare blocked (local file present) "
                            "→ redirected to doc_compare_with_local",
                            extra={"path": clean_path[:32]},
                        )

                    # ── GUARD 3: doc_get_file_content без attachment_id ────────
                    elif t_name == "doc_get_file_content" and not t_args.get(
                        "attachment_id"
                    ):
                        t_name = "read_local_file_content"
                        t_args["file_path"] = clean_path
                        t_args.pop("attachment_id", None)
                        logger.info(
                            "AUTO-PRIORITY: doc_get_file_content → read_local_file_content "
                            "(local file present, no explicit attachment_id)",
                            extra={"path": clean_path[:32]},
                        )

                if t_name == "doc_compare_with_local" and not clean_path:
                    t_name = "doc_compare"
                    t_args.pop("local_file_path", None)
                    t_args.pop("attachment_id", None)
                    logger.info(
                        "Routed doc_compare_with_local → doc_compare "
                        "(no file in context → version compare intended)",
                    )

                if path_is_local and t_name == "doc_get_file_content":
                    t_name = "read_local_file_content"
                    t_args["file_path"] = clean_path
                    t_args.pop("attachment_id", None)
                    logger.info(
                        "Routed doc_get_file_content → read_local_file_content",
                        extra={"path_prefix": clean_path[:32]},
                    )

                elif path_is_local and t_name == "read_local_file_content":
                    cur_fp = str(t_args.get("file_path", "")).strip()
                    if not cur_fp or cur_fp.lower() in (
                        "local_file",
                        "file_path",
                        "none",
                        "null",
                        "",
                    ):
                        t_args["file_path"] = clean_path
                        logger.info(
                            "Injected local file_path (placeholder replaced)",
                            extra={"path_prefix": clean_path[:32]},
                        )

                elif path_is_uuid and t_name == "read_local_file_content":
                    t_name = "doc_get_file_content"
                    t_args["attachment_id"] = clean_path
                    t_args.pop("file_path", None)
                    logger.info(
                        "Routed read_local_file_content → doc_get_file_content",
                        extra={"attachment_id": clean_path[:8]},
                    )

                elif path_is_uuid and t_name == "doc_get_file_content":
                    cur_att = str(t_args.get("attachment_id", "")).strip()
                    if not cur_att or not _is_valid_uuid(cur_att):
                        t_args["attachment_id"] = clean_path
                        logger.info(
                            "Injected attachment_id from context",
                            extra={"attachment_id": clean_path[:8]},
                        )

                # ── 2а. doc_compare_with_local: инжект local_file_path ─────────
                if t_name == "doc_compare_with_local" and path_is_local:
                    cur_local = str(t_args.get("local_file_path", "")).strip()
                    if (
                        not cur_local
                        or cur_local.lower() in _COMPARE_LOCAL_PLACEHOLDERS
                        or not Path(cur_local).exists()
                    ):
                        t_args["local_file_path"] = clean_path
                        logger.info(
                            "Force-injected local_file_path for doc_compare_with_local",
                            extra={"path": clean_path[:32]},
                        )

                    if context.uploaded_file_name and not t_args.get(
                        "original_filename"
                    ):
                        t_args["original_filename"] = context.uploaded_file_name
                        logger.debug(
                            "Injected original_filename for doc_compare_with_local",
                            extra={"file_name": context.uploaded_file_name},
                        )

                # ── 2б. doc_compare после doc_get_versions → БЛОК ────────────
                if t_name == "doc_compare":
                    _versions_result_complete = False
                    for prev_msg in reversed(messages):
                        if isinstance(prev_msg, ToolMessage):
                            try:
                                prev_data = json.loads(str(prev_msg.content))
                                if prev_data.get(
                                    "comparison_complete"
                                ) and prev_data.get("comparisons"):
                                    _versions_result_complete = True
                                    break
                            except (json.JSONDecodeError, AttributeError):
                                continue
                        if isinstance(prev_msg, HumanMessage):
                            break

                    if _versions_result_complete:
                        logger.warning(
                            "GUARD: doc_compare blocked — doc_get_versions already "
                            "completed all comparisons (comparison_complete=True). "
                            "Replacing with no-op to prevent redundant API call.",
                        )
                        t_name = "doc_get_details"
                        t_args = {}
                        if context.document_id:
                            t_args["document_id"] = context.document_id

                # ── 3. Инжект текста для суммаризации ─────────────────────
                if t_name == "doc_summarize_text":
                    if last_tool_text:
                        t_args["text"] = last_tool_text
                    if is_choice_active and not t_args.get("summary_type"):
                        t_args["summary_type"] = "extractive"
                        logger.warning(
                            "safety-net: summary_type=extractive "
                            "(is_choice_active but no type set)",
                        )

                patched_calls.append({"name": t_name, "args": t_args, "id": t_id})

            # Сохраняем AIMessage с исправленными tool_calls обратно в граф
            await self.state_manager.update_state(
                context.thread_id,
                [
                    AIMessage(
                        content=last_msg.content or "",
                        tool_calls=patched_calls,
                        id=last_msg.id,
                    )
                ],
                as_node="agent",
            )

            return await self._orchestrate(
                context=context,
                inputs=None,
                is_choice_active=is_choice_active,
                iteration=iteration + 1,
            )

        except asyncio.TimeoutError:
            logger.error(
                "Execution timeout",
                extra={
                    "thread_id": context.thread_id,
                    "timeout": self.EXECUTION_TIMEOUT,
                },
            )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышено время ожидания выполнения.",
            ).model_dump()

        except Exception as exc:
            err_str = str(exc)
            logger.error(
                "Orchestration error",
                exc_info=True,
                extra={"thread_id": context.thread_id, "iteration": iteration},
            )

            # ── Специфичная обработка ошибки несогласованного треда ──────────
            _BROKEN_THREAD_SIGNALS = (
                "tool_calls must be followed by tool messages",
                "tool_call_ids did not have response messages",
                "invalid_request_error",
                "messages.[",
            )
            is_broken_thread_error = any(
                sig in err_str for sig in _BROKEN_THREAD_SIGNALS
            )

            if is_broken_thread_error and iteration == 0:
                # Только на первой итерации — чтобы не зациклиться
                logger.warning(
                    "Broken thread error detected — attempting auto-repair",
                    extra={"thread_id": context.thread_id},
                )
                repaired = await self.state_manager.repair_thread(context.thread_id)
                if repaired:
                    return AgentResponse(
                        status=AgentStatus.ERROR,
                        message=(
                            "Предыдущий запрос завершился некорректно и был восстановлен. "
                            "Пожалуйста, повторите ваш вопрос."
                        ),
                    ).model_dump()

            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Ошибка оркестрации: {exc}",
            ).model_dump()

    # ── Graph compilation ─────────────────────────────────────────────────────

    def _build_graph(self) -> CompiledStateGraph:
        """
        Compiles the LangGraph ReAct workflow.

        Nodes:
        - ``agent``: invokes the LLM with bound tools
        - ``tools``: executes tool_calls via ToolNode
        - ``validator``: injects system notifications for tool errors

        Edges:
        - START → agent
        - agent → tools (if tool_calls present) | END
        - tools → validator → agent

        Interrupt:
        - ``interrupt_before=["tools"]`` pauses execution for Human-in-the-Loop
          (human choice injection in _orchestrate / _handle_human_choice)

        Returns:
            Compiled state graph ready for ainvoke/aget_state/aupdate_state.

        Raises:
            RuntimeError: If LangGraph compilation fails.
        """
        workflow: StateGraph = StateGraph(AgentState)

        # ── Нода: вызов LLM ──────────────────────────────────────────────────
        async def call_model(state: AgentState) -> Dict[str, Any]:
            """
            Invokes the LLM with bound tools.
            """
            _MAX_HISTORY_MSGS = 40
            sys_msgs = [m for m in state["messages"] if isinstance(m, SystemMessage)]
            non_sys = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
            if len(non_sys) > _MAX_HISTORY_MSGS:
                non_sys = non_sys[-_MAX_HISTORY_MSGS:]
            candidate_msgs = ([sys_msgs[-1]] if sys_msgs else []) + non_sys

            # ── Санация истории: убираем «висящие» tool_calls ─────────────────
            final_msgs = []
            for i, msg in enumerate(candidate_msgs):
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    # Проверяем: следующее сообщение должно быть ToolMessage
                    next_msg = (
                        candidate_msgs[i + 1] if i + 1 < len(candidate_msgs) else None
                    )
                    if not isinstance(next_msg, ToolMessage):
                        # Заменяем на «безопасную» копию без tool_calls
                        safe_msg = AIMessage(
                            content=msg.content or "",
                            id=msg.id,
                        )
                        final_msgs.append(safe_msg)
                        logger.warning(
                            "Sanitized dangling AIMessage tool_calls at position %d",
                            i,
                        )
                        continue
                final_msgs.append(msg)

            response = await self._model_with_tools.ainvoke(final_msgs)
            return {"messages": [response]}

        # ── Нода: валидация результатов инструментов ─────────────────────────
        async def validator(state: AgentState) -> Dict[str, Any]:
            """
            Post-tool validator: injects system notifications for failed or
            empty tool results as AIMessage with non-empty content.
            """
            last = state["messages"][-1]
            if not isinstance(last, ToolMessage):
                return {"messages": []}

            raw = str(last.content).strip()

            # Пустой результат инструмента
            if not raw or raw in ("None", "{}", "null"):
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "⚠️ Системное уведомление (не показывать пользователю): "
                                "Инструмент вернул пустой результат. "
                                "Попробуй другой подход или сообщи пользователю."
                            )
                        )
                    ]
                }

            if raw.startswith("{"):
                try:
                    tool_data = json.loads(raw)
                    interactive_status = tool_data.get("status", "")
                    if interactive_status in (
                        "requires_choice",
                        "requires_disambiguation",
                    ):
                        logger.info(
                            "Validator: interactive status '%s' — stopping graph",
                            interactive_status,
                        )
                        return {"messages": [AIMessage(content="")]}
                except json.JSONDecodeError:
                    pass

            # Явная ошибка от инструмента
            raw_lower = raw.lower()
            if '"status": "error"' in raw_lower or (
                raw_lower.startswith("{") and '"error"' in raw_lower
            ):
                try:
                    err_data = json.loads(raw)
                    err_msg = err_data.get("message", raw[:200])
                except (json.JSONDecodeError, KeyError):
                    err_msg = raw[:200]

                return {
                    "messages": [
                        AIMessage(
                            content=(
                                f"⚠️ Системное уведомление (не показывать пользователю): "
                                f"Инструмент вернул ошибку: {err_msg}. "
                                "Проинформируй пользователя понятным языком."
                            )
                        )
                    ]
                }

            return {"messages": []}

        # ── Регистрация нод и рёбер ───────────────────────────────────────────
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("validator", validator)
        workflow.add_edge(START, "agent")

        def should_continue(state: AgentState) -> str:
            """Routing function: go to tools if LLM produced tool_calls, else END."""
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools"
            return END

        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", END: END},
        )
        workflow.add_edge("tools", "validator")
        workflow.add_edge("validator", "agent")

        try:
            compiled = workflow.compile(
                checkpointer=self._checkpointer,
                interrupt_before=["tools"],
            )
            logger.debug("LangGraph compiled successfully")
            return compiled
        except Exception as exc:
            logger.error("Graph compilation failed", exc_info=True)
            raise RuntimeError(f"Failed to compile graph: {exc}") from exc

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_final_response(
        self, messages: List[BaseMessage], context: ContextParams
    ) -> Dict[str, Any]:
        """
        Extracts final content and wraps it into an AgentResponse.

        Before extracting text content, scans the last ToolMessage for
        structured interactive statuses:
        - ``requires_choice``       : Summarisation format selection needed.
        - ``requires_disambiguation``: Attachment or employee disambiguation needed.

        These statuses bypass text extraction and are returned directly to
        the HTTP layer so the frontend can render the appropriate widget.

        Args:
            messages: Complete LangGraph message chain.
            context: Execution context (thread_id, file info for sanitization).

        Returns:
            Serialized AgentResponse dict.
        """
        # ── Проверяем ToolMessage на интерактивные статусы ──────────────────
        interactive = self._detect_interactive_status(messages)
        if interactive:
            logger.info(
                "Interactive status detected",
                extra={
                    "status": interactive.get("status"),
                    "thread_id": context.thread_id,
                },
            )
            return interactive

        # ── Стандартное извлечение финального текста ────────────────────────
        final_content = ContentExtractor.extract_final_content(messages)

        if final_content:
            final_content = ContentExtractor.clean_json_artifacts(final_content)
            final_content = self._sanitize_technical_content(final_content, context)
            reload_needed = _is_mutation_response(final_content)

            logger.info(
                "Execution completed",
                extra={
                    "thread_id": context.thread_id,
                    "content_length": len(final_content),
                    "requires_reload": reload_needed,
                },
            )
            return AgentResponse(
                status=AgentStatus.SUCCESS,
                content=final_content,
                requires_reload=reload_needed,
            ).model_dump()

        logger.warning("No final content found", extra={"thread_id": context.thread_id})
        return AgentResponse(
            status=AgentStatus.SUCCESS,
            content="Операция завершена.",
        ).model_dump()

    @staticmethod
    def _detect_interactive_status(
        messages: List[BaseMessage],
    ) -> Optional[Dict[str, Any]]:
        """
        Сканирует ПОСЛЕДНИЙ ToolMessage на наличие статусов интерактива.

        Статусы 'requires_choice' или 'requires_disambiguation' означают, что
        агенту требуется ввод пользователя для продолжения. Если последний
        ToolMessage не содержит этих статусов, значит инструмент вернул данные
        и мы ждем финального ответа LLM.

        Args:
            messages: Полная цепочка сообщений LangGraph.

        Returns:
            Сериализованный словарь AgentResponse или None.
        """
        # Поиск последнего ToolMessage в цепочке
        last_tool_msg: Optional[ToolMessage] = None
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                last_tool_msg = m
                break

        if last_tool_msg is None:
            return None

        raw = str(last_tool_msg.content).strip()
        if not raw.startswith("{"):
            return None

        try:
            data: Dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            return None

        status = data.get("status", "")
        if status not in ("requires_choice", "requires_disambiguation"):
            return None

        logger.info(
            "Detect interactive: status=%s keys=%s payload_preview=%s",
            status,
            list(data.keys()),
            raw[:1000],
        )

        # ─── REQUIRES_CHOICE: Выбор формата суммаризации ───────────────────
        if status == "requires_choice":
            options = data.get("options", [])
            hint = data.get("hint", "extractive")
            hint_reason = data.get("hint_reason", "")
            msg = data.get("message", "Выберите формат анализа:")

            options_lines = [
                f"- **{opt['key']}** — {opt['label']}: {opt['description']}"
                for opt in options
                if isinstance(opt, dict)
            ]
            options_text = "\n".join(options_lines)
            hint_text = (
                f"\n\n💡 *Рекомендация:* **{hint}** — {hint_reason}"
                if hint_reason
                else ""
            )
            full_message = (
                f"{msg}\n\n{options_text}{hint_text}\n\n"
                "Ответьте: **extractive**, **abstractive** или **thesis**."
            )
            return AgentResponse(
                status=AgentStatus.REQUIRES_ACTION,
                action_type=ActionType.SUMMARIZE_SELECTION,
                message=full_message,
            ).model_dump()

        # ─── REQUIRES_DISAMBIGUATION: Уточнение объекта (сотрудник/файл) ────
        if status == "requires_disambiguation":
            _KNOWN_LIST_KEYS = (
                "available_attachments",
                "available_employees",
                "candidates",
                "employees",
                "results",
                "items",
                "users",
            )

            # Поиск списка доступных вариантов
            available: List[Dict[str, Any]] = next(
                (
                    v
                    for k in _KNOWN_LIST_KEYS
                    if isinstance(v := data.get(k), list) and v
                ),
                [],
            )

            if not available:
                for _k, _v in data.items():
                    if _k != "options" and isinstance(_v, list) and _v:
                        first_item = _v[0] if _v else {}
                        if (
                            isinstance(first_item, dict)
                            and "matches" in first_item
                            and not first_item.get("id")
                        ):
                            # Nested format — разворачиваем
                            flat: List[Dict[str, Any]] = []
                            for group in _v:
                                flat.extend(group.get("matches", []))
                            if flat:
                                available = flat
                                logger.info(
                                    "Disambiguation: unwrapped nested 'matches' "
                                    "from key=%s total=%d",
                                    _k,
                                    len(flat),
                                )
                                break
                        else:
                            available = _v
                            logger.info(
                                "Disambiguation list found via fallback key=%s len=%d",
                                _k,
                                len(_v),
                            )
                            break

            base_msg: str = data.get("message", "Уточните выбор:")
            base_msg = (
                re.sub(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                    "",
                    base_msg,
                    flags=re.I,
                )
                .strip()
                .rstrip("с «»")
                .strip()
            )
            if not base_msg:
                base_msg = "Уточните выбор:"
            candidates_structured: List[Dict[str, str]] = []

            for item in available:
                if not isinstance(item, dict):
                    continue

                # Извлечение ФИО (поддержка разных схем именования)
                first = (
                    item.get("firstName")
                    or item.get("first_name")
                    or item.get("firstname")
                    or item.get("givenName")
                    or ""
                ).strip()
                last = (
                    item.get("lastName")
                    or item.get("last_name")
                    or item.get("lastname")
                    or item.get("surname")
                    or item.get("familyName")
                    or ""
                ).strip()
                middle = (
                    item.get("middleName")
                    or item.get("middle_name")
                    or item.get("patronymic")
                    or ""
                ).strip()

                display_name = (
                    item.get("fullName")
                    or item.get("full_name")
                    or item.get("fio")
                    or item.get("FIO")
                    or " ".join(filter(None, [last, first, middle]))
                    or item.get("name")
                    or item.get("username")
                    or item.get("login")
                    or item.get("email", "").split("@")[0]
                    or "Без имени"
                ).strip()

                dept = (
                    item.get("department")
                    or item.get("departmentName")
                    or item.get("department_name")
                    or item.get("division")
                    or item.get("post")
                    or item.get("position")
                    or item.get("jobTitle")
                    or item.get("job_title")
                    or item.get("role")
                    or ""
                ).strip()

                item_id = str(
                    item.get("id")
                    or item.get("uuid")
                    or item.get("employeeId")
                    or item.get("employee_id")
                    or item.get("userId")
                    or item.get("user_id")
                    or item.get("personId")
                    or item.get("person_id")
                    or "?"
                )

                candidates_structured.append(
                    {
                        "id": item_id,
                        "name": display_name,
                        "dept": dept,
                    }
                )

            candidates_json = json.dumps(candidates_structured, ensure_ascii=False)
            # Маркер <!--CANDIDATES:[...]---> парсится в AssistantWidget.tsx
            full_msg = f"{base_msg}\n\n<!--CANDIDATES:{candidates_json}-->"

            return AgentResponse(
                status=AgentStatus.REQUIRES_ACTION,
                action_type=ActionType.DISAMBIGUATION,
                message=full_msg,
            ).model_dump()

        return None

    async def _build_context(self, request: AgentRequest) -> ContextParams:
        """
        Builds an immutable ContextParams from a validated AgentRequest.

        Args:
            request: Validated agent request.

        Returns:
            Fully populated ContextParams instance.
        """
        ctx = request.user_context
        first_name: str = (ctx.get("firstName") or ctx.get("first_name") or "").strip()
        last_name: str = (ctx.get("lastName") or ctx.get("last_name") or "").strip()
        full_name: str = (
            ctx.get("fullName") or ctx.get("full_name") or ctx.get("name") or ""
        ).strip()
        user_id: Optional[str] = (
            ctx.get("id") or ctx.get("userId") or ctx.get("user_id")
        )

        display_name: str = first_name or last_name or full_name or "пользователь"

        return ContextParams(
            user_token=request.user_token,
            document_id=request.context_ui_id,
            file_path=request.file_path,
            thread_id=request.thread_id or "default",
            user_name=display_name,
            user_first_name=first_name or None,
            user_last_name=last_name or None,
            user_full_name=full_name or None,
            user_id=user_id,
        )

    def _sanitize_technical_content(self, content: str, context: ContextParams) -> str:
        """
        Removes technical artifacts from user-visible response content.

        Применяет замены в строго определённом порядке, чтобы избежать
        артефактов от частичной замены составных имён temp-файлов.

        Порядок замен:
        1. Абсолютные пути (/tmp/..., C:\\...) → original filename label
        2. Составное имя UUID_hex32.ext (полный паттерн temp-файла) → filename label
        3. hex32.ext без UUID-prefix → filename label
        4. UUID с дефисами (оставшиеся) → «документ»
        5. UUID без дефисов — 32 hex chars → «документ»
        6. Финальная очистка артефактов «документ»«...» → «...»

        Args:
            content: Raw extracted response content.
            context: Execution context with file_path and uploaded_file_name.

        Returns:
            Sanitized content string safe for display to the user.
        """
        file_label = (
            f"«{context.uploaded_file_name}»"
            if context.uploaded_file_name
            else "«загруженный файл»"
        )

        # 1. Абсолютные пути файловой системы
        content = re.sub(
            r"[A-Za-z]:\\[^\s,;)'\"]{3,}|/(?:tmp|var|home|uploads)/[^\s,;)'\"]{3,}",
            file_label,
            content,
        )

        # 2. Составное имя: UUID-с-дефисами_hex32.ext (полный temp-файл паттерн)
        content = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
            r"_[0-9a-f]{32}\.[a-zA-Z]{2,5}",
            file_label,
            content,
            flags=re.I,
        )

        # 3. hex32.ext (с опциональным ведущим _) — частичный temp-файл паттерн
        content = re.sub(
            r"_?[0-9a-f]{32}\.[a-zA-Z]{2,5}\b",
            file_label,
            content,
            flags=re.I,
        )

        # 4. Конкретный UUID загруженного файла или документа (с дефисами)
        if context.file_path and _is_valid_uuid(str(context.file_path).strip()):
            content = content.replace(str(context.file_path).strip(), file_label)
        if context.document_id and _is_valid_uuid(context.document_id):
            content = content.replace(context.document_id, "«текущего документа»")

        # 5. Оставшиеся UUID с дефисами
        content = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "«документ»",
            content,
            flags=re.I,
        )

        # 6. UUID без дефисов — 32 hex chars (uuid4().hex)
        content = re.sub(
            r"(?<![a-zA-Z0-9])[0-9a-f]{32}(?![a-zA-Z0-9])",
            "«документ»",
            content,
            flags=re.I,
        )

        # 7. Артефакты вида "«документ»«имя файла»" → "«имя файла»"
        content = re.sub(r"«документ»\s*(?=«)", "", content)
        content = re.sub(r"«документ»_\s*", "", content)

        return content

    @staticmethod
    def _build_semantic_xml(semantic_context: Any) -> str:
        """
        Serializes semantic context into an XML block for prompt injection.

        Args:
            semantic_context: Output of SemanticDispatcher.build_context().

        Returns:
            XML string block appended to the system prompt.
        """
        return (
            "\n<semantic_context>\n"
            f"  <intent>{semantic_context.query.intent.value}</intent>\n"
            f"  <complexity>{semantic_context.query.complexity.value}</complexity>\n"
            f"  <original>{semantic_context.query.original}</original>\n"
            f"  <refined>{semantic_context.query.refined}</refined>\n"
            "</semantic_context>"
        )

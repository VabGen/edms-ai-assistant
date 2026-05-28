# edms_ai_assistant/tools/create_document_from_file.py
"""
EDMS AI Assistant — Create Document From File Tool (DI Factory).

Сценарий использования:
  Пользователь из любого места интерфейса загружает файл через скрепку
  и просит: «Создай обращение на основе этого файла».

  Инструмент:
    1. Находит активный профиль нужной категории (APPEAL, INCOMING, и т.д.)
    2. Создаёт новый пустой документ из профиля (POST /api/document)
    3. Загружает файл как основное вложение (POST /api/document/{id}/attachment)
    4. Для APPEAL — автозаполняет карточку через appeal_autofill_service
    5. Возвращает URL для навигации: /document-form/{new_document_id}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.agent.runnable_utils import get_token_from_config
from edms_ai_assistant.utils.regex_utils import UUID_RE

if TYPE_CHECKING:
    from edms_ai_assistant.clients.document_creator_client import DocumentCreatorClient
    from edms_ai_assistant.core.deps import AppDeps

logger = logging.getLogger(__name__)

_AUTOFILL_SUPPORTED: frozenset[str] = frozenset({"APPEAL"})

_VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "APPEAL",
        "INCOMING",
        "OUTGOING",
        "INTERN",
        "CONTRACT",
        "MEETING",
        "MEETING_QUESTION",
        "QUESTION",
        "CUSTOM",
    }
)

_CATEGORY_NAMES_RU: dict[str, str] = {
    "APPEAL": "обращение",
    "INCOMING": "входящий документ",
    "OUTGOING": "исходящий документ",
    "INTERN": "внутренний документ",
    "CONTRACT": "договор",
    "MEETING": "совещание",
    "MEETING_QUESTION": "вопрос повестки",
    "QUESTION": "вопрос",
    "CUSTOM": "произвольный документ",
}

_CATEGORY_KEYWORDS: list[tuple[list[str], str]] = [
    (["обращени", "жалоб", "заявлени", "appeal"], "APPEAL"),
    (["входящ", "incoming"], "INCOMING"),
    (["исходящ", "outgoing"], "OUTGOING"),
    (["внутренн", "intern"], "INTERN"),
    (["договор", "контракт", "contract"], "CONTRACT"),
    (["совещани", "meeting"], "MEETING"),
    (["повестк", "meeting_question"], "MEETING_QUESTION"),
]


def _extract_category_from_message(message: str) -> str | None:
    """Extract document category from free-form user message text."""
    lower = message.lower()
    for keywords, category in _CATEGORY_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return category
    return None


class CreateDocumentFromFileInput(BaseModel):
    """Validated input schema for create_document_from_file tool."""

    file_path: str = Field(
        ...,
        description=(
            "Путь к локальному файлу (из контекста агента) или UUID вложения EDMS. "
            "Инжектируется автоматически из <local_file_path> в system prompt."
        ),
    )
    doc_category: str = Field(
        "APPEAL",
        description=(
            "Категория создаваемого документа. Допустимые значения: "
            "APPEAL (обращение), INCOMING (входящий), OUTGOING (исходящий), "
            "INTERN (внутренний), CONTRACT (договор), MEETING (совещание), "
            "QUESTION (вопрос), CUSTOM (произвольный)."
        ),
    )
    file_name: str | None = Field(
        None,
        description="Имя файла для отображения в карточке документа (опционально).",
        max_length=260,
    )
    autofill: bool = Field(
        True,
        description=(
            "Автозаполнить карточку документа на основе содержимого файла. "
            "По умолчанию True. Применяется для APPEAL (обращения)."
        ),
    )

    @field_validator("doc_category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        normalized = v.strip().upper()
        if normalized not in _VALID_CATEGORIES:
            raise ValueError(
                f"Неизвестная категория документа: '{v}'. "
                f"Допустимые: {', '.join(sorted(_VALID_CATEGORIES))}"
            )
        return normalized

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("file_path не может быть пустым")
        lower = cleaned.lower()
        if "file_path" in lower or "<" in cleaned or ">" in cleaned:
            raise ValueError(
                "Передан плейсхолдер вместо реального пути. "
                "Используй значение из <local_file_path> в system prompt."
            )
        return cleaned


# ══════════════════════════════════════════════════════════════════════════════
# Tool Factory
# ══════════════════════════════════════════════════════════════════════════════


def create_document_from_file_tool(deps: AppDeps) -> StructuredTool:
    """Фабрика инструмента создания документа из файла с DI."""

    creator_client: DocumentCreatorClient = deps.document_creator_client
    autofill_service = deps.appeal_autofill_service

    async def create_document_from_file(
        file_path: str,
        doc_category: str = "APPEAL",
        file_name: str | None = None,
        autofill: bool = True,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Create a new EDMS document from a local file and open it in the browser.

        Use this tool when the user:
        - Attaches a file (via the paperclip) and asks to create a document from it.
        - Asks: «Создай обращение на основе этого файла», «Сделай входящий по этому письму».
        - Asks to create ANY type of document based on an uploaded file.

        Workflow:
        1. Find the active document profile for the requested category.
        2. Create an empty document via POST /api/document.
        3. Upload the file as the main attachment.
        4. For APPEAL — auto-fill the document card from the file content.
        5. Return navigate_url = /document-form/{new_document_id}.

        ВАЖНО: Токен авторизации передаётся системой АВТОМАТИЧЕСКИ через config.
        НЕ запрашивай его у пользователя и НЕ передавай в аргументах.
        """
        try:
            token = get_token_from_config(config)
        except Exception as e:
            logger.error(
                "Failed to get token from config: %s | config keys: %s",
                e,
                (
                    list((config or {}).get("configurable", {}).keys())
                    if config
                    else "None"
                ),
            )
            return {
                "status": "error",
                "message": f"Ошибка авторизации: токен не найден. {e}",
            }

        # ── Авто-определение категории из текста сообщения ───────────────────────
        _explicit_category = doc_category
        if doc_category == "APPEAL" and file_name:
            _detected = _extract_category_from_message(file_name)
            if _detected and _detected != "APPEAL":
                doc_category = _detected
                logger.info(
                    "doc_category refined from file_name '%s': %s -> %s",
                    file_name[:40],
                    _explicit_category,
                    doc_category,
                )

        category_ru = _CATEGORY_NAMES_RU.get(doc_category, doc_category.lower())

        logger.info(
            "create_document_from_file called",
            extra={
                "doc_category": doc_category,
                "explicit_category": _explicit_category,
                "file_path": file_path,
                "autofill": autofill,
            },
        )

        warnings: list[str] = []

        # ── 1. Определяем источник файла ─────────────────────────────────────────
        is_uuid = bool(UUID_RE.match(file_path.strip()))
        effective_local_path: str | None = None

        if is_uuid:
            return {
                "status": "error",
                "message": (
                    "Создание документа из UUID вложения EDMS пока не поддерживается. "
                    "Пожалуйста, загрузите файл локально через скрепку и повторите запрос."
                ),
            }
        else:
            effective_local_path = file_path
            if not Path(effective_local_path).exists():
                return {
                    "status": "error",
                    "message": (
                        f"Файл не найден: '{file_path}'. "
                        "Пожалуйста, загрузите файл через кнопку со скрепкой и повторите."
                    ),
                }

        effective_file_name = file_name or Path(effective_local_path).name

        # ── 2. Ищем профиль документа ─────────────────────────────────────────
        profile = await creator_client.find_profile_by_category(token, doc_category)
        if not profile:
            return {
                "status": "error",
                "message": (
                    f"Не найден активный профиль для категории «{category_ru}». "
                    "Возможно, у вас нет прав на создание документов данного типа "
                    "или профиль не настроен в системе."
                ),
            }

        profile_id: str = str(profile.get("id", ""))
        profile_name: str = profile.get("name", "?")
        logger.info("Using profile '%s' (%s…)", profile_name, profile_id[:8])

        # ── 3. Создаём документ ───────────────────────────────────────────────
        created = await creator_client.create_document(token, profile_id)
        if not created:
            return {
                "status": "error",
                "message": (
                    f"Не удалось создать документ по профилю «{profile_name}». "
                    "Проверьте права доступа."
                ),
            }

        document_id: str = str(created.document.id)

        if not document_id:
            return {
                "status": "error",
                "message": "Документ создан, но сервер не вернул его UUID.",
            }

        # ── 4. Загружаем файл как вложение ────────────────────────────────────
        attachment = await creator_client.upload_attachment(
            token=token,
            document_id=document_id,
            file_path=effective_local_path,
            file_name=effective_file_name,
        )

        if attachment is None:
            warnings.append(
                f"Файл '{effective_file_name}' не найден на сервере — вложение не загружено."
            )
            logger.warning(
                "Attachment upload skipped (file not found): %s", effective_local_path
            )

        # ── 5. Автозаполнение карточки ────────────────────────────────────────
        autofill_status: str = "skipped"

        if autofill and doc_category in _AUTOFILL_SUPPORTED:
            try:
                autofill_result = await autofill_service.process_and_fill(
                    token=token,
                    document_id=document_id,
                    attachment_id=None,
                    generate_summary_choices=False,
                )

                if autofill_result.status == "success":
                    autofill_status = "done"
                    if autofill_result.warnings:
                        warnings.extend(autofill_result.warnings)
                    logger.info("Autofill completed for document %s…", document_id[:8])
                else:
                    autofill_status = "failed"
                    warnings.append(
                        f"Автозаполнение завершилось с ошибкой: {autofill_result.message}"
                    )
                    logger.warning(
                        "Autofill failed for document %s: %s",
                        document_id[:8],
                        autofill_result.message,
                    )

            except Exception as exc:
                autofill_status = "failed"
                warnings.append(f"Автозаполнение не удалось: {exc!s}")
                logger.error(
                    "Autofill failed for document %s: %s",
                    document_id[:8],
                    exc,
                    exc_info=True,
                )

        elif autofill and doc_category not in _AUTOFILL_SUPPORTED:
            autofill_status = "not_supported"
            logger.info(
                "Autofill not supported for category '%s' — skipped", doc_category
            )

        # ── 6. Формируем ответ ────────────────────────────────────────────────
        navigate_url = f"/document-form/{document_id}"

        parts: list[str] = [f"✅ {category_ru.capitalize()} успешно создано."]
        if attachment is not None:
            parts.append(f"📎 Файл «{effective_file_name}» прикреплён как вложение.")
        if autofill_status == "done":
            parts.append("📋 Карточка документа заполнена автоматически.")
        elif autofill_status == "failed":
            parts.append(
                "⚠️ Карточку не удалось заполнить автоматически — проверьте и заполните вручную."
            )
        parts.append("🔗 Открываю документ в системе...")

        status = "success" if not warnings else "partial_success"

        logger.info(
            "create_document_from_file completed: doc=%s… status=%s autofill=%s",
            document_id[:8],
            status,
            autofill_status,
        )

        return {
            "status": status,
            "message": " ".join(parts),
            "document_id": document_id,
            "navigate_url": navigate_url,
            "autofill_status": autofill_status,
            "warnings": warnings if warnings else None,
            "requires_reload": False,
        }

    return StructuredTool.from_function(
        coroutine=create_document_from_file,
        name="create_document_from_file",
        description=(
            "Создает новый документ EDMS из локального файла и открывает его в браузере.\n"
            "Используй этот инструмент, когда пользователь:\n"
            "- Прикрепляет файл (через скрепку) и просит создать документ из него.\n"
            "- Просит: «Создай обращение на основе этого файла», «Сделай входящий по этому письму».\n\n"
            "Workflow:\n"
            "1. Находит активный профиль документа для запрошенной категории.\n"
            "2. Создает пустой документ.\n"
            "3. Загружает файл как основное вложение.\n"
            "4. Для APPEAL — автозаполняет карточку из содержимого файла.\n"
            "5. Возвращает navigate_url = /document-form/{new_document_id}.\n\n"
            "ВАЖНО: Токен авторизации передаётся системой АВТОМАТИЧЕСКИ. "
            "НЕ запрашивай его у пользователя."
        ),
        args_schema=CreateDocumentFromFileInput,
    )

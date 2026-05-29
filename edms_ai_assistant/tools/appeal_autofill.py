# edms_ai_assistant/tools/appeal_autofill.py
"""
EDMS AI Assistant — Appeal Autofill Tool (DI Factory).

Инструмент автозаполнения обращений.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.tools import InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.agent.runnable_utils import (
    get_document_id_from_config,
    get_token_from_config,
)
from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from edms_ai_assistant.core.deps import AppDeps

logger = logging.getLogger(__name__)


class AppealAutofillInput(BaseModel):
    """Схема ввода для автозаполнения обращения."""

    attachment_id: str | None = Field(
        None,
        description="UUID конкретного вложения для анализа",
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    )
    generate_summary_choices: bool = Field(
        False,
        description=(
            "Если True — после заполнения возвращает 3 варианта заголовка (shortSummary) "
            "для выбора пользователем. Используй когда пользователь просит "
            "выбрать или изменить заголовок документа."
        ),
    )

    @field_validator("attachment_id")
    @classmethod
    def validate_attachment_id(cls, v: str | None) -> str | None:
        if v and not v.strip():
            return None
        return v


def create_appeal_autofill_tool(deps: AppDeps) -> StructuredTool:
    """Фабрика инструмента автозаполнения обращения с DI."""

    async def autofill_appeal_document(
        attachment_id: str | None = None,
        generate_summary_choices: bool = False,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Автоматически заполняет карточку обращения (APPEAL) через LLM-анализ вложенного документа.

        ВЫЗЫВАЙ ЭТОТ ИНСТРУМЕНТ КОГДА:
        - Пользователь просит «создать обращение», «заполнить обращение»,
          «автозаполнение обращения», «заполнить карточку обращения»
        - Пользователь просит «проанализировать обращение» и открыт документ APPEAL
        - Пользователь говорит «заполни документ» для документа категории APPEAL
        - Нужно извлечь данные из вложения и заполнить поля обращения

        НЕ вызывай для других категорий (INCOMING, OUTGOING и т.д.).

        Токен авторизации и ID документа передаются системой АВТОМАТИЧЕСКИ.
        НЕ запрашивай их у пользователя и НЕ передавай в аргументах.
        """
        try:
            token = get_token_from_config(config)
            document_id = get_document_id_from_config(config)
        except Exception as e:
            logger.error("Failed to get token/document_id from config: %s", e)
            return {
                "status": "error",
                "message": f"Ошибка авторизации или контекста документа: {e}",
            }

        logger.info(
            "========== APPEAL AUTOFILL TOOL START ==========",
            extra={"document_id": document_id},
        )

        try:
            result = await deps.appeal_autofill_service.process_and_fill(
                token=token,
                document_id=document_id,
                attachment_id=attachment_id,
                generate_summary_choices=generate_summary_choices,
            )

            logger.info("========== APPEAL AUTOFILL TOOL SUCCESS ==========")
            return result.to_dict()

        except ValueError as e:
            logger.warning("Validation error in autofill: %s", e)
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(
                "========== APPEAL AUTOFILL TOOL ERROR ==========",
                exc_info=True,
            )
            return {"status": "error", "message": f"Ошибка автозаполнения: {e!s}"}

    return StructuredTool.from_function(
        coroutine=autofill_appeal_document,
        name="autofill_appeal_document",
        description=(
            "Автоматически заполняет карточку обращения (APPEAL) через LLM-анализ вложенного документа.\n"
            "ВЫЗЫВАЙ ЭТОТ ИНСТРУМЕНТ КОГДА:\n"
            "- Пользователь просит «создать обращение», «заполнить обращение», "
            "«автозаполнение обращения», «заполнить карточку обращения»\n"
            "- Пользователь просит «проанализировать обращение» и открыт документ APPEAL\n"
            "- Нужно извлечь данные из вложения и заполнить поля обращения\n\n"
            "НЕ вызывай для других категорий (INCOMING, OUTGOING и т.д.).\n"
            "Токен авторизации и ID документа передаются системой АВТОМАТИЧЕСКИ. "
            "НЕ запрашивай их у пользователя."
        ),
        args_schema=AppealAutofillInput,
    )

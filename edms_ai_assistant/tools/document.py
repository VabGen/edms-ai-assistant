# edms_ai_assistant/tools/document.py
"""
EDMS AI Assistant — doc_get_details tool.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel

from edms_ai_assistant.agent.runnable_utils import (
    get_document_id_from_config,
    get_token_from_config,
)
from edms_ai_assistant.clients.redis_client import get_redis
from edms_ai_assistant.services.document_service import (
    DocumentNotFoundError,
    DocumentService,
)
from edms_ai_assistant.utils.format_utils import clean_dict

logger = logging.getLogger(__name__)


class DocDetailsInput(BaseModel):
    """Input schema for doc_get_details tool."""
    pass


@tool("doc_get_details", args_schema=DocDetailsInput)
async def doc_get_details(config: RunnableConfig) -> dict[str, Any]:
    """Анализирует текущий открытый документ СЭД и все его вложенные сущности.

    Возвращает полный семантически структурированный контекст:
    основные данные, регистрацию, участников, жизненный цикл, контроль,
    задачи, вложения, адресатов, контрагента, ознакомления и
    специализированные секции (договор / обращение / совещание / повестка).

    Использует Redis-кэш — повторный вызов для того же документа
    не делает запрос к Java API.

    ВАЖНО: Токен авторизации и ID документа передаются системой АВТОМАТИЧЕСКИ.
    Тебе НЕ НУЖНО запрашивать их у пользователя или передавать в аргументах.

    Args:
        config: LangGraph RunnableConfig (инжектится фреймворком, содержит token и document_id).

    Returns:
        Dict со статусом и полным NLP-анализом документа.
    """
    try:
        token = get_token_from_config(config)
        document_id = get_document_id_from_config(config)
    except RuntimeError as exc:
        logger.error("Missing context in tool call: %s", exc)
        return {"status": "error", "error": str(exc)}

    try:
        svc = DocumentService(redis=get_redis())
        analysis = await svc.get_document_analysis(
            token=token,
            document_id=document_id,
        )
        return {"status": "success", "document_analytics": clean_dict(analysis)}

    except DocumentNotFoundError as exc:
        logger.warning(
            "Document not found in doc_get_details",
            extra={"document_id": document_id, "error": str(exc)},
        )
        return {"status": "error", "error": f"Документ {document_id} не найден."}

    except Exception as exc:
        logger.error(
            "doc_get_details failed",
            exc_info=True,
            extra={"document_id": document_id},
        )
        return {"status": "error", "error": f"Ошибка обработки документа: {exc}"}
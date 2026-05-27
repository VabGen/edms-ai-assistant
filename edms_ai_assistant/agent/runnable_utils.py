# edms_ai_assistant/agent/runnable_utils.py
"""Helpers for LangGraph RunnableConfig."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


def get_token_from_config(config: RunnableConfig | None) -> str:
    """Извлекает user_token из LangGraph RunnableConfig.

    Токен прокидывается из FastAPI слоя через _make_config().
    """
    if not config or not isinstance(config, dict):
        raise RuntimeError(
            "RunnableConfig is missing or invalid. "
            "Cannot extract authorization token."
        )

    configurable = config.get("configurable", {})
    token = configurable.get("user_token")

    if not token or not isinstance(token, str) or not token.strip():
        logger.error(
            "Token is missing or empty in RunnableConfig.configurable. "
            "Available keys: %s",
            list(configurable.keys()),
        )
        raise RuntimeError(
            "Authorization token not found in RunnableConfig. "
            "Ensure the API route passes it via _make_config()."
        )

    return token.strip()


def get_document_id_from_config(config: RunnableConfig) -> str:
    """Извлекает document_id из LangGraph RunnableConfig.

    ID документа прокидывается из UI (context_ui_id) через _make_config().
    """
    if not config or not isinstance(config, dict):
        raise RuntimeError(
            "RunnableConfig is missing or invalid. " "Cannot extract document_id."
        )

    doc_id = config.get("configurable", {}).get("document_id")
    if not doc_id:
        raise RuntimeError(
            "document_id not found in RunnableConfig. "
            "User must have an active document open in EDMS UI to use this tool."
        )
    return doc_id

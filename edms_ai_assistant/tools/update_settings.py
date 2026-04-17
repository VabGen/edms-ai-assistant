"""
Tool for AI Agent to modify runtime settings via chat.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from edms_ai_assistant.api.routes.settings import _store, UpdateSettingsRequest

logger = logging.getLogger(__name__)


class UpdateSettingsInput(BaseModel):
    """Input schema for the update_settings tool."""
    llm: dict[str, Any] | None = Field(
        None,
        description="Словарь с настройками LLM (generative_url, generative_model, temperature, max_tokens). Ключи в snake_case."
    )
    agent: dict[str, Any] | None = Field(
        None,
        description="Словарь с настройками Агента (max_iterations, max_context_messages)."
    )
    rag: dict[str, Any] | None = Field(
        None,
        description="Словарь с настройками RAG (chunk_size, chunk_overlap)."
    )
    edms: dict[str, Any] | None = Field(
        None,
        description="Словарь с настройками EDMS (base_url, timeout)."
    )


class UpdateSettingsTool(BaseTool):
    """Инструмент для изменения технических настроек системы в реальном времени."""
    name: str = "update_runtime_settings"
    description: str = (
        "Используй этот инструмент, если пользователь просит изменить технические параметры системы: "
        "URL модели, температуру, максимальное количество токенов, размер чанков RAG и т.д. "
        "Не используй для обычного общения, ТОЛЬКО по прямому запросу на изменение настроек."
    )
    args_schema: type[BaseModel] = UpdateSettingsInput

    def _run(self, llm: dict | None = None, agent: dict | None = None,
             rag: dict | None = None, edms: dict | None = None) -> str:
        try:
            payload_dict = {}
            if llm: payload_dict["llm"] = llm
            if agent: payload_dict["agent"] = agent
            if rag: payload_dict["rag"] = rag
            if edms: payload_dict["edms"] = edms

            # Валидируем через Pydantic схему роутера
            validated_payload = UpdateSettingsRequest(**payload_dict)

            # Применяем напрямую к глобальному объекту
            _store.apply_patch(validated_payload)

            return "✅ Настройки успешно обновлены в реальном времени. Следующие запросы будут использовать новые параметры."
        except Exception as e:
            logger.error(f"Failed to update settings via tool: {e}")
            return f"❌ Ошибка при обновлении настроек: {str(e)}"


# Экспортируем экземпляр для реестра
update_settings_tool = UpdateSettingsTool()
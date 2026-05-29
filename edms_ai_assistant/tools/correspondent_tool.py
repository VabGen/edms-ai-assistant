# edms_ai_assistant/tools/correspondent_tool.py
"""
EDMS AI Assistant — Correspondent Tool.
Инструменты для работы с корреспондентами и адресатами.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.tools import InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field

from edms_ai_assistant.agent.runnable_utils import get_token_from_config
from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from edms_ai_assistant.core.deps import AppDeps

logger = logging.getLogger(__name__)


class CorrespondentSearchInput(BaseModel):
    fts: str = Field(..., description="Строка поиска (название или код).")


def create_correspondent_tools(deps: AppDeps) -> list[StructuredTool]:
    """Фабрика инструментов для работы с корреспондентами."""

    async def correspondent_search_tool(
        fts: str,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Ищет корреспондента по названию или коду."""
        token = get_token_from_config(config)

        try:
            # Используем FTS поиск
            result = await deps.correspondent_client.search_correspondent_fts(
                token, fts
            )
            return {
                "status": "success",
                "correspondent": result.model_dump(by_alias=True),
            }
        except Exception:
            # Если не найдено точным совпадением, пробуем общий поиск
            slice_res = await deps.correspondent_client.get_correspondents(
                token, {"name": fts}
            )
            if slice_res.content:
                return {
                    "status": "success",
                    "correspondents": [
                        c.model_dump(by_alias=True) for c in slice_res.content
                    ],
                }
            return {
                "status": "not_found",
                "message": f"Корреспондент «{fts}» не найден.",
            }

    return [
        StructuredTool.from_function(
            coroutine=correspondent_search_tool,
            name="correspondent_search",
            description="Ищет информацию о корреспонденте или контрагенте по названию.",
            args_schema=CorrespondentSearchInput,
        ),
    ]

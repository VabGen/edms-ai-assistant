# edms_ai_assistant/tools/report_tool.py
"""
EDMS AI Assistant — Report Tool.
Инструменты для работы с отчетами V2.
"""

from __future__ import annotations

import logging
from typing import Any, Annotated, TYPE_CHECKING
from uuid import UUID

from langchain_core.tools import StructuredTool, InjectedToolArg
from pydantic import BaseModel, Field

from edms_ai_assistant.agent.runnable_utils import get_token_from_config
from edms_ai_assistant.domain.report import (
    ReportTaskFilter,
    PerformingDisciplineReportFilter,
    VolumeOfDocumentFlowReportFilter,
    ReceivedAppealsReportFilter,
)
from edms_ai_assistant.domain.enums import ReportTaskStatus, ReportFormatType, DocCategory, DeclarantType

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from edms_ai_assistant.core.deps import AppDeps

logger = logging.getLogger(__name__)


class ReportListInput(BaseModel):
    status: ReportTaskStatus | None = Field(None, description="Фильтр по статусу отчета.")
    page: int = Field(0, description="Номер страницы.")
    size: int = Field(10, description="Размер страницы.")


class PerformingDisciplineInput(BaseModel):
    date_reg_start: str = Field(..., description="Начальная дата (ISO).")
    date_reg_end: str = Field(..., description="Конечная дата (ISO).")
    doc_categories: list[DocCategory] = Field(default_factory=list, description="Категории документов.")
    in_archive: bool | None = Field(None, description="Включать архивные.")


def create_report_tools(deps: AppDeps) -> list[StructuredTool]:
    """Фабрика инструментов для работы с отчетами."""

    async def report_list_tool(
        status: ReportTaskStatus | None = None,
        page: int = 0,
        size: int = 10,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Возвращает список отчетов пользователя (V2)."""
        token = get_token_from_config(config)
        filter_params = ReportTaskFilter(status=status) if status else None

        result = await deps.report_client.find_all(token, filter_params, page, size)

        return {
            "status": "success",
            "reports": [r.model_dump(by_alias=True) for r in result.content],
            "has_next": result.has_next,
            "total": result.number_of_elements
        }

    async def report_performing_discipline_tool(
        date_reg_start: str,
        date_reg_end: str,
        doc_categories: list[DocCategory],
        in_archive: bool | None = None,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Создает отчет об исполнительской дисциплине."""
        token = get_token_from_config(config)
        from datetime import datetime

        filter_data = PerformingDisciplineReportFilter(
            date_reg_start=datetime.fromisoformat(date_reg_start),
            date_reg_end=datetime.fromisoformat(date_reg_end),
            doc_category_constants=doc_categories,
            in_archive=in_archive
        )

        result = await deps.report_client.create_performing_discipline_report(token, filter_data)
        return {
            "status": "success",
            "message": f"Задача на формирование отчета создана. ID: {result.id}",
            "report": result.model_dump(by_alias=True)
        }

    return [
        StructuredTool.from_function(
            coroutine=report_list_tool,
            name="report_list",
            description="Возвращает список отчетов пользователя.",
            args_schema=ReportListInput,
        ),
        StructuredTool.from_function(
            coroutine=report_performing_discipline_tool,
            name="report_create_performing_discipline",
            description="Создает отчет об исполнительской дисциплине за указанный период.",
            args_schema=PerformingDisciplineInput,
        ),
    ]

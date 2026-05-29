# edms_ai_assistant/tools/report_tool.py
"""
EDMS AI Assistant — Report Tool.
Инструменты для работы с отчетами V2.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.tools import InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field

from edms_ai_assistant.agent.runnable_utils import get_token_from_config
from edms_ai_assistant.domain.enums import (
    DeclarantType,
    DocCategory,
    DocumentStatus,
    ReportFormatType,
    ReportTaskStatus,
)
from edms_ai_assistant.domain.report import (
    PerformingDisciplineReportFilter,
    ReceivedAppealsReportFilter,
    ReportTaskFilter,
    VolumeOfDocumentFlowReportFilter,
)
from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from edms_ai_assistant.core.deps import AppDeps

logger = logging.getLogger(__name__)


class ReportListInput(BaseModel):
    status: ReportTaskStatus | None = Field(
        None, description="Фильтр по статусу отчета."
    )
    page: int = Field(0, description="Номер страницы.")
    size: int = Field(10, description="Размер страницы.")


class PerformingDisciplineInput(BaseModel):
    date_reg_start: str = Field(..., description="Начальная дата (ISO).")
    date_reg_end: str = Field(..., description="Конечная дата (ISO).")
    doc_categories: list[DocCategory] = Field(
        default_factory=list, description="Категории документов."
    )
    in_archive: bool | None = Field(None, description="Включать архивные.")


class VolumeDocumentFlowInput(BaseModel):
    date_reg_start: str = Field(..., description="Начальная дата (ISO).")
    date_reg_end: str = Field(..., description="Конечная дата (ISO).")
    flag_diagram_circular: bool = Field(True, description="Круговая диаграмма.")
    flag_diagram_by_type: bool = Field(True, description="Диаграмма по типам.")
    in_archive: bool | None = Field(None, description="Включать архивные.")


class ReceivedAppealsInput(BaseModel):
    date_reg_start: str = Field(..., description="Начальная дата (ISO).")
    date_reg_end: str = Field(..., description="Конечная дата (ISO).")
    declarant_types: list[DeclarantType] = Field(
        default_factory=list, description="Типы заявителей (INDIVIDUAL/ENTITY)."
    )
    in_archive: bool | None = Field(None, description="Включать архивные.")


class DocumentStatusReportInput(BaseModel):
    doc_categories: list[DocCategory] = Field(..., description="Категории документов.")
    statuses: list[DocumentStatus] = Field(..., description="Статусы документов.")
    in_archive: bool | None = Field(None, description="Включать архивные.")
    type: ReportFormatType = Field(ReportFormatType.XLSX, description="Формат отчета.")


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
            "total": result.number_of_elements,
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
            in_archive=in_archive,
        )

        result = await deps.report_client.create_performing_discipline_report(
            token, filter_data
        )
        return {
            "status": "success",
            "message": f"Задача на формирование отчета создана. ID: {result.id}",
            "report": result.model_dump(by_alias=True),
        }

    async def report_volume_flow_tool(
        date_reg_start: str,
        date_reg_end: str,
        flag_diagram_circular: bool = True,
        flag_diagram_by_type: bool = True,
        in_archive: bool | None = None,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Создает отчет об объеме документооборота."""
        token = get_token_from_config(config)
        from datetime import datetime

        filter_data = VolumeOfDocumentFlowReportFilter(
            date_reg_start=datetime.fromisoformat(date_reg_start),
            date_reg_end=datetime.fromisoformat(date_reg_end),
            flag_diagram_circular=flag_diagram_circular,
            flag_diagram_by_type=flag_diagram_by_type,
            in_archive=in_archive,
        )

        result = await deps.report_client.create_volume_of_document_flow_report(
            token, filter_data
        )
        return {
            "status": "success",
            "message": f"Задача на формирование отчета создана. ID: {result.id}",
            "report": result.model_dump(by_alias=True),
        }

    async def report_received_appeals_tool(
        date_reg_start: str,
        date_reg_end: str,
        declarant_types: list[DeclarantType],
        in_archive: bool | None = None,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Создает отчет о поступивших обращениях."""
        token = get_token_from_config(config)
        from datetime import datetime

        filter_data = ReceivedAppealsReportFilter(
            date_reg_start=datetime.fromisoformat(date_reg_start),
            date_reg_end=datetime.fromisoformat(date_reg_end),
            declarant_types=declarant_types,
            in_archive=in_archive,
        )

        result = await deps.report_client.create_received_appeals_report(
            token, filter_data
        )
        return {
            "status": "success",
            "message": f"Задача на формирование отчета создана. ID: {result.id}",
            "report": result.model_dump(by_alias=True),
        }

    async def report_document_status_tool(
        doc_categories: list[DocCategory],
        statuses: list[DocumentStatus],
        in_archive: bool | None = None,
        report_type: ReportFormatType = ReportFormatType.XLSX,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Создает отчет по статусам документов."""
        token = get_token_from_config(config)
        from edms_ai_assistant.domain.report import DocumentOnStatusReportFilter

        filter_data = DocumentOnStatusReportFilter(
            type=report_type,
            doc_category_constants=doc_categories,
            status=statuses,
            in_archive=in_archive,
        )

        result = await deps.report_client.create_document_on_status_report(
            token, filter_data
        )
        return {
            "status": "success",
            "message": f"Задача на формирование отчета по статусам создана. ID: {result.id}",
            "report": result.model_dump(by_alias=True),
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
        StructuredTool.from_function(
            coroutine=report_volume_flow_tool,
            name="report_create_volume_flow",
            description="Создает отчет об объеме документооборота.",
            args_schema=VolumeDocumentFlowInput,
        ),
        StructuredTool.from_function(
            coroutine=report_received_appeals_tool,
            name="report_create_received_appeals",
            description="Создает отчет о поступивших обращениях граждан.",
            args_schema=ReceivedAppealsInput,
        ),
        StructuredTool.from_function(
            coroutine=report_document_status_tool,
            name="report_create_document_status",
            description="Создает отчет по статусам документов.",
            args_schema=DocumentStatusReportInput,
        ),
    ]

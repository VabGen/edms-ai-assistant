# edms_ai_assistant/clients/report_client.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.domain.employee import SliceDto
from edms_ai_assistant.domain.enums import ReportTaskControlField
from edms_ai_assistant.domain.report import (
    CountOfExecutedAndUnexecutedControlTaskFilter,
    DocumentOnControlReportFilter,
    DocumentOnRegistrationReportFilter,
    DocumentOnStatusReportFilter,
    IdsDto,
    PerformingDisciplineReportFilter,
    ReceivedAppealsReportFilter,
    ReportConstructRequest,
    ReportTaskDto,
    ReportTaskFilter,
    TaskOnControlReportFilter,
    TaskOnStatusReportFilter,
    VolumeOfDocumentFlowReportFilter,
)

if TYPE_CHECKING:
    from edms_ai_assistant.clients.transport import IAsyncTransport
    from edms_ai_assistant.config import EdmsSettings

logger = logging.getLogger(__name__)


class ReportClient(EdmsBaseClient):
    """Клиент для работы с отчетами V2."""

    def __init__(self, transport: IAsyncTransport, settings: EdmsSettings):
        super().__init__(transport, settings)

    async def find_all(
        self,
        token: str,
        filter_params: ReportTaskFilter | None = None,
        page: int = 0,
        size: int = 20,
    ) -> SliceDto[ReportTaskDto]:
        """GET api/report/v2 - Получить отчеты пользователя."""
        logger.info("Fetching user reports (V2)")
        params = (
            filter_params.model_dump(exclude_none=True, by_alias=True)
            if filter_params
            else {}
        )
        params.update({"page": page, "size": size, "sort": "createDate,desc"})
        return await self._request_dto(
            "GET", "api/report/v2", token, SliceDto[ReportTaskDto], params=params
        )

    async def find_by_id(self, token: str, report_id: UUID) -> ReportTaskDto:
        """GET api/report/v2/{id} - Получить отчет по id."""
        logger.info(f"Fetching report V2 by id: {report_id}")
        return await self._request_dto(
            "GET", f"api/report/v2/{report_id}", token, ReportTaskDto
        )

    async def create_construct_report(
        self, token: str, report_type: str, request: ReportConstructRequest
    ) -> ReportTaskDto:
        """POST api/report/v2/construct-report/{type} - Формирование отчета по конструктору."""
        logger.info(f"Creating construct report V2 of type: {report_type}")
        return await self._request_dto(
            "POST",
            f"api/report/v2/construct-report/{report_type}",
            token,
            ReportTaskDto,
            json_data=request.model_dump(exclude_none=True, by_alias=True),
        )

    async def create_task_control_construct_report(
        self, token: str, fields: dict[ReportTaskControlField, str] | None = None
    ) -> ReportTaskDto:
        """POST api/report/v2/task-control - Формирование отчета по контролю поручений через конструктор."""
        logger.info("Creating task control construct report V2")
        return await self._request_dto(
            "POST",
            "api/report/v2/task-control",
            token,
            ReportTaskDto,
            json_data=fields if fields else {},
        )

    async def create_document_on_registration_report(
        self, token: str, filter_data: DocumentOnRegistrationReportFilter
    ) -> ReportTaskDto:
        """POST api/report/v2/document-on-registration - Отчет по зарегистрированным документам."""
        logger.info("Creating document on registration report V2")
        return await self._request_dto(
            "POST",
            "api/report/v2/document-on-registration",
            token,
            ReportTaskDto,
            json_data=filter_data.model_dump(exclude_none=True, by_alias=True),
        )

    async def create_document_on_control_report(
        self, token: str, filter_data: DocumentOnControlReportFilter
    ) -> ReportTaskDto:
        """POST api/report/v2/document-on-control - Отчет по контрольным документам."""
        logger.info("Creating document on control report V2")
        return await self._request_dto(
            "POST",
            "api/report/v2/document-on-control",
            token,
            ReportTaskDto,
            json_data=filter_data.model_dump(exclude_none=True, by_alias=True),
        )

    async def create_document_on_status_report(
        self, token: str, filter_data: DocumentOnStatusReportFilter
    ) -> ReportTaskDto:
        """POST api/report/v2/document-on-status - Отчет по статусам документов."""
        logger.info("Creating document on status report V2")
        return await self._request_dto(
            "POST",
            "api/report/v2/document-on-status",
            token,
            ReportTaskDto,
            json_data=filter_data.model_dump(exclude_none=True, by_alias=True),
        )

    async def create_task_on_status_report(
        self, token: str, filter_data: TaskOnStatusReportFilter
    ) -> ReportTaskDto:
        """POST api/report/v2/task-on-status - Отчет по поручениям."""
        logger.info("Creating task on status report V2")
        return await self._request_dto(
            "POST",
            "api/report/v2/task-on-status",
            token,
            ReportTaskDto,
            json_data=filter_data.model_dump(exclude_none=True, by_alias=True),
        )

    async def create_task_on_control_report(
        self, token: str, filter_data: TaskOnControlReportFilter
    ) -> ReportTaskDto:
        """POST api/report/v2/task-on-control - Отчет по контролю поручений."""
        logger.info("Creating task on control report V2")
        return await self._request_dto(
            "POST",
            "api/report/v2/task-on-control",
            token,
            ReportTaskDto,
            json_data=filter_data.model_dump(exclude_none=True, by_alias=True),
        )

    async def create_volume_of_document_flow_report(
        self, token: str, filter_data: VolumeOfDocumentFlowReportFilter
    ) -> ReportTaskDto:
        """POST api/report/v2/volume-of-document-flow - Отчет об объеме документооборота."""
        logger.info("Creating volume of document flow report V2")
        return await self._request_dto(
            "POST",
            "api/report/v2/volume-of-document-flow",
            token,
            ReportTaskDto,
            json_data=filter_data.model_dump(exclude_none=True, by_alias=True),
        )

    async def create_received_appeals_report(
        self, token: str, filter_data: ReceivedAppealsReportFilter
    ) -> ReportTaskDto:
        """POST api/report/v2/received-appeals - Отчет о поступивших обращениях."""
        logger.info("Creating received appeals report V2")
        return await self._request_dto(
            "POST",
            "api/report/v2/received-appeals",
            token,
            ReportTaskDto,
            json_data=filter_data.model_dump(exclude_none=True, by_alias=True),
        )

    async def create_performing_discipline_report(
        self, token: str, filter_data: PerformingDisciplineReportFilter
    ) -> ReportTaskDto:
        """POST api/report/v2/performing-discipline - Отчет об исполнительской дисциплине."""
        logger.info("Creating performing discipline report V2")
        return await self._request_dto(
            "POST",
            "api/report/v2/performing-discipline",
            token,
            ReportTaskDto,
            json_data=filter_data.model_dump(exclude_none=True, by_alias=True),
        )

    async def create_count_of_executed_and_unexecuted_control_task_report(
        self, token: str, filter_data: CountOfExecutedAndUnexecutedControlTaskFilter
    ) -> ReportTaskDto:
        """POST api/report/v2/count-of-executed-and-unexecuted-control-task - Отчет о кол-ве контрольных поручений."""
        logger.info("Creating executed/unexecuted control task report V2")
        return await self._request_dto(
            "POST",
            "api/report/v2/count-of-executed-and-unexecuted-control-task",
            token,
            ReportTaskDto,
            json_data=filter_data.model_dump(exclude_none=True, by_alias=True),
        )

    async def download_report(self, token: str, report_id: UUID) -> bytes:
        """GET api/report/v2/{id}/download - Скачать сформированный отчет."""
        logger.info(f"Downloading report V2: {report_id}")
        return await self.make_request(
            "GET",
            f"api/report/v2/{report_id}/download",
            token,
            is_json_response=False,
            long_timeout=True,
        )

    async def delete_report(self, token: str, report_id: UUID) -> None:
        """DELETE api/report/v2/{id} - Удалить отчет по id."""
        logger.info(f"Deleting report V2: {report_id}")
        await self.make_request(
            "DELETE", f"api/report/v2/{report_id}", token, is_json_response=False
        )

    async def delete_reports_batch(self, token: str, ids: list[UUID]) -> None:
        """DELETE api/report/v2 - Удалить отчеты по списку id."""
        logger.info(f"Deleting reports V2 batch: {ids}")
        await self.make_request(
            "DELETE",
            "api/report/v2",
            token,
            json_data=IdsDto(ids=ids).model_dump(by_alias=True),
            is_json_response=False,
        )

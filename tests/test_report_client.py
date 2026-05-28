import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from edms_ai_assistant.clients.report_client import ReportClient
from edms_ai_assistant.domain.report import ReportTaskFilter
from edms_ai_assistant.domain.enums import ReportTaskStatus, ReportType
from edms_ai_assistant.config import EdmsSettings

@pytest.fixture
def mock_transport():
    transport = MagicMock()
    transport.request = AsyncMock()
    return transport

@pytest.fixture
def edms_settings():
    return EdmsSettings(base_url="http://test", timeout=10, long_timeout=30)

@pytest.mark.asyncio
async def test_report_client_init(mock_transport, edms_settings):
    client = ReportClient(mock_transport, edms_settings)
    assert client._transport == mock_transport
    assert client._settings == edms_settings

@pytest.mark.asyncio
async def test_report_client_find_all(mock_transport, edms_settings):
    client = ReportClient(mock_transport, edms_settings)
    token = "test-token"

    # Mock response for SliceDto[ReportTaskDto]
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'{"content": [{"id": "' + str(uuid4()).encode() + b'", "status": "COMPLETED", "type": "CONSTRUCT"}], "hasNext": false, "number": 0, "size": 20, "numberOfElements": 1}'
    mock_response.json.return_value = {
        "content": [
            {
                "id": str(uuid4()),
                "status": "COMPLETED",
                "type": "CONSTRUCT"
            }
        ],
        "hasNext": False,
        "number": 0,
        "size": 20,
        "numberOfElements": 1
    }
    mock_transport.request.return_value = mock_response

    result = await client.find_all(token, filter_params=ReportTaskFilter(status=ReportTaskStatus.COMPLETED))

    assert len(result.content) == 1
    assert result.content[0].status == ReportTaskStatus.COMPLETED
    assert result.content[0].type == ReportType.CONSTRUCT

    mock_transport.request.assert_called_once()
    args, kwargs = mock_transport.request.call_args
    assert args[0] == "GET"
    assert args[1] == "api/report/v2"
    assert kwargs["params"]["status"] == "COMPLETED"
    assert kwargs["params"]["sort"] == "createDate,desc"

@pytest.mark.asyncio
async def test_report_client_find_by_id(mock_transport, edms_settings):
    client = ReportClient(mock_transport, edms_settings)
    token = "test-token"
    report_id = uuid4()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'{"id": "' + str(report_id).encode() + b'", "status": "NEW", "type": "DOCUMENT_ON_REGISTRATION"}'
    mock_response.json.return_value = {
        "id": str(report_id),
        "status": "NEW",
        "type": "DOCUMENT_ON_REGISTRATION"
    }
    mock_transport.request.return_value = mock_response

    result = await client.find_by_id(token, report_id)

    assert result.id == report_id
    assert result.status == ReportTaskStatus.NEW
    assert result.type == ReportType.DOCUMENT_ON_REGISTRATION

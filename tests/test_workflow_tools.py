import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from edms_ai_assistant.tools.doc_process_action import create_doc_process_action_tool
from edms_ai_assistant.domain.enums import DocumentProcessType
from edms_ai_assistant.domain.employee import CurrentUserDto

@pytest.fixture
def mock_deps():
    deps = MagicMock()
    deps.document_process_client = AsyncMock()
    deps.employee_client = AsyncMock()
    return deps

@pytest.mark.asyncio
async def test_doc_process_action_agreement(mock_deps):
    tool = create_doc_process_action_tool(mock_deps)

    doc_id = str(uuid4())
    token = "test-token"
    user_id = uuid4()

    # Mock process
    mock_process = MagicMock()
    mock_process.current_id = uuid4()
    mock_process.current = MagicMock()
    mock_process.current.type = DocumentProcessType.AGREEMENT
    mock_deps.document_process_client.get_process.return_value = mock_process

    # Mock user
    mock_user = MagicMock(spec=CurrentUserDto)
    mock_user.id = user_id
    mock_deps.employee_client.get_current_user.return_value = mock_user

    result = await tool.coroutine(
        action_type=DocumentProcessType.AGREEMENT,
        result=True,
        comment="Approved",
        config={"configurable": {"user_token": token, "document_id": doc_id}}
    )

    if result["status"] == "error":
        print(f"Error message: {result['message']}")

    assert result["status"] == "success"
    assert "успешно выполнено" in result["message"]
    mock_deps.document_process_client.agreement.assert_called_once()

from unittest.mock import MagicMock
import redis.asyncio as aioredis
from edms_ai_assistant.core.deps import init_deps
from edms_ai_assistant.clients.transport import IAsyncTransport
from edms_ai_assistant.agent.agent import EdmsDocumentAgent

def test_deps_initialization():
    """Verify that AppDeps can be initialized with mocked transport and redis."""
    mock_transport = MagicMock(spec=IAsyncTransport)
    mock_redis = MagicMock(spec=aioredis.Redis)
    mock_llm = MagicMock()

    deps = init_deps(mock_transport, mock_redis, mock_llm)

    assert deps.transport == mock_transport
    assert deps.redis == mock_redis
    assert deps.document_client is not None
    assert deps.employee_client is not None
    assert deps.document_service is not None
    assert deps.resolution_service is not None

def test_agent_initialization():
    """Verify that EdmsDocumentAgent can be initialized with AppDeps."""
    mock_transport = MagicMock(spec=IAsyncTransport)
    mock_redis = MagicMock(spec=aioredis.Redis)
    mock_llm = MagicMock()

    deps = init_deps(mock_transport, mock_redis, mock_llm)
    agent = EdmsDocumentAgent(deps=deps, llm=mock_llm)

    assert agent.deps == deps
    assert agent.tools is not None
    # Check if some tools are present
    tool_names = [t.name for t in agent.tools]
    assert "doc_get_details" in tool_names
    assert "doc_search_tool" in tool_names
    assert "ask_user_to_select" in tool_names

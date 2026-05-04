# tests/agent/test_agent_integration.py
"""Интеграционные тесты EdmsDocumentAgent с mock-компонентами."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from edms_ai_assistant.agent.agent import EdmsDocumentAgent
from edms_ai_assistant.agent.context import AgentStatus


@pytest.fixture
def mock_llm():
    """Mock LLM, возвращающий простой ответ без tool_calls."""
    from langchain_core.messages import AIMessage
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="Тестовый ответ агента для проверки.")
    )
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def agent(mock_llm):
    """EdmsDocumentAgent с mock LLM и MemorySaver."""
    from langgraph.checkpoint.memory import MemorySaver
    return EdmsDocumentAgent(
        llm=mock_llm,
        checkpointer=MemorySaver(),
    )


@pytest.mark.asyncio
async def test_health_check_all_true(agent):
    health = await agent.health_check()
    assert all(health.values()), f"Health check failed: {health}"


@pytest.mark.asyncio
async def test_invalid_request_returns_error(agent):
    result = await agent.chat(
        message="",
        user_token="short",  # min_length=10 не соблюдён
        human_choice=None,
    )
    assert result["status"] == AgentStatus.ERROR.value


@pytest.mark.asyncio
async def test_basic_chat_returns_success(agent):
    result = await agent.chat(
        message="Привет, расскажи о документе",
        user_token="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjMmIwZWEyZS0zMzJjLTRjYjktYjEwMi1jYmYxY2JlNTMxMmIiLCJpZCI6ImMyYjBlYTJlLTMzMmMtNGNiOS1iMTAyLWNiZjFjYmU1MzEyYiIsIm9yZ0lkIjoiMTE0IiwiYXV0aFR5cGUiOiJCQVNJQyIsImlhdCI6MTc3NzM4NDMwNiwiZXhwIjoxNzc5MTg0MzAxfQ.7Q1ynlqrrV4C1AFiWdsBLCkBgPz2e_CQUhLKuD8MJUI",
        thread_id="tests-thread-001",
    )
    assert result["status"] == AgentStatus.SUCCESS.value
    assert result.get("content") is not None
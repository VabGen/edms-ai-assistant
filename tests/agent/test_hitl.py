# tests/agent/test_hitl.py
"""Тесты HITL обработчиков."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from unittest.mock import AsyncMock, MagicMock

from edms_ai_assistant.agent.orchestration.hitl import (
    HumanChoiceHandler,
    ChoiceType,
    classify_choice,
    find_pending_tool_call,
    _is_real_tool_result,
)
from edms_ai_assistant.agent.context import ContextParams


@pytest.fixture
def context() -> ContextParams:
    return ContextParams(
        user_token="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjMmIwZWEyZS0zMzJjLTRjYjktYjEwMi1jYmYxY2JlNTMxMmIiLCJpZCI6ImMyYjBlYTJlLTMzMmMtNGNiOS1iMTAyLWNiZjFjYmU1MzEyYiIsIm9yZ0lkIjoiMTE0IiwiYXV0aFR5cGUiOiJCQVNJQyIsImlhdCI6MTc3NzM4NDMwNiwiZXhwIjoxNzc5MTg0MzAxfQ.7Q1ynlqrrV4C1AFiWdsBLCkBgPz2e_CQUhLKuD8MJUI",
        thread_id="tests-thread",
    )


class TestClassifyChoice:
    def test_fix_field(self):
        assert classify_choice("fix_field:shortSummary:New Title") == ChoiceType.FIX_FIELD

    def test_summary_extractive(self):
        assert classify_choice("extractive") == ChoiceType.SUMMARY_TYPE

    def test_summary_abstractive(self):
        assert classify_choice("abstractive") == ChoiceType.SUMMARY_TYPE

    def test_employee_uuid(self):
        uuid = "c2b0ea2e-332c-4cb9-b102-cbf1cbe5312b"
        assert classify_choice(uuid) == ChoiceType.EMPLOYEE_UUID

    def test_multiple_uuids(self):
        uuids = "c2b0ea2e-332c-4cb9-b102-cbf1cbe5312b,87654321-4321-4321-4321-210987654321"
        assert classify_choice(uuids) == ChoiceType.EMPLOYEE_UUID

    def test_attachment_uuid_with_tool(self):
        uuid = "12345678-1234-1234-1234-123456789012"
        result = classify_choice(uuid, "doc_compare_attachment_with_local")
        assert result == ChoiceType.ATTACHMENT_UUID

    def test_plain_message(self):
        assert classify_choice("Найди Иванова") == ChoiceType.PLAIN_MESSAGE

    def test_partial_invalid_not_uuid(self):
        assert classify_choice("not-a-valid-uuid") == ChoiceType.PLAIN_MESSAGE


class TestFindPendingToolCall:
    def test_finds_pending_without_response(self):
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "doc_summarize_text", "args": {}, "id": "call_1"}],
            id="msg_1",
        )
        messages = [HumanMessage(content="tests"), ai_msg]
        result = find_pending_tool_call(messages)
        assert result is not None
        assert result.id == "msg_1"

    def test_no_pending_when_tool_responded(self):
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "doc_summarize_text", "args": {}, "id": "call_1"}],
            id="msg_1",
        )
        tool_msg = ToolMessage(
            content='{"status": "success", "content": "Summary here for testing purposes"}',
            tool_call_id="call_1",
            name="doc_summarize_text",
        )
        messages = [ai_msg, tool_msg]
        result = find_pending_tool_call(messages)
        assert result is None

    def test_no_ai_messages_returns_none(self):
        messages = [HumanMessage(content="tests")]
        assert find_pending_tool_call(messages) is None


class TestIsRealToolResult:
    def test_real_result(self):
        msg = ToolMessage(
            content='{"status": "success", "content": "Some real content here"}',
            tool_call_id="call_1",
        )
        assert _is_real_tool_result(msg) is True

    def test_empty_result(self):
        msg = ToolMessage(content="{}", tool_call_id="call_1")
        assert _is_real_tool_result(msg) is False

    def test_null_result(self):
        msg = ToolMessage(content="null", tool_call_id="call_1")
        assert _is_real_tool_result(msg) is False


class TestHumanChoiceHandler:
    @pytest.mark.asyncio
    async def test_fix_field_handler(self, context):
        handler = HumanChoiceHandler()
        result = await handler.process(
            "fix_field:shortSummary:New Title",
            context,
            [HumanMessage(content="tests")],
        )
        assert result.new_inputs is not None
        msg = result.new_inputs["messages"][0]
        assert isinstance(msg, HumanMessage)
        assert "shortSummary" in msg.content
        assert "New Title" in msg.content

    @pytest.mark.asyncio
    async def test_summary_type_with_pending(self, context):
        ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "name": "doc_summarize_text",
                "args": {"text": "Some text"},
                "id": "call_1"
            }],
            id="msg_1",
        )
        messages = [HumanMessage(content="summarize"), ai_msg]

        handler = HumanChoiceHandler()
        result = await handler.process("extractive", context, messages)

        assert result.patched_messages is not None
        assert result.resume_from_interrupt is True
        patched = result.patched_messages[0]
        assert isinstance(patched, AIMessage)
        assert patched.tool_calls[0]["args"]["summary_type"] == "extractive"

    @pytest.mark.asyncio
    async def test_unknown_summary_defaults_to_extractive(self, context):
        ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "name": "doc_summarize_text",
                "args": {},
                "id": "call_1"
            }],
            id="msg_1",
        )
        handler = HumanChoiceHandler()
        result = await handler.process("unknown_type", context, [ai_msg])
        # unknown_type не в _SUMMARIZE_TYPES → plain message
        assert result.new_inputs is not None
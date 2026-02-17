import pytest
from unittest.mock import Mock, AsyncMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class TestToolCallProcessorFixes:
    def test_inject_credentials_no_document(self):
        from edms_ai_assistant.agent import ToolCallProcessor, ContextParams

        processor = ToolCallProcessor()
        context = ContextParams(user_token="test_token")
        t_args = {}

        result = processor.inject_credentials(t_args, context)

        assert result["token"] == "test_token"
        assert result.get("document_id") is None

    def test_extract_final_content_from_tool_message(self):
        from edms_ai_assistant.agent import ContentExtractor

        messages = [
            HumanMessage(content="Hello"),
            ToolMessage(
                content='{"content": "Extracted content from tool with enough length"}',
                tool_call_id="call_123",
            ),
        ]

        result = ContentExtractor.extract_final_content(messages)

        assert result is not None
        assert len(result) > 50

    def test_extract_last_text(self):
        from edms_ai_assistant.agent import ContentExtractor

        long_text = "This is extracted text content with sufficient length to be returned. " * 3

        messages = [
            HumanMessage(content="Hello"),
            ToolMessage(
                content=f'{{"text": "{long_text}"}}',
                tool_call_id="call_123",
            ),
        ]

        result = ContentExtractor.extract_last_text(messages)

        assert result is not None
        assert len(result) > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
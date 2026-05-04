# tests/agent/test_content_extractor.py
"""Тесты ContentExtractor — single pass extraction."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from edms_ai_assistant.agent.extraction.content_extractor import (
    ContentExtractor,
    _is_technical_content,
)


class TestIsTechnicalContent:
    def test_tool_call_marker(self):
        assert _is_technical_content('"tool_calls": [...]') is True

    def test_normal_text(self):
        assert _is_technical_content("Документ успешно проанализирован.") is False

    def test_russian_tool_mention(self):
        assert _is_technical_content("вызвал инструмент doc_get_details") is True


class TestExtractFinalContent:
    def test_prefers_ai_message(self):
        messages = [
            HumanMessage(content="tests"),
            ToolMessage(
                content='{"status": "success", "content": "Tool result here for user visibility"}',
                tool_call_id="c1",
            ),
            AIMessage(content="Документ содержит следующую информацию о поставках за квартал."),
        ]
        result = ContentExtractor.extract_final_content(messages)
        assert result == "Документ содержит следующую информацию о поставках за квартал."

    def test_falls_back_to_tool_json(self):
        long_content = "А" * 50  # >= _MIN_CONTENT_LENGTH
        messages = [
            ToolMessage(
                content=f'{{"status": "success", "content": "{long_content}"}}',
                tool_call_id="c1",
            ),
        ]
        result = ContentExtractor.extract_final_content(messages)
        assert result == long_content

    def test_skips_technical_ai_messages(self):
        long_content = "Б" * 50
        messages = [
            ToolMessage(
                content=f'{{"status": "success", "content": "{long_content}"}}',
                tool_call_id="c1",
            ),
            AIMessage(content='"tool_calls": [{"name": "doc_get_details"}]'),
        ]
        result = ContentExtractor.extract_final_content(messages)
        assert result == long_content

    def test_returns_none_on_empty(self):
        messages = [HumanMessage(content="tests")]
        assert ContentExtractor.extract_final_content(messages) is None

    def test_skips_error_status_in_tool_json(self):
        messages = [
            ToolMessage(
                content='{"status": "error", "message": "Not found"}',
                tool_call_id="c1",
            ),
        ]
        result = ContentExtractor.extract_final_content(messages)
        assert result is None

    def test_informational_status_returned(self):
        messages = [
            ToolMessage(
                content='{"status": "already_exists", "message": "Сотрудник уже добавлен в список"}',
                tool_call_id="c1",
            ),
        ]
        result = ContentExtractor.extract_final_content(messages)
        assert result == "Сотрудник уже добавлен в список"


class TestCleanJsonArtifacts:
    def test_extracts_content_from_full_json(self):
        long_val = "Д" * 50
        content = f'{{"status": "success", "content": "{long_val}"}}'
        result = ContentExtractor.clean_json_artifacts(content)
        assert result == long_val

    def test_plain_text_unchanged(self):
        text = "Обычный текст без JSON."
        assert ContentExtractor.clean_json_artifacts(text) == text


class TestExtractNavigateUrl:
    def test_extracts_url(self):
        messages = [
            ToolMessage(
                content='{"status": "success", "navigate_url": "/document-form/45e1a069-1d36-11f1-9031-309c2375a126"}',
                tool_call_id="c1",
            ),
        ]
        url = ContentExtractor.extract_navigate_url(messages)
        assert url == "/document-form/45e1a069-1d36-11f1-9031-309c2375a126"

    def test_ignores_external_urls(self):
        messages = [
            ToolMessage(
                content='{"status": "success", "navigate_url": "https://external.com"}',
                tool_call_id="c1",
            ),
        ]
        assert ContentExtractor.extract_navigate_url(messages) is None

    def test_returns_none_when_absent(self):
        messages = [HumanMessage(content="tests")]
        assert ContentExtractor.extract_navigate_url(messages) is None
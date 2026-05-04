# tests/agent/test_context.py
"""Тесты иммутабельности ContextParams."""
import pytest
import dataclasses
from edms_ai_assistant.agent.context import ContextParams
from edms_ai_assistant.services.nlp_service import UserIntent


class TestContextParamsImmutability:
    def test_frozen_raises_on_direct_mutation(self):
        ctx = ContextParams(user_token="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjMmIwZWEyZS0zMzJjLTRjYjktYjEwMi1jYmYxY2JlNTMxMmIiLCJpZCI6ImMyYjBlYTJlLTMzMmMtNGNiOS1iMTAyLWNiZjFjYmU1MzEyYiIsIm9yZ0lkIjoiMTE0IiwiYXV0aFR5cGUiOiJCQVNJQyIsImlhdCI6MTc3NzM4NDMwNiwiZXhwIjoxNzc5MTg0MzAxfQ.7Q1ynlqrrV4C1AFiWdsBLCkBgPz2e_CQUhLKuD8MJUI")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.intent = UserIntent.SEARCH  # type: ignore[misc]

    def test_with_intent_creates_new_instance(self):
        ctx = ContextParams(user_token="valid-token-1234567890")
        new_ctx = ctx.with_intent(UserIntent.SEARCH)
        assert new_ctx is not ctx
        assert new_ctx.intent == UserIntent.SEARCH
        assert ctx.intent is None

    def test_with_intent_preserves_all_fields(self):
        ctx = ContextParams(
            user_token="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjMmIwZWEyZS0zMzJjLTRjYjktYjEwMi1jYmYxY2JlNTMxMmIiLCJpZCI6ImMyYjBlYTJlLTMzMmMtNGNiOS1iMTAyLWNiZjFjYmU1MzEyYiIsIm9yZ0lkIjoiMTE0IiwiYXV0aFR5cGUiOiJCQVNJQyIsImlhdCI6MTc3NzM4NDMwNiwiZXhwIjoxNzc5MTg0MzAxfQ.7Q1ynlqrrV4C1AFiWdsBLCkBgPz2e_CQUhLKuD8MJUI",
            thread_id="thread-123",
            document_id="45e1a069-1d36-11f1-9031-309c2375a126",
        )
        new_ctx = ctx.with_intent(UserIntent.SUMMARIZE)
        assert new_ctx.thread_id == "thread-123"
        assert new_ctx.document_id == "45e1a069-1d36-11f1-9031-309c2375a126"
        assert new_ctx.user_token == "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjMmIwZWEyZS0zMzJjLTRjYjktYjEwMi1jYmYxY2JlNTMxMmIiLCJpZCI6ImMyYjBlYTJlLTMzMmMtNGNiOS1iMTAyLWNiZjFjYmU1MzEyYiIsIm9yZ0lkIjoiMTE0IiwiYXV0aFR5cGUiOiJCQVNJQyIsImlhdCI6MTc3NzM4NDMwNiwiZXhwIjoxNzc5MTg0MzAxfQ.7Q1ynlqrrV4C1AFiWdsBLCkBgPz2e_CQUhLKuD8MJUI"


class TestContextParamsValidation:
    def test_empty_token_raises(self):
        with pytest.raises(ValueError, match="user_token"):
            ContextParams(user_token="")

    def test_auto_derives_filename_from_path(self):
        ctx = ContextParams(
            user_token="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjMmIwZWEyZS0zMzJjLTRjYjktYjEwMi1jYmYxY2JlNTMxMmIiLCJpZCI6ImMyYjBlYTJlLTMzMmMtNGNiOS1iMTAyLWNiZjFjYmU1MzEyYiIsIm9yZ0lkIjoiMTE0IiwiYXV0aFR5cGUiOiJCQVNJQyIsImlhdCI6MTc3NzM4NDMwNiwiZXhwIjoxNzc5MTg0MzAxfQ.7Q1ynlqrrV4C1AFiWdsBLCkBgPz2e_CQUhLKuD8MJUI",
            file_path="/tmp/uploads/document.pdf",
        )
        assert ctx.uploaded_file_name == "document.pdf"

    def test_uuid_path_no_filename_derived(self):
        ctx = ContextParams(
            user_token="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjMmIwZWEyZS0zMzJjLTRjYjktYjEwMi1jYmYxY2JlNTMxMmIiLCJpZCI6ImMyYjBlYTJlLTMzMmMtNGNiOS1iMTAyLWNiZjFjYmU1MzEyYiIsIm9yZ0lkIjoiMTE0IiwiYXV0aFR5cGUiOiJCQVNJQyIsImlhdCI6MTc3NzM4NDMwNiwiZXhwIjoxNzc5MTg0MzAxfQ.7Q1ynlqrrV4C1AFiWdsBLCkBgPz2e_CQUhLKuD8MJUI",
            file_path="45e1a069-1d36-11f1-9031-309c2375a126",
        )
        assert ctx.uploaded_file_name is None

    def test_auto_derives_full_name(self):
        ctx = ContextParams(
            user_token="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjMmIwZWEyZS0zMzJjLTRjYjktYjEwMi1jYmYxY2JlNTMxMmIiLCJpZCI6ImMyYjBlYTJlLTMzMmMtNGNiOS1iMTAyLWNiZjFjYmU1MzEyYiIsIm9yZ0lkIjoiMTE0IiwiYXV0aFR5cGUiOiJCQVNJQyIsImlhdCI6MTc3NzM4NDMwNiwiZXhwIjoxNzc5MTg0MzAxfQ.7Q1ynlqrrV4C1AFiWdsBLCkBgPz2e_CQUhLKuD8MJUI",
            user_first_name="Иван",
            user_last_name="Иванов",
        )
        assert ctx.user_full_name == "Иванов Иван"
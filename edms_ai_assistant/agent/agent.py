# edms_ai_assistant/agent/agent.py
"""EdmsDocumentAgent — thin wrapper вокруг LangGraph-native пайплайна.

См. docstring класса для контракта.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from edms_ai_assistant.agent.context import AgentRequest, ContextParams, is_valid_uuid
from edms_ai_assistant.agent.escaping import xml_escape_text
from edms_ai_assistant.agent.graph import GraphBuilder
from edms_ai_assistant.agent.prompts import PromptBuilder
from edms_ai_assistant.agent.repositories import IDocumentRepository
from edms_ai_assistant.config import settings
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.services.nlp_service import UserIntent
from edms_ai_assistant.tools import all_tools

logger = logging.getLogger(__name__)


class EdmsDocumentAgent:
    """LangGraph-native ReAct агент с universal HITL через ``interrupt()``.

    Контракт с внешним миром:
      • ``self.graph`` — скомпилированный LangGraph со всеми тулами, работает по
        формуле ``astream(inputs | Command(resume=...), config)``;
      • ``build_initial_inputs(...)`` — собирает первичные messages для astream;
      • ``health_check()`` — лёгкий readiness probe.

    HITL живёт нативно: тулы вызывают ``ask_human(payload)`` →
    ``langgraph.types.interrupt()``, что атомарно пишет чекпойнт и поднимает
    ``GraphInterrupt``. Resume идёт через ``Command(resume=value)`` из API.
    """

    def __init__(
        self,
        document_repo: IDocumentRepository | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        llm: Any = None,
    ) -> None:
        self._document_repo = document_repo or _NullDocumentRepository()
        self._checkpointer: BaseCheckpointSaver = checkpointer or MemorySaver()

        if llm is not None:
            self._model = llm
        else:
            from edms_ai_assistant.llm import get_chat_model

            self._model = get_chat_model()

        self._graph_builder = GraphBuilder(
            tools=all_tools,
            checkpointer=self._checkpointer,
        )
        self._graph_builder.set_model(self._model.bind_tools(all_tools))
        self._graph = self._graph_builder.compile()
        logger.info(
            "EdmsDocumentAgent initialized: tools=%d", len(all_tools)
        )

    @property
    def graph(self) -> Any:
        """Native LangGraph; паузы делают сами тулы через ``ask_human()``."""
        return self._graph

    @property
    def checkpointer(self) -> BaseCheckpointSaver:
        return self._checkpointer

    def refresh_model(self) -> None:
        """Перепривязать тулы к новой инстансе LLM (после смены настроек).

        Перекомпилировать граф не нужно — узел ``call_model`` читает ссылку
        builder._model в рантайме (см. closure в ``GraphBuilder.compile``).
        """
        from edms_ai_assistant.llm import get_chat_model  # noqa: PLC0415

        current = get_chat_model()
        if current is not self._model:
            self._model = current
            self._graph_builder.set_model(current.bind_tools(all_tools))
            logger.info("EdmsDocumentAgent: LLM model refreshed")

    def build_initial_inputs(
        self,
        message: str,
        user_token: str,
        context_ui_id: str | None,
        thread_id: str,
        user_context: dict[str, Any] | None,
        file_path: str | None,
        file_name: str | None,
        doc_info: DocumentDto | None = None,
    ) -> tuple[dict[str, Any], ContextParams]:
        """Build graph inputs + context for a brand-new turn.

        Mirrors the prompt/context construction performed by ``chat()`` but
        without invoking the legacy ``OrchestrationLoop``. The returned
        ``inputs`` should be passed to ``self.graph.astream(inputs, config)``.
        """
        request = AgentRequest(
            message=message,
            user_token=user_token,
            context_ui_id=context_ui_id or "",
            thread_id=thread_id,
            user_context=user_context or {},
            file_path=file_path,
            file_name=file_name,
        )
        context = self._build_context(request)
        inputs = self._build_inputs(context, request.message, doc_info)
        return inputs, context

    async def health_check(self) -> dict[str, bool]:
        checks: dict[str, bool] = {}
        try:
            checks["llm"] = self._model is not None
        except Exception:
            checks["llm"] = False
        try:
            checks["graph"] = self._graph is not None
        except Exception:
            checks["graph"] = False
        try:
            checks["tools"] = bool(all_tools)
        except Exception:
            checks["tools"] = False
        return checks

    def _build_inputs(
        self,
        context: ContextParams,
        message: str,
        doc_info: DocumentDto | None,
    ) -> dict[str, Any]:
        semantic_xml = self._build_semantic_xml(doc_info=doc_info)
        fp = context.file_path
        if fp and not is_valid_uuid(fp):
            intent = UserIntent.FILE_ANALYSIS
        else:
            intent = UserIntent.UNKNOWN
        system_prompt = PromptBuilder.build(
            context=context,
            intent=intent,
            semantic_xml=semantic_xml,
            lean=settings.AGENT_LEAN_PROMPT,
        )
        return {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message),
            ]
        }

    @staticmethod
    def _build_context(request: AgentRequest) -> ContextParams:
        uc: dict[str, Any] = request.user_context or {}
        first_name = (uc.get("first_name") or uc.get("firstName") or "").strip()
        last_name = (uc.get("last_name") or uc.get("lastName") or "").strip()
        middle_name = (uc.get("middle_name") or uc.get("middleName") or "").strip()
        user_id = uc.get("id") or uc.get("user_id") or uc.get("userId")
        parts = [p for p in (last_name, first_name, middle_name) if p]
        full_name = " ".join(parts) if parts else None

        return ContextParams(
            user_token=request.user_token,
            document_id=request.context_ui_id or None,
            file_path=request.file_path,
            thread_id=request.thread_id or "default",
            user_name=first_name or last_name or "пользователь",
            user_first_name=first_name or None,
            user_last_name=last_name or None,
            user_full_name=full_name,
            user_id=str(user_id) if user_id else None,
            uploaded_file_name=request.file_name,
            user_context=uc,
        )

    @staticmethod
    def _build_semantic_xml(doc_info: DocumentDto | None) -> str:
        if not doc_info:
            return ""

        def esc(value: Any) -> str:
            return xml_escape_text(str(value)) if value is not None else ""

        lines: list[str] = ["\n\n<semantic_context>"]
        short_summary = getattr(doc_info, "shortSummary", None)
        if short_summary:
            lines.append(f"  <title>{esc(short_summary)}</title>")
        reg_number = getattr(doc_info, "regNumber", None)
        if reg_number:
            lines.append(f"  <reg_number>{esc(reg_number)}</reg_number>")
        status = getattr(doc_info, "status", None)
        if status:
            lines.append(f"  <status>{esc(status)}</status>")
        category = getattr(doc_info, "docCategoryConstant", None)
        if category:
            lines.append(f"  <category>{esc(category)}</category>")
        lines.append("</semantic_context>")
        return "\n".join(lines)


class _NullDocumentRepository:
    """Null-object implementation of IDocumentRepository.

    Used as a default fallback when no repository is provided.
    Method must remain an instance method to satisfy the Protocol signature.
    """

    async def get_document(self, _token: str, _doc_id: str) -> None:  # noqa: PLR6301
        return None
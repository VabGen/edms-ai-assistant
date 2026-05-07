# edms_ai_assistant/agent/agent.py
"""
EdmsDocumentAgent v2 — Planning-first архитектура.

Главное изменение: SemanticDispatcher убран из chat() метода.
IntentPlanner (внутри OrchestrationLoop) заменяет NLP routing.

Публичный API не изменился — обратная совместимость сохранена.
"""
from __future__ import annotations

import html
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from edms_ai_assistant.agent.context import (
    AgentRequest,
    AgentResponse,
    AgentStatus,
    ContextParams,
    is_valid_uuid,
)
from edms_ai_assistant.agent.graph import GraphBuilder
from edms_ai_assistant.agent.orchestration import handle_human_choice
from edms_ai_assistant.agent.orchestration.loop import OrchestrationLoop
from edms_ai_assistant.agent.prompts import PromptBuilder
from edms_ai_assistant.agent.repositories import IDocumentRepository
from edms_ai_assistant.agent.state_manager import AgentStateManager
from edms_ai_assistant.agent.tool_injector import ToolCallInjector
from edms_ai_assistant.config import settings
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.services.nlp_service import UserIntent
from edms_ai_assistant.tools import all_tools

logger = logging.getLogger(__name__)


def _xml_escape(value: str | None) -> str:
    if not value:
        return ""
    return html.escape(value, quote=False)


class EdmsDocumentAgent:
    """
    Основная точка входа EDMS AI Agent v2.

    Изменения от v1:
    - Убран SemanticDispatcher из chat()
    - OrchestrationLoop содержит IntentPlanner
    - LLM всегда видит ВСЕ tools (нет routing по subset)
    - Параллельное выполнение tools через PlanExecutor
    - Прямые ответы без tools для общих вопросов
    """

    def __init__(
        self,
        document_repo: IDocumentRepository | None = None,
        checkpointer: Any = None,
        llm: Any = None,
    ) -> None:
        self._document_repo = document_repo or _NullDocumentRepository()
        checkpointer = checkpointer or MemorySaver()

        if llm is not None:
            self._model = llm
        else:
            from edms_ai_assistant.llm import get_chat_model
            self._model = get_chat_model()

        self._graph_builder = GraphBuilder(
            tools=all_tools,
            checkpointer=checkpointer,
        )
        compiled_graph = self._graph_builder.compile()

        self._state_manager = AgentStateManager(
            graph=compiled_graph,
            checkpointer=checkpointer,
        )
        self._injector = ToolCallInjector()
        self._loop = OrchestrationLoop(
            state_manager=self._state_manager,
            injector=self._injector,
            all_tools=all_tools,
            graph_builder=self._graph_builder,
            model=self._model,
        )
        logger.info(
            "EdmsDocumentAgent v2 initialized: tools=%d",
            len(all_tools),
        )

    async def health_check(self) -> dict[str, bool]:
        checks: dict[str, bool] = {}
        try:
            checks["llm"] = self._model is not None
        except Exception:
            checks["llm"] = False
        try:
            checks["graph"] = self._state_manager.graph is not None
        except Exception:
            checks["graph"] = False
        try:
            checks["tools"] = bool(all_tools)
        except Exception:
            checks["tools"] = False
        return checks

    async def chat(
        self,
        message: str,
        user_token: str,
        context_ui_id: str | None = None,
        thread_id: str | None = None,
        user_context: dict[str, Any] | None = None,
        file_path: str | None = None,
        file_name: str | None = None,
        human_choice: str | None = None,
    ) -> dict[str, Any]:
        try:
            request = AgentRequest(
                message=message,
                user_token=user_token,
                context_ui_id=context_ui_id or "",
                thread_id=thread_id,
                user_context=user_context or {},
                file_path=file_path,
                file_name=file_name,
                human_choice=human_choice,
            )
        except Exception as exc:
            logger.warning("AgentRequest validation failed: %s", exc)
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Некорректный запрос: {exc}",
            ).model_dump()

        context = self._build_context(request)

        logger.info(
            "chat: thread=%s has_file=%s has_doc=%s has_choice=%s",
            context.thread_id,
            bool(context.file_path),
            bool(context.document_id),
            bool(human_choice),
        )

        # Метаданные документа для semantic_xml в промпте
        doc_info: DocumentDto | None = None
        if context.document_id:
            doc_info = await self._document_repo.get_document(
                token=user_token,
                doc_id=context.document_id,
            )

        # Проверка состояния треда
        try:
            if await self._state_manager.is_thread_broken(context.thread_id):
                await self._state_manager.repair_thread(context.thread_id)
        except Exception as exc:
            logger.error("Thread health check failed: %s", exc, exc_info=True)

        # HITL: возобновление после выбора пользователя
        if human_choice and human_choice.strip():
            return await handle_human_choice(
                context=context,
                human_choice=human_choice.strip(),
                state_manager=self._state_manager,
                loop=self._loop,
            )

        # Строим inputs для graph
        inputs = self._build_inputs(context, request.message, doc_info)

        return await self._loop.run(
            context=context,
            inputs=inputs,
            is_choice_active=False,
        )

    def _build_inputs(
        self,
        context: ContextParams,
        message: str,
        doc_info: DocumentDto | None,
    ) -> dict[str, Any]:
        semantic_xml = self._build_semantic_xml(doc_info=doc_info)
        # Если есть локальный файл (путь, не UUID) — явно добавляем workflow-сниппет
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
        user_id = (
            uc.get("id") or uc.get("user_id") or uc.get("userId")
        )
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
            return _xml_escape(str(value)) if value is not None else ""

        def _user_name(user: Any) -> str | None:
            if user is None:
                return None
            ln = getattr(user, "lastName", None) or ""
            fn = getattr(user, "firstName", None) or ""
            mn = getattr(user, "middleName", None) or ""
            parts = [p for p in (ln, fn, mn) if p]
            return " ".join(parts) if parts else None

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
    async def get_document(self, token: str, doc_id: str) -> None:
        return None
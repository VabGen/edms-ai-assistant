"""
EdmsDocumentAgent — основная точка входа агента.

- Убран _try_forced_tool_call (прямой вызов tool в обход графа)
- Исправлена мутация context.intent (теперь context иммутабелен, with_intent())
- Исправлен "протёк" контекста между диалогами через ToolCallInjector
- Добавлен guard: если предыдущий turn содержал employee_search → не накапливать
  task_create контекст в следующем turn
- handle_human_choice теперь использует рефакторинговый loop.py
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
)
from edms_ai_assistant.agent.graph import GraphBuilder
from edms_ai_assistant.agent.orchestration.loop import (
    OrchestrationLoop,
    handle_human_choice,
)
from edms_ai_assistant.agent.prompts import PromptBuilder
from edms_ai_assistant.agent.repositories import IDocumentRepository
from edms_ai_assistant.agent.state_manager import AgentStateManager
from edms_ai_assistant.agent.tool_injector import ToolCallInjector
from edms_ai_assistant.generated.resources_openapi import DocumentDto
from edms_ai_assistant.services.nlp_service import SemanticDispatcher, UserIntent
from edms_ai_assistant.tools import all_tools

logger = logging.getLogger(__name__)

# Set to True for small LLMs (≤13 B)
USE_LEAN_PROMPT: bool = False


def _xml_escape(value: str | None) -> str:
    if not value:
        return ""
    return html.escape(value, quote=False)


class EdmsDocumentAgent:
    """
    Основная точка входа для EDMS AI Agent.

    Изменения по сравнению с предыдущей версией:
    - ContextParams теперь иммутабелен (frozen=True), intent задаётся через with_intent()
    - Убран _try_forced_tool_call — создание документа идёт через нормальный граф
    - handle_human_choice делегирует в рефакторинговый orchestration/loop.py
    """

    def __init__(
        self,
        document_repo: IDocumentRepository | None = None,
        semantic_dispatcher: SemanticDispatcher | None = None,
        checkpointer: Any = None,
        llm: Any = None,
    ) -> None:
        self._document_repo: IDocumentRepository = (
            document_repo or _NullDocumentRepository()
        )
        self._semantic_dispatcher: SemanticDispatcher = (
            semantic_dispatcher or SemanticDispatcher()
        )
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
            "EdmsDocumentAgent initialised",
            extra={
                "checkpointer": type(checkpointer).__name__,
                "tools": len(all_tools),
                "lean_prompt": USE_LEAN_PROMPT,
            },
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

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
        try:
            checks["state_manager"] = isinstance(self._state_manager, AgentStateManager)
        except Exception:
            checks["state_manager"] = False
        try:
            checks["injector"] = isinstance(self._injector, ToolCallInjector)
        except Exception:
            checks["injector"] = False
        try:
            checks["document_repo"] = self._document_repo is not None
        except Exception:
            checks["document_repo"] = False
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
        # Валидация входных данных
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

        try:
            intent = await self._semantic_dispatcher.classify(
                message=request.message,
                context=context,
            )
        except Exception as exc:
            logger.warning("Intent classification failed: %s — using UNKNOWN", exc)
            intent = UserIntent.UNKNOWN

        context = context.with_intent(intent)

        logger.info(
            "chat: thread=%s intent=%s has_file=%s has_doc=%s has_choice=%s",
            context.thread_id,
            intent.value if intent else "—",
            bool(context.file_path),
            bool(context.document_id),
            bool(human_choice),
        )

        # Получаем метаданные документа
        doc_info: DocumentDto | None = None
        if context.document_id:
            doc_info = await self._document_repo.get_document(
                token=user_token,
                doc_id=context.document_id,
            )

        # Проверяем состояние треда
        try:
            if await self._state_manager.is_thread_broken(context.thread_id):
                logger.warning(
                    "Broken thread detected — repairing",
                    extra={"thread_id": context.thread_id},
                )
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

        # Обычный путь оркестрации
        return await self._loop.run(
            context=context,
            inputs=self._build_inputs(context, request.message, doc_info),
            is_choice_active=False,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _build_inputs(
        self,
        context: ContextParams,
        message: str,
        doc_info: DocumentDto | None,
    ) -> dict[str, Any]:
        semantic_xml = self._build_semantic_xml(doc_info=doc_info)
        system_prompt = PromptBuilder.build(
            context=context,
            intent=context.intent or UserIntent.UNKNOWN,
            semantic_xml=semantic_xml,
            lean=USE_LEAN_PROMPT,
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

        first_name: str = (
            uc.get("first_name") or uc.get("firstName") or uc.get("name") or ""
        ).strip()
        last_name: str = (
            uc.get("last_name") or uc.get("lastName") or uc.get("surname") or ""
        ).strip()
        middle_name: str = (
            uc.get("middle_name") or uc.get("middleName") or uc.get("patronymic") or ""
        ).strip()
        user_id: str | None = (
            uc.get("id")
            or uc.get("user_id")
            or uc.get("userId")
            or uc.get("employeeId")
            or uc.get("employee_id")
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

        reg_number = getattr(doc_info, "regNumber", None) or getattr(
            doc_info, "reservedRegNumber", None
        )
        if reg_number:
            lines.append(f"  <reg_number>{esc(reg_number)}</reg_number>")

        reg_date = getattr(doc_info, "regDate", None)
        if reg_date:
            lines.append(f"  <reg_date>{esc(reg_date)}</reg_date>")

        status = getattr(doc_info, "status", None)
        if status:
            status_val = status.value if hasattr(status, "value") else str(status)
            lines.append(f"  <status>{esc(status_val)}</status>")

        doc_type = getattr(doc_info, "documentType", None)
        if doc_type:
            type_name = getattr(doc_type, "typeName", None)
            if type_name:
                lines.append(f"  <doc_type>{esc(type_name)}</doc_type>")

        category = getattr(doc_info, "docCategoryConstant", None)
        if category:
            cat_val = category.value if hasattr(category, "value") else str(category)
            lines.append(f"  <category>{esc(cat_val)}</category>")

        author = getattr(doc_info, "author", None)
        if author:
            author_name = _user_name(author)
            if author_name:
                lines.append(f"  <author>{esc(author_name)}</author>")

        executor = getattr(doc_info, "responsibleExecutor", None)
        if executor:
            executor_name = _user_name(executor)
            if executor_name:
                lines.append(f"  <executor>{esc(executor_name)}</executor>")

        corr_name = getattr(doc_info, "correspondentName", None)
        if corr_name:
            lines.append(f"  <correspondent>{esc(corr_name)}</correspondent>")

        control_flag = getattr(doc_info, "controlFlag", None)
        if control_flag is not None:
            lines.append(f"  <on_control>{esc(str(control_flag))}</on_control>")

        lines.append("</semantic_context>")
        return "\n".join(lines)


class _NullDocumentRepository:
    """No-op репозиторий — используется когда document_repo не передан."""

    async def get_document(self, token: str, doc_id: str) -> None:
        return None

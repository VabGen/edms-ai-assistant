"""
OrchestrationLoop + handle_human_choice.

The loop replaces the original recursive ``_orchestrate()`` with an explicit
``while`` iteration, which:
  - avoids Python stack growth on busy conversations,
  - makes the iteration counter visible in one place,
  - simplifies ``asyncio`` timeout handling.

``handle_human_choice`` is a free function (not a method) so it can be called
from ``EdmsDocumentAgent.chat()`` without coupling the agent to the loop class.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from edms_ai_assistant.agent.content_extractor import ContentExtractor
from edms_ai_assistant.agent.context import (
    ActionType,
    AgentResponse,
    AgentStatus,
    ContextParams,
    is_valid_uuid,
)
from edms_ai_assistant.agent.graph import GraphBuilder
from edms_ai_assistant.agent.state_manager import AgentStateManager
from edms_ai_assistant.agent.tool_injector import (
    BROKEN_THREAD_SIGNALS,
    DISAMBIGUATION_TOOLS,
    TOOL_DISAMBIG_ID_FIELD,
    ToolCallInjector,
)
from edms_ai_assistant.services.nlp_service import UserIntent

logger = logging.getLogger(__name__)

_MAX_ITERATIONS: int = 10
_EXECUTION_TIMEOUT: float = 120.0

# ---------------------------------------------------------------------------
# Mutation detection
# ---------------------------------------------------------------------------

_MUTATION_SUCCESS_PHRASES: tuple[str, ...] = (
    "успешно добавлен",
    "успешно создан",
    "список ознакомления",
    "поручение создано",
    "поручение успешно",
    "обращение заполнено",
    "обращение успешно",
    "карточка заполнена",
    "добавлено в список",
    "добавлен в список",
    "ознакомление создано",
    "задача создана",
    "заполнение обращения",
    "обращение автоматически заполнен",
    "карточка обращения заполнен",
    "автозаполнен",
    "заголовок обновлен",
    "заголовок изменен",
    "адрес заявителя обновлен",
    "адрес заявителя изменен",
    "телефон в карточке обновлен",
    "изменение выполнино успешно",
    "операция выполнина успешно",
)


def _is_mutation_response(content: str | None) -> bool:
    """Return True if the response describes a successful mutating EDMS operation."""
    if not content:
        return False
    lower = content.lower()
    return any(phrase in lower for phrase in _MUTATION_SUCCESS_PHRASES)


# ---------------------------------------------------------------------------
# Technical content sanitizer
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
_FS_PATH_RE = re.compile(
    r"[A-Za-z]:\\[^\s,;)'\"]{3,}|/(?:tmp|var|home|uploads)/[^\s,;)'\"]{3,}"
)
_HASH_FILENAME_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"_[0-9a-f]{32}\.[a-zA-Z]{2,5}",
    re.IGNORECASE,
)
_SHORT_HASH_RE = re.compile(r"_?[0-9a-f]{32}\.[a-zA-Z]{2,5}\b", re.IGNORECASE)


def _sanitize_line(line: str, context: ContextParams, file_label: str) -> str:
    line = _FS_PATH_RE.sub(file_label, line)
    line = _HASH_FILENAME_RE.sub(file_label, line)
    line = _SHORT_HASH_RE.sub(file_label, line)
    if context.file_path and is_valid_uuid(context.file_path.strip()):
        line = line.replace(context.file_path.strip(), file_label)
    if context.document_id and is_valid_uuid(context.document_id):
        line = line.replace(context.document_id, "«текущего документа»")
    line = re.sub(r"«документ»\s*(?=«)", "", line)
    line = re.sub(r"«документ»_\s*", "", line)
    return line


def sanitize_technical_content(content: str, context: ContextParams) -> str:
    """Remove filesystem paths, UUIDs, and hash filenames from user-visible content.

    Table rows (lines starting with ``|``) are left untouched because they
    intentionally contain the ``id`` column needed for frontend navigation.

    Args:
        content: Raw response string from the LLM.
        context: Execution context (provides file_path, document_id for replacement).

    Returns:
        Sanitized string safe for display.
    """
    file_label = (
        f"«{context.uploaded_file_name}»"
        if context.uploaded_file_name
        else "«загруженный файл»"
    )
    lines = content.split("\n")
    result: list[str] = []
    for line in lines:
        if line.strip().startswith("|"):
            result.append(line)
        else:
            result.append(_sanitize_line(line, context, file_label))
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Interactive status detector
# ---------------------------------------------------------------------------


def detect_interactive_status(
        messages: list[BaseMessage],
) -> dict[str, Any] | None:
    """Scan the last ToolMessage for requires_choice / requires_disambiguation.

    Called after each graph invocation to decide whether the agent must stop
    and surface a choice to the user before proceeding.

    Args:
        messages: Complete LangGraph message chain.

    Returns:
        Serialized ``AgentResponse`` dict, or ``None`` when no action is needed.
    """
    last_tool: ToolMessage | None = next(
        (m for m in reversed(messages) if isinstance(m, ToolMessage)), None
    )
    if last_tool is None:
        return None

    raw = str(last_tool.content).strip()
    if not raw.startswith("{"):
        return None

    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        return None

    status: str = data.get("status", "")
    if status not in ("requires_choice", "requires_disambiguation", "requires_action"):
        return None

    logger.info("Detect interactive: status=%s keys=%s", status, list(data.keys()))

    if status == "requires_choice":
        return _build_requires_choice(data)
    if status == "requires_disambiguation":
        return _build_disambiguation(data)
    if status == "requires_action":
        return _build_requires_action(data)
    return None  # unreachable, but satisfies mypy


# ---- interactive response builders ----------------------------------------


def _build_requires_choice(data: dict[str, Any]) -> dict[str, Any]:
    options: list[dict[str, Any]] = data.get("options", [])
    hint: str = data.get("hint", "extractive")
    hint_reason: str = data.get("hint_reason", "")
    msg: str = data.get("message", "Выберите формат анализа:")

    option_lines = "\n".join(
        f"- **{opt['key']}** — {opt['label']}: {opt['description']}"
        for opt in options
        if isinstance(opt, dict)
    )
    hint_text = (
        f"\n\n💡 *Рекомендация:* **{hint}** — {hint_reason}" if hint_reason else ""
    )
    full_msg = (
        f"{msg}\n\n{option_lines}{hint_text}\n\n"
        "Ответьте: **extractive**, **abstractive** или **thesis**."
    )
    return AgentResponse(
        status=AgentStatus.REQUIRES_ACTION,
        action_type=ActionType.SUMMARIZE_SELECTION,
        message=full_msg,
    ).model_dump()


def _extract_available_list(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the candidate list from a disambiguation payload.

    Tries known key names first, then scans all list-valued fields and
    unwraps nested ``matches`` arrays.
    """
    _KNOWN_KEYS: tuple[str, ...] = (
        "available_attachments", "available_employees", "candidates",
        "employees", "results", "items", "users",
    )
    for key in _KNOWN_KEYS:
        val = data.get(key)
        if isinstance(val, list) and val:
            return val

    for key, val in data.items():
        if key == "options" or not isinstance(val, list) or not val:
            continue
        first = val[0] if val else {}
        if isinstance(first, dict) and "matches" in first and not first.get("id"):
            flat: list[dict[str, Any]] = []
            for group in val:
                flat.extend(group.get("matches", []))
            if flat:
                return flat
        return val

    return []


def _find_candidate_name(messages: list[BaseMessage], choice_id: str) -> str:
    """Find the display name of the selected candidate from the last disambiguation tool output."""
    last_tool_msg = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)
    if last_tool_msg and isinstance(last_tool_msg.content, str):
        try:
            data = json.loads(last_tool_msg.content)
            candidates = _extract_available_list(data)
            for c in candidates:
                if isinstance(c, dict):
                    c_id = str(c.get("id") or c.get("uuid") or c.get("employeeId") or "")
                    if c_id == choice_id:
                        return (
                                c.get("fullName") or c.get("full_name") or c.get("name") or
                                " ".join(filter(None, [c.get("lastName"), c.get("firstName"), c.get("middleName")]))
                                or "Выбранный сотрудник"
                        ).strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    return ""


def _item_to_candidate(item: dict[str, Any]) -> dict[str, str]:
    """Normalise a candidate dict into ``{id, name, dept}``."""
    first = (
            item.get("firstName") or item.get("first_name") or item.get("firstname")
            or item.get("givenName") or ""
    ).strip()
    last = (
            item.get("lastName") or item.get("last_name") or item.get("lastname")
            or item.get("surname") or item.get("familyName") or ""
    ).strip()
    middle = (
            item.get("middleName") or item.get("middle_name") or item.get("patronymic") or ""
    ).strip()
    display_name = (
            item.get("fullName") or item.get("full_name") or item.get("fio")
            or item.get("FIO")
            or " ".join(filter(None, [last, first, middle]))
            or item.get("name") or item.get("username") or item.get("login")
            or (item.get("email") or "").split("@")[0]
            or "Без имени"
    ).strip()
    dept = (
            item.get("department") or item.get("departmentName") or item.get("department_name")
            or item.get("division") or item.get("post") or item.get("position")
            or item.get("jobTitle") or item.get("job_title") or item.get("role") or ""
    ).strip()
    item_id = str(
        item.get("id") or item.get("uuid") or item.get("employeeId")
        or item.get("employee_id") or item.get("userId") or item.get("user_id")
        or item.get("personId") or item.get("person_id") or "?"
    )
    return {"id": item_id, "name": display_name, "dept": dept}


def _build_disambiguation(data: dict[str, Any]) -> dict[str, Any]:
    available = _extract_available_list(data)
    base_msg: str = data.get("message", "Уточните выбор:")
    # Strip residual UUIDs from the message text.
    base_msg = (
            _UUID_RE.sub("", base_msg).strip().rstrip("с «»").strip() or "Уточните выбор:"
    )
    candidates = [_item_to_candidate(item) for item in available if isinstance(item, dict)]
    candidates_json = json.dumps(candidates, ensure_ascii=False)
    # The ``<!--CANDIDATES:...-->`` marker is parsed by AssistantWidget.tsx.
    return AgentResponse(
        status=AgentStatus.REQUIRES_ACTION,
        action_type=ActionType.DISAMBIGUATION,
        message=f"{base_msg}\n\n<!--CANDIDATES:{candidates_json}-->",
    ).model_dump()


def _build_requires_action(data: dict[str, Any]) -> dict[str, Any] | None:
    choices: list[dict[str, Any]] = data.get("choices", [])
    base_msg: str = data.get("message", "Выберите сотрудника:")
    candidates: list[dict[str, str]] = [
        {
            "id": str(item.get("id", "?")),
            "name": (
                    item.get("full_name") or item.get("fullName")
                    or item.get("name") or "Без имени"
            ).strip(),
            "dept": (item.get("department") or item.get("post") or "").strip(),
        }
        for item in choices
        if isinstance(item, dict)
    ]
    if not candidates:
        return None
    candidates_json = json.dumps(candidates, ensure_ascii=False)
    return AgentResponse(
        status=AgentStatus.REQUIRES_ACTION,
        action_type=ActionType.DISAMBIGUATION,
        message=f"{base_msg}\n\n<!--CANDIDATES:{candidates_json}-->",
    ).model_dump()


# ---------------------------------------------------------------------------
# Pending disambiguation call detector (Bug 1 fix)
# ---------------------------------------------------------------------------


def _is_real_tool_result(msg: ToolMessage) -> bool:
    """Return True if ToolMessage contains a real tool result, not an empty stub."""
    content = str(msg.content).strip()
    if not content or content in ("", "{}", "null", "None"):
        return False
    if len(content) < 10:
        return False
    return True


def _find_pending_disambiguation_call(
        messages: list[BaseMessage],
) -> AIMessage | None:
    """Find the last AIMessage with tool_calls that has no real ToolMessage response.

    The graph stops at ``interrupt_before=['tools']``, the tool executes,
    then ``validator`` may add an empty AIMessage — and the graph completes.
    On the next ``handle_human_choice`` we must find that original AIMessage
    with tool_calls and reuse its ``id`` for patching.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, AIMessage):
            continue
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue

        # Check that there is no real ToolMessage with data after it
        subsequent = messages[i + 1:]
        has_real_tool_result = any(
            isinstance(m, ToolMessage) and _is_real_tool_result(m)
            for m in subsequent
        )
        if not has_real_tool_result:
            logger.debug(
                "_find_pending_disambiguation_call: found pending AIMessage "
                "tool=%s id=%s",
                tool_calls[0].get("name"),
                msg.id,
            )
            return msg  # type: ignore[return-value]

    return None


# ---------------------------------------------------------------------------
# OrchestrationLoop
# ---------------------------------------------------------------------------


class OrchestrationLoop:
    """Iterative ReAct orchestration loop.

    Replaces the original recursive ``_orchestrate()`` with an explicit
    ``while`` loop so that the call stack depth is bounded and the iteration
    counter is visible without reading return values.

    Args:
        state_manager: Provides graph state access.
        injector: Patches tool_call arguments.
        all_tools: Full tool list for intent-based binding.
        graph_builder: Provides ``set_model()`` to update the LLM binding
            inside the compiled graph's ``call_model`` closure.
        model: The base LLM instance (unbound to tools).
    """

    def __init__(
            self,
            state_manager: AgentStateManager,
            injector: ToolCallInjector,
            all_tools: list[Any],
            graph_builder: GraphBuilder,
            model: Any,
    ) -> None:
        self._state_manager = state_manager
        self._injector = injector
        self._all_tools = all_tools
        self._graph_builder = graph_builder
        self._model = model
        # Cache of tool-bound model instances keyed by sorted tool name set.
        self._tool_binding_cache: dict[str, Any] = {}

    async def run(
            self,
            context: ContextParams,
            inputs: dict[str, Any] | None,
            is_choice_active: bool,
    ) -> dict[str, Any]:
        """Run the iterative ReAct loop until completion or ``_MAX_ITERATIONS``.

        Args:
            context: Immutable execution context.
            inputs: Initial graph inputs, or ``None`` to resume from interrupt.
            is_choice_active: ``True`` when resuming from a HITL choice.

        Returns:
            Serialized ``AgentResponse`` dict.
        """
        current_inputs = inputs
        current_is_choice = is_choice_active

        for iteration in range(_MAX_ITERATIONS + 1):

            if iteration > _MAX_ITERATIONS:
                logger.error(
                    "Max iterations exceeded",
                    extra={"thread_id": context.thread_id, "max": _MAX_ITERATIONS},
                )
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Превышен лимит итераций обработки.",
                ).model_dump()

            # --- Bind tools for the detected intent (first iteration only) ---
            if iteration == 0:
                self._bind_tools_for_intent(context)

            # --- Invoke the graph ---
            try:
                await self._state_manager.invoke(
                    inputs=current_inputs,
                    thread_id=context.thread_id,
                    timeout=_EXECUTION_TIMEOUT,
                )
            except TimeoutError:
                logger.error(
                    "Execution timeout",
                    extra={
                        "thread_id": context.thread_id,
                        "timeout": _EXECUTION_TIMEOUT,
                        "iteration": iteration,
                    },
                )
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Превышено время ожидания выполнения.",
                ).model_dump()
            except Exception as exc:
                return await self._handle_error(exc, context, iteration)

            # --- Inspect state ---
            state = await self._state_manager.get_state(context.thread_id)
            messages: list[BaseMessage] = state.values.get("messages", [])

            logger.debug(
                "Iteration %d: messages=%d last=%s next=%s",
                iteration,
                len(messages),
                type(messages[-1]).__name__ if messages else "—",
                list(state.next) if state.next else [],
            )

            if not messages:
                return AgentResponse(
                    status=AgentStatus.ERROR, message="Пустое состояние агента."
                ).model_dump()

            last_msg = messages[-1]
            last_is_tool = isinstance(last_msg, ToolMessage)
            last_has_calls = isinstance(last_msg, AIMessage) and bool(
                getattr(last_msg, "tool_calls", None)
            )
            is_finished = not state.next and not last_is_tool and not last_has_calls

            if is_finished:
                return self._build_final_response(messages, context)

            if not last_has_calls:
                # Unexpected state: no tool_calls to patch — surface what we have.
                return self._build_final_response(messages, context)

            # --- Enforce single tool_call ---
            raw_calls: list[dict[str, Any]] = list(last_msg.tool_calls)
            if len(raw_calls) > 1:
                logger.warning(
                    "Parallel tool_calls — keeping only the first: kept=%s dropped=%s",
                    raw_calls[0]["name"],
                    [tc["name"] for tc in raw_calls[1:]],
                )
                raw_calls = raw_calls[:1]

            # --- Patch args ---
            last_tool_text = ContentExtractor.extract_last_tool_text(messages)
            patched = self._injector.patch(
                raw_calls=raw_calls,
                context=context,
                messages=messages,
                last_tool_text=last_tool_text,
                is_choice_active=current_is_choice,
            )

            await self._state_manager.update_state(
                context.thread_id,
                [AIMessage(
                    content=last_msg.content or "",
                    tool_calls=patched,
                    id=last_msg.id,
                )],
                as_node="agent",
            )

            # Propagate is_choice_active only for tools that consume it.
            if current_is_choice and patched:
                last_patched_name = patched[-1]["name"]
                current_is_choice = last_patched_name in (
                    "doc_compare_attachment_with_local",
                    "doc_summarize_text",
                )
            else:
                current_is_choice = False

            current_inputs = None  # resume from interrupt on subsequent iterations

        # Unreachable, but satisfies mypy.
        return AgentResponse(
            status=AgentStatus.ERROR, message="Превышен лимит итераций обработки."
        ).model_dump()

    # -----------------------------------------------------------------------
    # Tool binding
    # -----------------------------------------------------------------------

    def _bind_tools_for_intent(self, context: ContextParams) -> None:
        """Select the minimal tool subset for the detected intent and bind it.

        The bound model is cached by its tool-name set so rebinding is O(1) for
        repeated calls with the same intent within the same agent lifetime.
        """
        from edms_ai_assistant.tools.router import (  # noqa: PLC0415
            estimate_tools_tokens,
            get_tools_for_intent,
        )

        active_intent: UserIntent = context.intent or UserIntent.UNKNOWN
        is_appeal = context.user_context.get("doc_category", "") == "APPEAL"
        selected = get_tools_for_intent(active_intent, self._all_tools, include_appeal=is_appeal)

        cache_key = ",".join(sorted(getattr(t, "name", "") for t in selected))
        if cache_key not in self._tool_binding_cache:
            bound = self._model.bind_tools(selected)
            self._tool_binding_cache[cache_key] = bound
            logger.info(
                "Tool binding created: intent=%s tools=%d (~%d tokens)",
                active_intent.value,
                len(selected),
                estimate_tools_tokens(selected),
            )

        # Push the bound model into the graph's call_model closure.
        self._graph_builder.set_model(self._tool_binding_cache[cache_key])

    # -----------------------------------------------------------------------
    # Response builders
    # -----------------------------------------------------------------------

    def _build_final_response(
            self, messages: list[BaseMessage], context: ContextParams
    ) -> dict[str, Any]:
        """Build and return the serialized AgentResponse."""
        interactive = detect_interactive_status(messages)
        if interactive:
            logger.info(
                "Interactive status detected",
                extra={"status": interactive.get("status"), "thread_id": context.thread_id},
            )
            return interactive

        # Probe for a compliance result from the last tool section.
        compliance_data: dict[str, Any] | None = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                break
            if isinstance(msg, ToolMessage):
                compliance_data = ContentExtractor.extract_compliance_data(messages)
                break

        final_content = ContentExtractor.extract_final_content(messages)
        navigate_url = ContentExtractor.extract_navigate_url(messages)

        metadata: dict[str, Any] = {}
        if compliance_data:
            metadata["compliance"] = compliance_data
            logger.info(
                "Compliance added to metadata: overall=%s fields=%d",
                compliance_data.get("overall"),
                len(compliance_data.get("fields", [])),
            )

        if final_content:
            final_content = ContentExtractor.clean_json_artifacts(final_content)
            final_content = sanitize_technical_content(final_content, context)
            return AgentResponse(
                status=AgentStatus.SUCCESS,
                content=final_content,
                requires_reload=_is_mutation_response(final_content),
                navigate_url=navigate_url,
                metadata=metadata,
            ).model_dump()

        logger.warning("No final content found", extra={"thread_id": context.thread_id})
        return AgentResponse(
            status=AgentStatus.SUCCESS,
            content="Операция завершена.",
            navigate_url=navigate_url,
            metadata=metadata,
        ).model_dump()

    async def _handle_error(
            self,
            exc: Exception,
            context: ContextParams,
            iteration: int,
    ) -> dict[str, Any]:
        """Handle orchestration exceptions, repairing broken threads when possible.

        On the first iteration, broken-thread signals trigger ``repair_thread``
        followed by a single retry.  All other errors are surfaced immediately.

        Args:
            exc: The exception raised during graph invocation.
            context: Immutable execution context.
            iteration: Current iteration number (repair only happens on 0).

        Returns:
            Serialized ``AgentResponse`` dict describing the error or retry result.
        """
        err_str = str(exc)
        logger.error(
            "Orchestration error at iteration %d: %s",
            iteration,
            err_str[:300],
            extra={"thread_id": context.thread_id},
            exc_info=True,
        )

        is_broken_thread = any(sig in err_str for sig in BROKEN_THREAD_SIGNALS)

        if iteration == 0 and is_broken_thread:
            logger.warning(
                "Broken thread detected — attempting repair",
                extra={"thread_id": context.thread_id},
            )
            repaired = await self._state_manager.repair_thread(context.thread_id)
            if repaired:
                logger.info(
                    "Thread repaired — retrying iteration 0",
                    extra={"thread_id": context.thread_id},
                )
                # Re-run the loop with a single retry (iteration=0 again via recursion).
                # We call run() recursively here — safe because retry depth is exactly 1.
                return await self.run(
                    context=context,
                    inputs=None,  # resume from repaired state
                    is_choice_active=False,
                )
            else:
                logger.error(
                    "Thread repair failed — returning error to user",
                    extra={"thread_id": context.thread_id},
                )
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message=(
                        "Произошла техническая ошибка при обработке запроса. "
                        "Пожалуйста, начните новый диалог."
                    ),
                ).model_dump()

        if "TimeoutError" in err_str or "asyncio.TimeoutError" in err_str:
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Превышено время ожидания выполнения. Попробуйте ещё раз.",
            ).model_dump()

        if "rate_limit" in err_str.lower() or "429" in err_str:
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Сервис временно перегружен. Пожалуйста, повторите через несколько секунд.",
            ).model_dump()

        return AgentResponse(
            status=AgentStatus.ERROR,
            message="Произошла внутренняя ошибка при обработке запроса. Попробуйте переформулировать.",
        ).model_dump()


# ---------------------------------------------------------------------------
# handle_human_choice
# ---------------------------------------------------------------------------

async def handle_human_choice(
        context: ContextParams,
        human_choice: str,
        state_manager: AgentStateManager,
        loop: OrchestrationLoop,
) -> dict[str, Any]:
    choice = human_choice.strip()

    if choice.lower().startswith("fix_field:"):
        parts = choice.split(":", 2)
        if len(parts) == 3:
            field_name, field_value = parts[1].strip(), parts[2].strip()
            fix_message = f"Обнови поле «{field_name}» документа значением «{field_value}»."
            return await loop.run(
                context=context,
                inputs={"messages": [HumanMessage(content=fix_message)]},
                is_choice_active=False,
            )

    state = await state_manager.get_state(context.thread_id)
    messages: list[BaseMessage] = state.values.get("messages", [])

    pending_ai: AIMessage | None = _find_pending_disambiguation_call(messages)

    if pending_ai is None:
        logger.info(
            "handle_human_choice: no pending tool_call found — attempting reconstruction",
            extra={"thread_id": context.thread_id, "choice": choice[:40]},
        )

        last_tool_msg = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)

        if last_tool_msg and isinstance(last_tool_msg.content, str) and is_valid_uuid(choice):
            try:
                data = json.loads(last_tool_msg.content)
                if data.get("status") in ("requires_disambiguation", "requires_action"):
                    tool_call_id = getattr(last_tool_msg, "tool_call_id", None)
                    t_name = getattr(last_tool_msg, "name", None)

                    original_ai_msg = None
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage):
                            for tc in getattr(msg, "tool_calls", []):
                                if tc.get("id") == tool_call_id:
                                    original_ai_msg = msg
                                    break
                        if original_ai_msg:
                            break

                    if original_ai_msg and tool_call_id and t_name:
                        raw_calls = list(original_ai_msg.tool_calls)
                        t_args = dict(raw_calls[0]["args"])

                        selected_name = _find_candidate_name(messages, choice)

                        id_field = TOOL_DISAMBIG_ID_FIELD.get(t_name, "selected_employee_ids")

                        selected_ids = [s.strip() for s in choice.replace(";", ",").split(",") if s.strip()]
                        valid_ids = [sid for sid in selected_ids if is_valid_uuid(sid)]

                        if valid_ids:
                            t_args[id_field] = valid_ids[0] if t_name == "doc_control" else valid_ids
                            patched_calls = [{"name": t_name, "args": t_args, "id": tool_call_id}]

                            logger.info(
                                "Reconstructed tool call %s with %s=%s",
                                t_name, id_field, valid_ids,
                                extra={"thread_id": context.thread_id},
                            )

                            await state_manager.update_state(
                                context.thread_id,
                                [AIMessage(
                                    content=f"Выбран(а): {selected_name}." if selected_name else "",
                                    tool_calls=patched_calls,
                                    id=tool_call_id,
                                )],
                                as_node="agent",
                            )

                            return await loop.run(
                                context=context,
                                inputs=None,
                                is_choice_active=True,
                            )
            except (json.JSONDecodeError, AttributeError, Exception) as exc:
                logger.warning("Failed to reconstruct tool call: %s", exc)

        instruction = choice
        if last_tool_msg and isinstance(last_tool_msg.content, str):
            try:
                data = json.loads(last_tool_msg.content)
                if data.get("status") in ("requires_choice",):
                    if choice.lower() in {"extractive", "abstractive", "thesis"}:
                        instruction = (
                            f"Пользователь выбрал формат анализа: **{choice.lower()}**. "
                            f"Вызови инструмент `doc_summarize_text` с аргументом summary_type={choice.lower()}."
                        )
            except json.JSONDecodeError:
                pass

        return await loop.run(
            context=context,
            inputs={"messages": [HumanMessage(content=instruction)]},
            is_choice_active=True,
        )

    raw_calls: list[dict[str, Any]] = list(pending_ai.tool_calls)
    t_name: str = raw_calls[0]["name"]
    t_args: dict[str, Any] = dict(raw_calls[0]["args"])
    t_id: str = raw_calls[0]["id"]
    patched_calls = raw_calls

    if t_name == "doc_summarize_text":
        valid_types = {"extractive", "abstractive", "thesis"}
        t_args["summary_type"] = choice.lower() if choice.lower() in valid_types else "extractive"
        patched_calls = [{"name": t_name, "args": t_args, "id": t_id}]
        logger.info(
            "Injected summary_type=%s for doc_summarize_text",
            choice.lower(),
            extra={"thread_id": context.thread_id},
        )

    elif t_name in DISAMBIGUATION_TOOLS:
        id_field = TOOL_DISAMBIG_ID_FIELD.get(t_name, "selected_employee_ids")
        selected_ids = [s.strip() for s in choice.replace(";", ",").split(",") if s.strip()]
        valid_ids = [sid for sid in selected_ids if is_valid_uuid(sid)]
        if not valid_ids:
            logger.warning(
                "handle_human_choice: no valid UUIDs in choice for %s — routing as message",
                t_name,
                extra={"thread_id": context.thread_id},
            )
            return await loop.run(
                context=context,
                inputs={"messages": [HumanMessage(content=choice)]},
                is_choice_active=True,
            )
        t_args[id_field] = valid_ids[0] if t_name == "doc_control" else valid_ids
        patched_calls = [{"name": t_name, "args": t_args, "id": t_id}]
        logger.info(
            "Injected %s=%s for %s",
            id_field, valid_ids, t_name,
            extra={"thread_id": context.thread_id},
        )

    elif t_name == "doc_compare_attachment_with_local":
        candidate = choice.strip()
        if is_valid_uuid(candidate):
            t_args["attachment_id"] = candidate
            patched_calls = [{"name": t_name, "args": t_args, "id": t_id}]
            logger.info(
                "Injected attachment_id=%s for doc_compare_attachment_with_local",
                candidate[:8],
                extra={"thread_id": context.thread_id},
            )
        else:
            logger.warning(
                "Choice '%s' is not a UUID — routing as new message for compare",
                candidate[:40],
            )
            return await loop.run(
                context=context,
                inputs={"messages": [HumanMessage(content=choice)]},
                is_choice_active=True,
            )

    elif t_name == "task_create_tool":
        id_field = "selected_employee_ids"
        selected_ids = [s.strip() for s in choice.replace(";", ",").split(",") if s.strip()]
        valid_ids = [sid for sid in selected_ids if is_valid_uuid(sid)]
        if valid_ids:
            t_args[id_field] = valid_ids
            patched_calls = [{"name": t_name, "args": t_args, "id": t_id}]

    selected_name = _find_candidate_name(messages, choice)

    await state_manager.update_state(
        context.thread_id,
        [AIMessage(
            content=f"Выбран(а): {selected_name}." if selected_name else (pending_ai.content or ""),
            tool_calls=patched_calls,
            id=pending_ai.id,
        )],
        as_node="agent",
    )

    return await loop.run(
        context=context,
        inputs=None,
        is_choice_active=True,
    )

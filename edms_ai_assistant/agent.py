import logging
import asyncio
import re
from typing import Dict, List, Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime

from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.tools import all_tools

logger = logging.getLogger(__name__)

UUID_PATTERN = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{12}', re.I)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


class EdmsDocumentAgent:
    def __init__(self):
        self.model = get_chat_model()
        self.tools = all_tools
        self.checkpointer = MemorySaver()

        self.tool_manifesto_template = (
            "### ROLE\n"
            "You are a STRICT EXPERT ANALYST for EDMS (Electronic Document Management System).\n"
            "Your goal is to provide data-driven insights using a predefined toolchain. NEVER answer from internal knowledge if a tool can provide the data.\n\n"

            "### CURRENT CONTEXT\n"
            "- User: {user_name}\n"
            "- Target Document ID (active_id): {context_ui_id}\n\n"

            "### PRIMARY DIRECTIVES (CRITICAL)\n"
            "1. **MANDATORY TOOL USE**: If the user asks about 'the page', 'the document', or 'summary', you MUST NOT answer without calling tools.\n"
            "2. **CHAIN OF THOUGHT**: Always follow this sequence:\n"
            "   - [Step 1: Meta-data] Call `doc_get_details` to see what files exist.\n"
            "   - [Step 2: Content] Call `doc_get_file_content` using the `id` found in Step 1.\n"
            "   - [Step 3: Analysis] Call `doc_summarize_text` with the content from Step 2.\n"
            "3. **FORCE RE-ANALYSIS**: Even if you have information about the document in the chat history, if the user explicitly asks for a 'summary' (сводка), 'analysis' (анализ), or 'report' (отчет), you MUST call `doc_summarize_text` again. DO NOT answer from memory.\n\n"

            "### SUMMARY RULES (FOR BUTTONS UI)\n"
            "1. **STRICT NULL POLICY**: When calling `doc_summarize_text`, you MUST leave `summary_type` EMPTY (null) unless the user explicitly used keywords: 'тезисы' (thesis), 'факты' (extractive), 'детально/пересказ' (abstractive).\n"
            "2. If the user says 'сделай сводку', 'проанализируй', 'о чем файл' — call `doc_summarize_text` with ONLY the `text` argument. This triggers the format selection UI.\n\n"

            "### TOOL EXECUTION LOGIC\n"
            "#### SCENARIO A: General inquiry (e.g., 'Analyze this page', 'What is this?')\n"
            "- ACTION: Immediately call `doc_get_details(document_id=\"{context_ui_id}\")`.\n"
            "- REASON: You need to identify attachments before analysis.\n\n"

            "#### SCENARIO B: File-specific inquiry (e.g., 'Summarize Cover Letter')\n"
            "- ACTION 1: Call `doc_get_details` to find the exact `attachment_id` for 'Cover Letter'.\n"
            "- ACTION 2: Use that ID to call `doc_get_file_content`.\n"
            "- ACTION 3: Pass text to `doc_summarize_text` with `summary_type=null`.\n\n"

            "#### SCENARIO C: Local file path is provided (LOCAL_FILE is not empty)\n"
            "- ACTION: Prioritize `read_local_file_content` over EDMS API calls.\n\n"

            "### GUARDRAILS & FORMATTING\n"
            "- **NO SELF-ANALYSIS**: Do not summarize text yourself. You are only the 'orchestrator'. The actual summary MUST come from `doc_summarize_text`.\n"
            "- **INTERRUPT RULE**: If you call `doc_summarize_text` without a `summary_type`, STOP your response immediately after the tool call. The system will handle the UI selection.\n"
            "- **LANGUAGE**: Respond in Russian, addressing the user as {user_name}.\n"
            "- **ERROR**: If a tool fails, state: 'К сожалению, автоматический анализ сейчас недоступен. Попробуйте позже.'\n\n"

            "### FINAL REQUIREMENT\n"
            "Every response MUST end with a Tool Call until the final analytical Summary is delivered."
        )

        self.agent = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        async def call_model(state: AgentState):
            model_with_tools = self.model.bind_tools(self.tools)
            response = await model_with_tools.ainvoke(state["messages"])
            return {"messages": [response]}

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_edge(START, "agent")

        def should_continue(state: AgentState):
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tools"
            return END

        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")

        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["tools"]
        )

    async def _clear_stuck_tool_calls(self, config, last_msg: AIMessage):
        if not last_msg.tool_calls: return
        error_msgs = [ToolMessage(tool_call_id=tc["id"], content="Прервано пользователем.") for tc in
                      last_msg.tool_calls]
        await self.agent.aupdate_state(config, {"messages": error_msgs}, as_node="tools")

    async def chat(self, message: str, user_token: str, context_ui_id: str = None,
                   thread_id: str = None, user_context: Dict = None,
                   file_path: str = None, human_choice: str = None) -> Dict:

        config = {"configurable": {"thread_id": thread_id or "default"}}
        user_context = user_context or {}
        user_name = (user_context.get('firstName') or user_context.get('name') or "пользователь").strip()

        state = await self.agent.aget_state(config)

        # 1. Если кнопка нажата в процессе (прерывание)
        if human_choice and state.next:
            last_msg = state.values["messages"][-1]
            fixed_calls = []
            for tc in getattr(last_msg, "tool_calls", []):
                args = dict(tc["args"])
                if tc["name"] == "doc_summarize_text":
                    args["summary_type"] = human_choice
                fixed_calls.append({"name": tc["name"], "args": args, "id": tc["id"]})

            await self.agent.aupdate_state(config, {
                "messages": [AIMessage(content="", tool_calls=fixed_calls, id=last_msg.id)]},
                                           as_node="agent")
            return await self._orchestrate(None, config, user_token, context_ui_id, file_path, is_choice_active=True)

        effective_message = message
        if human_choice and not state.next:
            format_map = {"extractive": "факты", "thesis": "тезисы", "abstractive": "пересказ"}
            format_word = format_map.get(human_choice, human_choice)
            effective_message = f"{message}. Формат анализа: {format_word}."

        if state.next and message:
            await self._clear_stuck_tool_calls(config, state.values["messages"][-1])

        current_date = datetime.now().strftime("%d.%m.%Y")
        manifesto = self.tool_manifesto_template.format(
            context_ui_id=context_ui_id or "Не указан",
            user_name=user_name
        )
        env = (
            f"\n### ENVIRONMENT\n"
            f"CURRENT_DATE: {current_date}\n"
            f"ACTIVE_DOC_ID: {context_ui_id}\n"
            f"LOCAL_FILE: {file_path}\n"
        )

        if not state.values.get("messages"):
            inputs = {"messages": [SystemMessage(content=manifesto + env), HumanMessage(content=effective_message)]}
        else:
            inputs = {"messages": [HumanMessage(content=effective_message)]}

        return await self._orchestrate(inputs, config, user_token, context_ui_id, file_path,
                                       is_choice_active=bool(human_choice))

    async def _orchestrate(self, inputs, config, token, doc_id, file_path, is_choice_active=False, iteration=0):
        if iteration > 10:
            return {"status": "error", "message": "Слишком много итераций"}

        try:
            await asyncio.wait_for(self.agent.ainvoke(inputs, config=config), timeout=210.0)
            state = await self.agent.aget_state(config)
            messages = state.values.get("messages", [])
            if not messages: return {"status": "error", "message": "Пустая цепочка"}

            last_msg = messages[-1]

            if not state.next or not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
                for m in reversed(messages):
                    if isinstance(m, AIMessage) and m.content and not m.tool_calls:
                        if "предыдущий анализ отменен" not in m.content.lower():
                            return {"status": "success", "content": m.content}
                return {"status": "success", "content": "Запрос обработан."}

            fixed_calls = []
            format_keywords = {"факты": "extractive", "тезисы": "thesis", "пересказ": "abstractive"}

            newly_read_content = None
            for m in reversed(messages):
                if isinstance(m, ToolMessage):
                    try:
                        import json
                        data = json.loads(m.content)
                        if isinstance(data, dict) and data.get("status") == "success" and data.get("content"):
                            newly_read_content = data["content"]
                            break
                    except:
                        if len(str(m.content)) > 100:
                            newly_read_content = m.content
                            break

            for tc in last_msg.tool_calls:
                new_args = dict(tc["args"])
                new_args["token"] = token

                if doc_id and "document_id" in new_args:
                    new_args["document_id"] = doc_id

                if file_path and tc["name"] in ["doc_get_file_content", "read_local_file_content"]:
                    new_args["file_path"] = file_path
                    if "attachment_id" in new_args: new_args["attachment_id"] = None

                if tc["name"] == "doc_summarize_text":
                    if newly_read_content:
                        new_args["text"] = newly_read_content

                    elif not new_args.get("text") or len(str(new_args.get("text"))) < 20:
                        new_args["text"] = newly_read_content  # Мы уже нашли его выше

                    if not new_args.get("summary_type"):
                        format_found = False
                        for m in reversed(messages):
                            if isinstance(m, HumanMessage):
                                text = str(m.content).lower()
                                for kw, fmt in format_keywords.items():
                                    if re.search(rf'\b{kw}\b', text):
                                        new_args["summary_type"] = fmt
                                        is_choice_active = True
                                        format_found = True
                                        break
                            if format_found: break

                fixed_calls.append({"name": tc["name"], "args": new_args, "id": tc["id"]})

            summary_call = next((tc for tc in fixed_calls if tc["name"] == "doc_summarize_text"), None)

            if summary_call:
                user_msg_text = next(
                    (str(m.content).lower() for m in reversed(messages) if isinstance(m, HumanMessage)), "")
                has_type_in_msg = any(re.search(rf'\b{kw}\b', user_msg_text) for kw in format_keywords.keys())
                has_type_in_args = bool(summary_call["args"].get("summary_type"))

                if not (has_type_in_msg or has_type_in_args or is_choice_active):
                    await self.agent.aupdate_state(config, {
                        "messages": [AIMessage(content="", tool_calls=fixed_calls, id=last_msg.id)]
                    }, as_node="agent")

                    logger.info("Interrupting for format selection buttons...")
                    return {
                        "status": "requires_action",
                        "action_type": "summarize_selection",
                        "message": "Пожалуйста, выберите формат анализа документа."
                    }

            await self.agent.aupdate_state(config, {
                "messages": [AIMessage(content=last_msg.content, tool_calls=fixed_calls, id=last_msg.id)]
            }, as_node="agent")

            return await self._orchestrate(None, config, token, doc_id, file_path, is_choice_active=True,
                                           iteration=iteration + 1)

        except asyncio.TimeoutError:
            return {"status": "error", "message": "Превышено время ожидания."}
        except Exception as e:
            logger.error(f"Error in orchestrate: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

# edms_ai_assistant/agent.py
import logging
import asyncio
import re
from typing import Dict, List, Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

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
            "### ROLE: EXPERT EDMS ANALYST (СЭД)\n"
            "Ты — высококвалифицированный аналитик систем электронного документооборота. "
            "Твоя задача: профессиональный анализ документов с использованием строгого алгоритма работы с инструментами.\n\n"

            "### CONTEXT:\n"
            "- Текущий пользователь: {user_name}\n"
            "- ID активного документа в СЭД: {context_ui_id}\n\n"

            "### OPERATION ALGORITHM (ALGORITHMIC STEPS):\n"
            "Соблюдай последовательность действий (Chain-of-Thought):\n\n"
            "1. **IDENTIFICATION**: При запросе анализа файла по его НАЗВАНИЮ (например, 'Обложка дела'):\n"
            "   - ВСЕГДА начинай с вызова `doc_get_details(document_id=active_id)`.\n"
            "   - В полученном JSON найди массив `attachmentDocument` и извлеки `id` (attachment_id) файла, имя которого максимально совпадает с запросом.\n\n"
            "2. **EXTRACTION**: Получив ID вложения:\n"
            "   - Вызови `doc_get_file_content(attachment_id=...)` для получения текста.\n"
            "   - Если указан путь в LOCAL_FILE, используй `read_local_file_content` как приоритетный источник.\n\n"
            "3. **PROCESSING (STRICT RULE)**:\n"
            "   - Любой вопрос о содержании или деталях файла (например: 'о чем этот файл?', 'что в нем?', 'анализ') ОБЯЗАТЕЛЬНО должен проходить через `doc_summarize_text`.\n"
            "   - КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО отвечать самостоятельно.\n"
            "   - ПОРЯДОК ДЕЙСТВИЙ: Сначала получи текст файла, затем вызови инструмент `doc_summarize_text` (не указывай тип суммаризации, если пользователь не просил конкретный формат).\n"
            "   - Это запустит процесс выбора формата в интерфейсе.\n"

            "### STRICT CONSTRAINTS (GUARDRAILS):\n"
            "- **NO HALLUCINATIONS**: Запрещено пересказывать текст своими словами. Используй ТОЛЬКО результат инструмента `doc_summarize_text`.\n"
            "- **ERROR HANDLING**: Если инструмент суммаризации вернул техническую ошибку, ответь: 'К сожалению, автоматическая суммаризация сейчас недоступна. Попробуйте повторить запрос позже', не пытаясь анализировать текст самостоятельно.\n"
            "- **COMMUNICATION**: Соблюдай деловой этикет, обращайся к пользователю по имени {user_name}.\n\n"

            "### EXECUTION RULE:\n"
            "Твой ответ должен всегда завершаться вызовом инструмента, если цепочка анализа (Details -> Content -> Summary) не завершена."
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
        user_name = f"{user_context.get('firstName', '')}".strip() or "пользователь"

        state = await self.agent.aget_state(config)

        if human_choice and state.next:
            last_msg = state.values["messages"][-1]
            fixed_calls = []
            for tc in getattr(last_msg, "tool_calls", []):
                args = dict(tc["args"])
                if tc["name"] == "doc_summarize_text":
                    args["summary_type"] = human_choice
                fixed_calls.append({"name": tc["name"], "args": args, "id": tc["id"]})

            await self.agent.aupdate_state(config, {
                "messages": [AIMessage(content=last_msg.content, tool_calls=fixed_calls, id=last_msg.id)]},
                                           as_node="agent")
            return await self._orchestrate(None, config, user_token, context_ui_id, file_path, is_choice_active=True)

        if state.next and message:
            await self._clear_stuck_tool_calls(config, state.values["messages"][-1])

        manifesto = self.tool_manifesto_template.format(context_ui_id=context_ui_id or "Не указан", user_name=user_name)
        env = f"\n### ENVIRONMENT\nACTIVE_DOC_ID: {context_ui_id}\nLOCAL_FILE: {file_path}\n"

        if not state.values.get("messages"):
            inputs = {"messages": [SystemMessage(content=manifesto + env), HumanMessage(content=message)]}
        else:
            inputs = {"messages": [HumanMessage(content=message)]}

        return await self._orchestrate(inputs, config, user_token, context_ui_id, file_path)

    async def _orchestrate(self, inputs, config, token, doc_id, file_path, is_choice_active=False, iteration=0):
        if iteration > 10:
            return {"status": "error", "message": "Слишком много итераций (возможная зацикленность)"}

        try:
            await asyncio.wait_for(self.agent.ainvoke(inputs, config=config), timeout=210.0)

            state = await self.agent.aget_state(config)
            messages = state.values.get("messages", [])
            if not messages:
                return {"status": "error", "message": "Цепочка сообщений пуста"}

            last_msg = messages[-1]

            if not state.next or not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
                for m in reversed(messages):
                    if isinstance(m, AIMessage) and m.content and not m.tool_calls:
                        if "предыдущий анализ отменен" not in m.content.lower():
                            return {"status": "success", "content": m.content}
                return {"status": "success", "content": "Запрос обработан."}

            fixed_calls = []
            for tc in last_msg.tool_calls:
                new_args = dict(tc["args"])
                new_args["token"] = token

                if tc["name"] not in ["employee_search_tool", "read_local_file_content"]:
                    new_args["document_id"] = doc_id

                if (tc["name"] in ["doc_get_file_content", "read_local_file_content"]) and file_path:
                    if not new_args.get("attachment_id") and not new_args.get("file_path"):
                        new_args["file_path"] = file_path

                if tc["name"] == "doc_get_file_content" and not new_args.get("attachment_id") and not new_args.get(
                        "file_path"):
                    for m in reversed(messages):
                        if isinstance(m, ToolMessage) and UUID_PATTERN.search(str(m.content)):
                            uuids = UUID_PATTERN.findall(str(m.content))
                            if uuids:
                                new_args["attachment_id"] = uuids[0]
                                logger.info(f"Автоподстановка ID вложения: {uuids[0]}")
                                break

                if tc["name"] == "doc_summarize_text" and (
                        not new_args.get("text") or len(str(new_args.get("text"))) < 20):
                    logger.info("Суммаризатор вызван без контента, извлекаю из истории...")
                    import json
                    for m in reversed(messages):
                        if isinstance(m, ToolMessage):
                            try:
                                data = json.loads(m.content)
                                if isinstance(data, dict):
                                    if data.get("status") == "success" and data.get("content"):
                                        new_args["text"] = data["content"]
                                        break
                                    if data.get("status") == "error":
                                        break
                            except:
                                if len(str(m.content)) > 100:
                                    new_args["text"] = m.content
                                    break

                fixed_calls.append({"name": tc["name"], "args": new_args, "id": tc["id"]})

            summary_call = next((tc for tc in fixed_calls if tc["name"] == "doc_summarize_text"), None)
            if summary_call and not is_choice_active:
                user_msg = next((m.content.lower() for m in reversed(messages) if isinstance(m, HumanMessage)), "")
                keywords = ["факты", "тезисы", "extractive", "thesis", "пересказ", "абстракт", "подробно"]

                if not any(kw in user_msg for kw in keywords):
                    await self.agent.aupdate_state(config, {
                        "messages": [AIMessage(content=last_msg.content, tool_calls=fixed_calls, id=last_msg.id)]
                    }, as_node="agent")

                    return {
                        "status": "requires_action",
                        "action_type": "summarize_selection",
                        "message": "В каком формате подготовить анализ?"
                    }

            await self.agent.aupdate_state(config, {
                "messages": [AIMessage(content=last_msg.content, tool_calls=fixed_calls, id=last_msg.id)]
            }, as_node="agent")

            return await self._orchestrate(None, config, token, doc_id, file_path, is_choice_active=True,
                                           iteration=iteration + 1)

        except asyncio.TimeoutError:
            logger.error("Timeout во время выполнения оркестрации")
            return {"status": "error", "message": "Превышено время ожидания ответа от ИИ."}
        except Exception as e:
            logger.error(f"Ошибка в _orchestrate: {e}", exc_info=True)
            return {"status": "error", "message": f"Ошибка оркестрации: {str(e)}"}

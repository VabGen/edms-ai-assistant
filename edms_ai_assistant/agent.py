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
            "### ROLE: EXPERT EDMS ANALYST (СЭД)\n"
            "Ты — высококвалифицированный аналитик систем электронного документооборота. "
            "Твоя задача: профессиональный анализ документов с использованием строгого алгоритма работы с инструментами.\n\n"

            "### CONTEXT:\n"
            "- Текущий пользователь: {user_name}\n"
            "- ID активного документа в СЭД: {context_ui_id}\n\n"

            "### OPERATION ALGORITHM (ALGORITHMIC STEPS):\n"
            "Соблюдай последовательность действий (Chain-of-Thought):\n\n"
            "1. **IDENTIFICATION**: При запросе анализа файла по его НАЗВАНИЮ:\n"
            "   - ВСЕГДА начинай с вызова `doc_get_details(document_id=active_id)`.\n"
            "   - Найди в `attachmentDocument` нужный `id` (attachment_id).\n\n"
            "2. **EXTRACTION**: Получив ID вложения:\n"
            "   - Вызови `doc_get_file_content(attachment_id=...)` для получения текста.\n\n"
            "3. **PROCESSING (STRICT RULE)**:\n"
            "   - Любой запрос анализа ОБЯЗАТЕЛЬНО проходит через `doc_summarize_text`.\n"
            "   - ВАЖНО: Если в сообщении пользователя УЖЕ указан формат (факты, тезисы, пересказ), "
            "     используй его СРАЗУ в параметре `summary_type`. Не уточняй это у пользователя.\n"
            "   - Если формат НЕ указан, оставь `summary_type` пустым (null).\n\n"

            "### STRICT CONSTRAINTS:\n"
            "- **NO HALLUCINATIONS**: Запрещено пересказывать текст своими словами без инструмента.\n"
            "- **COMMUNICATION**: Соблюдай деловой этикет, обращайся к пользователю по имени {user_name}.\n"
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
        # Извлекаем имя для приветствия (сначала firstName, потом общее имя)
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

        # 2. Если формат выбран заранее (внешняя кнопка вложения)
        effective_message = message
        if human_choice and not state.next:
            format_map = {"extractive": "факты", "thesis": "тезисы", "abstractive": "пересказ"}
            format_word = format_map.get(human_choice, human_choice)
            effective_message = f"{message}. Формат анализа: {format_word}."

        if state.next and message:
            await self._clear_stuck_tool_calls(config, state.values["messages"][-1])

        # Передача текущей даты и контекста
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

            # 1. Если цепочка завершена или нет вызовов инструментов
            if not state.next or not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
                for m in reversed(messages):
                    if isinstance(m, AIMessage) and m.content and not m.tool_calls:
                        if "предыдущий анализ отменен" not in m.content.lower():
                            return {"status": "success", "content": m.content}
                return {"status": "success", "content": "Запрос обработан."}

            # 2. Подготовка вызовов инструментов
            fixed_calls = []
            format_keywords = {"факты": "extractive", "тезисы": "thesis", "пересказ": "abstractive"}

            # Определяем, был ли в этом прогоне только что прочитан новый контент
            newly_read_content = None
            for m in reversed(messages):
                if isinstance(m, ToolMessage):
                    # Ищем результат doc_get_file_content или read_local_file_content
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

                # Подстановка ID документа, если он есть
                if doc_id and "document_id" in new_args:
                    new_args["document_id"] = doc_id

                # ПРИНУДИТЕЛЬНЫЙ ПУТЬ К ФАЙЛУ (Решает вашу проблему с подгрузкой старого файла)
                if file_path and tc["name"] in ["doc_get_file_content", "read_local_file_content"]:
                    new_args["file_path"] = file_path
                    if "attachment_id" in new_args: new_args["attachment_id"] = None

                # ОБРАБОТКА СУММАРИЗАТОРА
                if tc["name"] == "doc_summarize_text":
                    # Если есть свежепрочитанный контент — берем его приоритетно
                    if newly_read_content:
                        new_args["text"] = newly_read_content

                    # Если текста все еще нет (модель не передала), ищем в истории (но осторожно)
                    elif not new_args.get("text") or len(str(new_args.get("text"))) < 20:
                        new_args["text"] = newly_read_content  # Мы уже нашли его выше

                    # Поиск выбора формата пользователем
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

            # --- КРИТИЧЕСКИЙ БЛОК ПРОВЕРКИ (ВНЕ ЦИКЛА) ---
            summary_call = next((tc for tc in fixed_calls if tc["name"] == "doc_summarize_text"), None)

            if summary_call and not is_choice_active:
                user_msg_text = next(
                    (str(m.content).lower() for m in reversed(messages) if isinstance(m, HumanMessage)), "")
                user_already_chose = any(re.search(rf'\b{kw}\b', user_msg_text) for kw in format_keywords.keys())

                if not (user_already_chose or summary_call["args"].get("summary_type")):
                    # Очищаем ответ AI и запрашиваем кнопки
                    await self.agent.aupdate_state(config, {
                        "messages": [AIMessage(content="", tool_calls=fixed_calls, id=last_msg.id)]
                    }, as_node="agent")

                    return {
                        "status": "requires_action",
                        "action_type": "summarize_selection",
                        "message": "Пожалуйста, выберите формат анализа документа."
                    }

            # Продолжаем выполнение: обновляем состояние и вызываем инструменты
            await self.agent.aupdate_state(config, {
                "messages": [AIMessage(content=last_msg.content, tool_calls=fixed_calls, id=last_msg.id)]
            }, as_node="agent")

            # Передаем inputs=None, чтобы агент продолжил с текущего состояния (вызов инструментов)
            return await self._orchestrate(None, config, token, doc_id, file_path, is_choice_active=True,
                                           iteration=iteration + 1)

        except asyncio.TimeoutError:
            return {"status": "error", "message": "Превышено время ожидания."}
        except Exception as e:
            logger.error(f"Error in orchestrate: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

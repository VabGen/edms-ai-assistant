import logging
import asyncio
import re
from typing import Dict

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.tools import all_tools

logger = logging.getLogger(__name__)

UUID_PATTERN = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.I)


class EdmsDocumentAgent:
    def __init__(self):
        self.model = get_chat_model()
        self.tools = all_tools
        self.checkpointer = MemorySaver()

        self.tool_manifesto_template = (
            "### SYSTEM ROLE: EDMS EXPERT AGENT\n"
            "Ты — профессиональный аналитик СЭД.\n"
            "Текущий UUID документа (context_ui_id): {context_ui_id}\n\n"

            "### ЛОГИКА РАБОТЫ:\n"
            "1. Если вопрос касается реквизитов документа СЭД — используй `doc_get_details`.\n"
            "2. Если нужно проанализировать файл:\n"
            "   - Для файлов из СЭД используй `doc_get_file_content` (нужен attachment_id).\n"
            "   - Для ЛОКАЛЬНЫХ файлов (загруженных пользователем в чат) используй тот же `doc_get_file_content`, передавая путь в `file_path`.\n\n"

            "### ПРАВИЛА СУММАРИЗАЦИИ (КРИТИЧНО):\n"
            "1. Как только ты получил текст файла, ты ОБЯЗАН вызвать инструмент `doc_summarize_text`.\n"
            "2. **ВЫБОР ТИПА СУММАРИЗАЦИИ**:\n"
            "   - Если формат НЕ указан — используй `summary_type='abstractive'`. Это вызовет показ кнопок выбора пользователю.\n"
            "   - Если формат УКАЗАН (кнопкой или текстом) — используй его сразу ('extractive', 'abstractive', 'thesis').\n"
            "3. ЗАПРЕЩЕНО предлагать форматы текстом. Только вызов инструмента.\n\n"

            "### ПРАВИЛА ПОВЕДЕНИЯ:\n"
            "- Обращайся по имени: {user_name}.\n"
            "- Если в контексте есть LOCAL_FILE, приоритет отдавай его анализу.\n"
        )

        self.agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            checkpointer=self.checkpointer,
            interrupt_before=["tools"]
        )

    async def chat(self, message: str, user_token: str, context_ui_id: str = None,
                   thread_id: str = None, user_context: Dict = None,
                   file_path: str = None, human_choice: str = None) -> Dict:

        config = {"configurable": {"thread_id": thread_id or "default"}}
        user_name = f"{user_context.get('firstName', '')} {user_context.get('middleName', '')}".strip() or "пользователь"

        state = await self.agent.aget_state(config)

        # --- КЕЙС 1: Нажата кнопка ---
        if human_choice:
            if state.next:
                last_msg = state.values["messages"][-1]
                fixed_calls = []
                for tc in getattr(last_msg, "tool_calls", []):
                    args = tc["args"].copy()
                    args["token"] = user_token
                    if tc["name"] == "doc_summarize_text":
                        args["summary_type"] = human_choice
                    fixed_calls.append({**tc, "args": args})

                await self.agent.aupdate_state(
                    config,
                    {"messages": [AIMessage(content=last_msg.content, tool_calls=fixed_calls, id=last_msg.id)]},
                    as_node="agent"
                )
                return await self._orchestrate(None, config, user_token, context_ui_id, file_path, human_choice)
            else:
                # Быстрый старт с кнопки
                prompt = f"Проанализируй файл и выдели {human_choice}"
                return await self._start_new_cycle(prompt, config, user_token, context_ui_id, file_path, user_name)

        # --- КЕЙС 2: Обычное сообщение ---
        if state.next and message:
            # Сброс висящего прерывания, если пользователь сменил тему
            last_msg = state.values["messages"][-1]
            await self.agent.aupdate_state(
                config,
                {"messages": [AIMessage(content="(предыдущее действие отменено)", id=last_msg.id)]},
                as_node="agent"
            )

        return await self._start_new_cycle(message, config, user_token, context_ui_id, file_path, user_name)

    async def _start_new_cycle(self, message, config, token, doc_id, file_path, user_name):
        manifesto = self.tool_manifesto_template.format(context_ui_id=doc_id or "Не указан", user_name=user_name)
        env = f"### CONTEXT\nACTIVE_DOC_ID: {doc_id}\nLOCAL_FILE: {file_path}\n"
        inputs = {"messages": [SystemMessage(content=manifesto + env), HumanMessage(content=message)]}
        return await self._orchestrate(inputs, config, token, doc_id, file_path)

    async def _orchestrate(self, inputs, config, token, doc_id, file_path, human_choice=None):
        max_iterations = 10
        current_input = inputs

        for i in range(max_iterations):
            try:
                await asyncio.wait_for(self.agent.ainvoke(current_input, config=config), timeout=60.0)
                current_input = None
                state = await self.agent.aget_state(config)

                if not state.next: break

                last_msg = state.values["messages"][-1]
                if not isinstance(last_msg, AIMessage): break
                tool_calls = getattr(last_msg, "tool_calls", [])
                if not tool_calls: break

                # 1. Проверка на прерывание суммаризации
                summary_call = next((tc for tc in tool_calls if tc["name"] == "doc_summarize_text"), None)
                if summary_call and summary_call["args"].get("summary_type") == "abstractive" and not human_choice:
                    return {
                        "status": "requires_action",
                        "action_type": "summarize_selection",
                        "message": "Выберите формат сводки для документа:"
                    }

                # 2. Инъекция параметров в инструменты
                fixed_calls = []
                for tc in tool_calls:
                    new_args = dict(tc["args"])
                    new_args["token"] = token

                    # Если работаем с вложением СЭД
                    if "document_id" in new_args: new_args["document_id"] = doc_id

                    # Если пользователь загрузил свой файл в чат, прокидываем его путь
                    if tc["name"] == "doc_get_file_content" and file_path and not new_args.get("attachment_id"):
                        new_args["file_path"] = file_path

                    # Автоподбор attachment_id из истории (для СЭД)
                    if tc["name"] == "doc_get_file_content" and not new_args.get("attachment_id") and not new_args.get(
                            "file_path"):
                        for m in reversed(state.values["messages"]):
                            if isinstance(m, ToolMessage) and "ID:" in str(m.content):
                                uuids = UUID_PATTERN.findall(str(m.content))
                                if uuids:
                                    new_args["attachment_id"] = uuids[0]
                                    break

                    fixed_calls.append({**tc, "args": new_args})

                await self.agent.aupdate_state(
                    config,
                    {"messages": [AIMessage(content=last_msg.content, tool_calls=fixed_calls, id=last_msg.id)]},
                    as_node="agent"
                )

            except asyncio.TimeoutError:
                return {"status": "error", "message": "Модель долго думает, попробуйте упростить запрос."}

        final_state = await self.agent.aget_state(config)
        for m in reversed(final_state.values["messages"]):
            if isinstance(m, AIMessage) and m.content:
                return {"status": "success", "content": m.content}
        return {"status": "success", "content": "Готово."}
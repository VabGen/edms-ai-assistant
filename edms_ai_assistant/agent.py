# edms_ai_assistant/agent.py
import logging
import asyncio
import re
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from edms_ai_assistant.llm import get_chat_model
from edms_ai_assistant.tools import all_tools

logger = logging.getLogger(__name__)

UUID_PATTERN = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.I)


class EdmsDocumentAgent:
    def __init__(self):
        self.model = get_chat_model()
        self.tools = all_tools
        self.checkpointer = MemorySaver()

        self.tool_manifesto_template = (
            "### SYSTEM ROLE: EDMS EXPERT AGENT\n"
            "Ты — ведущий профессиональный аналитик СЭД (Систем Электронного Документооборота).\n"
            "Текущий UUID документа в интерфейсе (context_ui_id): {context_ui_id}\n\n"

            "### ЛОГИКА РАБОТЫ С ФАЙЛАМИ:\n"
            "1. **Если передано НАЗВАНИЕ файла** (например, из LOCAL_FILE или сообщения):\n"
            "   - СНАЧАЛА вызови `doc_get_details` для поиска этого файла в текущем документе.\n"
            "   - Найди объект, где имя совпадает, извлеки `attachment_id` и используй его в `doc_get_file_content`.\n"
            "2. **Разделение источников**:\n"
            "   - Файлы из СЭД -> используй `doc_get_file_content` (нужен attachment_id).\n"
            "   - Локальные файлы (пути temp/...) -> используй `read_local_file_content` или `doc_get_file_content` с file_path.\n\n"

            "### ПРАВИЛА СУММАРИЗАЦИИ (КРИТИЧНО):\n"
            "1. Как только ты получил текст файла, ты ОБЯЗАН вызвать инструмент `doc_summarize_text`.\n"
            "2. **ВЫБОР ТИПА СУММАРИЗАЦИИ**:\n"
            "   - Если формат НЕ указан пользователем — используй `summary_type='abstractive'`. Это активирует выбор кнопок на фронтенде.\n"
            "   - Если формат УКАЗАН (факты, тезисы, пересказ) — используй соответствующий тип ('extractive', 'thesis', 'abstractive') сразу.\n"
            "3. ЗАПРЕЩЕНО пересказывать текст самостоятельно. Используй только результат работы инструмента.\n\n"

            "### ПРАВИЛА ПОВЕДЕНИЯ:\n"
            "- Обращайся по имени: {user_name}.\n"
            "- Если в контексте LOCAL_FILE есть путь, приоритет отдавай его анализу.\n"
            "- Сохраняй деловой, но дружелюбный стиль.\n"
        )

        self.agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            checkpointer=self.checkpointer,
            interrupt_before=["tools"]
        )

    async def _clear_stuck_tool_calls(self, config, last_msg: AIMessage):
        """
        Метод для 'закрытия' зависших вызовов инструментов.
        Если AIMessage содержит tool_calls, а ответа от инструментов нет, LangGraph выдает ValueError.
        Мы принудительно добавляем сообщения об ошибке для каждого вызова.
        """
        if not last_msg.tool_calls:
            return

        logger.warning(
            f"Очистка зависших вызовов инструментов для thread_id: {config['configurable'].get('thread_id')}")
        error_tool_messages = []
        for tc in last_msg.tool_calls:
            error_tool_messages.append(ToolMessage(
                tool_call_id=tc["id"],
                content="Ошибка: Обработка этого инструмента была прервана из-за таймаута или ошибки контекста."
            ))

        await self.agent.aupdate_state(config, {"messages": error_tool_messages})

    async def chat(self, message: str, user_token: str, context_ui_id: str = None,
                   thread_id: str = None, user_context: Dict = None,
                   file_path: str = None, human_choice: str = None) -> Dict:

        config = {"configurable": {"thread_id": thread_id or "default"}}
        user_context = user_context or {}
        user_name = f"{user_context.get('firstName', '')}".strip() or "пользователь"

        state = await self.agent.aget_state(config)

        if state.next:
            last_msg = state.values["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                if not human_choice:
                    await self._clear_stuck_tool_calls(config, last_msg)
                    state = await self.agent.aget_state(config)

        if human_choice:
            if state.next:
                last_msg = state.values["messages"][-1]
                fixed_calls = []
                for tc in getattr(last_msg, "tool_calls", []):
                    args = dict(tc["args"])
                    if tc["name"] == "doc_summarize_text":
                        args["summary_type"] = human_choice
                    fixed_calls.append({**tc, "args": args})

                await self.agent.aupdate_state(
                    config,
                    {"messages": [AIMessage(content=last_msg.content, tool_calls=fixed_calls, id=last_msg.id)]},
                    as_node="agent"
                )
                return await self._orchestrate(None, config, user_token, context_ui_id, file_path,
                                               is_choice_active=True)
            else:
                message = f"Проанализируй файл и подготовь {human_choice}"

        elif state.next and message:
            last_msg = state.values["messages"][-1]
            await self.agent.aupdate_state(
                config,
                {"messages": [AIMessage(content="(предыдущий анализ отменен)", id=last_msg.id)]},
                as_node="agent"
            )
            await self._clear_stuck_tool_calls(config, last_msg)

        manifesto = self.tool_manifesto_template.format(
            context_ui_id=context_ui_id or "Не указан",
            user_name=user_name
        )
        env = f"\n### ENVIRONMENT\nACTIVE_DOC_ID: {context_ui_id}\nLOCAL_FILE: {file_path}\n"

        inputs = {"messages": [SystemMessage(content=manifesto + env), HumanMessage(content=message)]}
        return await self._orchestrate(inputs, config, user_token, context_ui_id, file_path)

    async def _orchestrate(self, inputs, config, token, doc_id, file_path, is_choice_active=False):
        current_input = inputs
        max_iterations = 10

        for _ in range(max_iterations):
            try:
                await asyncio.wait_for(self.agent.ainvoke(current_input, config=config), timeout=210.0)
                current_input = None

                state = await self.agent.aget_state(config)
                messages = state.values.get("messages", [])
                if not messages: break

                last_msg = messages[-1]
                if not state.next or not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
                    break

                summary_call = next((tc for tc in last_msg.tool_calls if tc["name"] == "doc_summarize_text"), None)
                if summary_call and not is_choice_active:
                    user_msg = ""
                    for m in reversed(messages):
                        if isinstance(m, HumanMessage):
                            user_msg = m.content.lower()
                            break

                    explicit_keywords = ["факты", "тезисы", "extractive", "thesis", "пересказ", "подробно"]
                    if not any(kw in user_msg for kw in explicit_keywords):
                        return {
                            "status": "requires_action",
                            "action_type": "summarize_selection",
                            "message": "В каком формате подготовить анализ документа?"
                        }

                fixed_calls = []
                for tc in last_msg.tool_calls:
                    new_args = dict(tc["args"])
                    new_args["token"] = token

                    if tc["name"] not in ["employee_search_tool", "read_local_file_content"]:
                        new_args["document_id"] = doc_id

                    if (tc["name"] in ["doc_get_file_content", "read_local_file_content"]) and file_path:
                        if not new_args.get("attachment_id"):
                            new_args["file_path"] = file_path

                    if tc["name"] == "doc_get_file_content" and not new_args.get("attachment_id") and not new_args.get(
                            "file_path"):
                        for m in reversed(messages):
                            if isinstance(m, ToolMessage) and UUID_PATTERN.search(str(m.content)):
                                found_uuids = UUID_PATTERN.findall(str(m.content))
                                if found_uuids:
                                    new_args["attachment_id"] = found_uuids[0]
                                    break

                    fixed_calls.append({**tc, "args": new_args})

                await self.agent.aupdate_state(
                    config,
                    {"messages": [AIMessage(content=last_msg.content, tool_calls=fixed_calls, id=last_msg.id)]},
                    as_node="agent"
                )
                is_choice_active = False

            except asyncio.TimeoutError:
                logger.error("Таймаут выполнения агента (ainvoke)")
                return {"status": "error",
                        "message": "Файл обрабатывается слишком долго. Попробуйте обновить чат или запросить краткие тезисы."}
            except Exception as e:
                logger.error(f"Ошибка оркестрации: {e}", exc_info=True)
                state = await self.agent.aget_state(config)
                if state.next:
                    await self._clear_stuck_tool_calls(config, state.values["messages"][-1])
                return {"status": "error", "message": f"Произошла ошибка: {str(e)}"}

        final_state = await self.agent.aget_state(config)
        for m in reversed(final_state.values.get("messages", [])):
            if isinstance(m, AIMessage) and m.content and not m.tool_calls:
                if "предыдущее действие" not in m.content.lower() and "анализ отменен" not in m.content.lower():
                    return {"status": "success", "content": m.content}

        return {"status": "success", "content": "Запрос обработан. Проверьте историю чата."}
# test_llm_invoke.py
import sys
import os
from enum import Enum
from typing import Literal

from pydantic import create_model, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edms_ai_assistant.llm import get_chat_model, logger

# print("--- Конфигурация ---")
# print(f"LLM Endpoint: {settings.LLM_ENDPOINT}")
# print(f"LLM Model: {settings.LLM_MODEL_NAME}")
# print(f"Temperature: {settings.LLM_TEMPERATURE}")

# print("\n--- Инициализация и вызов ChatModel ---")

class AgentType(Enum):
    DOCUMENT_ANALYZE = 0
    DOCUMENT_ATTACHMENT_ANALYZE= 1
    DEFAULT=2

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))

DynamicRouteDecision = create_model(
    "DynamicRouteDecision",
    next_agent=(
        # Literal[tuple(AgentType.values())] if AgentType.values() else Literal[AgentType.DEFAULT],
        Literal[AgentType.DEFAULT],
        Field(..., description="Имя агента, которому нужно передать задачу. Должно быть одним из"),
    ),
    reasoning=(str, Field(..., description="Почему выбран именно этот агент.")),
)

async def orchestrate_node():
    llm = get_chat_model()
    # print(f"ChatModel успешно инициализирован: {type(llm).__name__}")

    enhanced_message_content = "Привет! Проанализируй документ"

    system_prompt = f"""Ты - маршрутизатор AI-ассистента для СЭД (edms).
            Твоя задача - строго определить, какой из специализированных под-агентов должен обработать запрос пользователя.
            Доступные под-агенты: {AgentType.values()}.
            Проанализируй следующий запрос пользователя и контекст. Ответь строго в формате Pydantic-модели DynamicRouteDecision."""

    llm_input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": enhanced_message_content},
    ]

    # try:
    orchestrator_llm = llm.with_structured_output(AgentType)
    # orchestrate_node()

    decision: DynamicRouteDecision = await orchestrator_llm.ainvoke(
        llm_input_messages
    )
    logger.info(
        f"Оркестратор выбрал под-агента: {decision.next_agent}. Причина: {decision.reasoning}"
    )

    print("Выполняется вызов LLM...")
    response = llm.invoke(enhanced_message_content)

    # ответ
    print(f"\nПолучен ответ от LLM:")
    print(f"Тип ответа: {type(response)}")
    print(f"Содержимое: {response}")

if __name__ == "__main__":
    # Вызываем функцию напрямую
    orchestrate_node()
    # print("Результат:", result)




# try:
#
#     # MethodAssemblyType.DOCUMENT_ANALYZE
#
#     llm = get_chat_model()
#     # print(f"ChatModel успешно инициализирован: {type(llm).__name__}")
#
#     enhanced_message_content = "Привет! Проанализируй документ"
#
#     system_prompt = f"""Ты - маршрутизатор AI-ассистента для СЭД (edms).
#         Твоя задача - строго определить, какой из специализированных под-агентов должен обработать запрос пользователя.
#         Доступные под-агенты: {AgentType.values()}.
#         Проанализируй следующий запрос пользователя и контекст. Ответь строго в формате Pydantic-модели DynamicRouteDecision."""
#
#     llm_input_messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": enhanced_message_content},
#     ]
#
#     # try:
#     orchestrator_llm = llm.with_structured_output(AgentType)
#     orchestrate_node()
#     # decision: DynamicRouteDecision = await orchestrator_llm.ainvoke(
#     #     llm_input_messages
#     # )
#     # logger.info(
#     #     f"Оркестратор выбрал под-агента: {decision.next_agent}. Причина: {decision.reasoning}"
#     # )
#
#
#     # print(f"\nОтправка сообщения: '{test_message}'")
#
#     # Вызов LLM
#     print("Выполняется вызов LLM...")
#     response = llm.invoke(enhanced_message_content)
#
#     # ответ
#     print(f"\nПолучен ответ от LLM:")
#     print(f"Тип ответа: {type(response)}")
#     print(f"Содержимое: {response}")
#
#     # if hasattr(response, 'content'):
#     #     print(f"Текст ответа: {response.content}")
#     # elif hasattr(response, 'text'):
#     #     print(f"Текст ответа: {response.text}")
#     # else:
#     #     print(f"Текст ответа (str): {str(response)}")
#
# except Exception as e:
#     print(f"Ошибка при инициализации или вызове ChatModel: {e}")
#     import traceback
#     traceback.print_exc()
#
# print("\n--- Тест вызова LLM завершен ---")
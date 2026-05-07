# edms_ai_assistant/agent/planning/planner.py
"""
IntentPlanner — LLM-based планировщик вместо NLP keyword routing.

Принципиальное отличие от старого SemanticDispatcher:
- Не использует keywords/rules
- LLM сама анализирует запрос и решает что делать
- Знает о всех доступных tools через описание
- Производит типизированный ExecutionPlan

Используется в OrchestrationLoop ПЕРЕД bind_tools.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

from edms_ai_assistant.agent.context import ContextParams
from edms_ai_assistant.agent.planning.models import (
    DirectAnswerStep,
    ExecutionPlan,
    ParallelGroup,
    ToolCallStep,
)

logger = logging.getLogger(__name__)


def _extract_json(raw: str) -> str:
    """Strip thinking blocks, markdown fences, and locate the first JSON object.

    Handles output from CoT/reasoning models (Qwen3, DeepSeek-R1, gpt-oss)
    that emit ``<think>...</think>`` before the actual JSON payload.
    """
    # 1. Remove <think>...</think> blocks (including nested content)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # 2. Strip markdown code fences (```json ... ``` or ``` ... ```)
    raw = re.sub(r"^```(?:json)?\s*", "", raw).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()

    # 3. If there's still non-JSON prefix, find the first '{'
    if raw and not raw.startswith("{"):
        idx = raw.find("{")
        if idx != -1:
            raw = raw[idx:]

    return raw


def _build_tools_description(tools: list[Any]) -> str:
    """Generate tool descriptions from the actual tool registry.

    Uses tool.name and the first line of tool.description so the
    planner prompt stays in sync with the tool registry automatically.
    """
    if not tools:
        return "Инструменты не зарегистрированы."
    lines = ["Доступные инструменты EDMS:\n"]
    for tool in tools:
        name: str = getattr(tool, "name", str(tool))
        desc: str = getattr(tool, "description", "") or ""
        first_line = desc.split("\n")[0].strip()
        lines.append(f"- {name}: {first_line}")
    return "\n".join(lines)


_PLANNING_SYSTEM = """Ты — планировщик задач для AI-агента системы документооборота (EDMS).

Твоя задача: проанализировать запрос пользователя и составить ОПТИМАЛЬНЫЙ план выполнения.

КЛЮЧЕВОЕ ПРАВИЛО — когда tools НЕ нужны (can_answer_directly: true):
- Общие вопросы не связанные с EDMS ("что такое договор оферты", "привет", "объясни понятие")
- Юридический/экспертный анализ загруженного файла ("проверь на соответствие законодательству", "найди ошибки", "оцени договор")
- Математика, грамматика, перевод, форматирование текста
- Вопросы на основе уже полученных данных в истории диалога
- Продолжение темы из предыдущих сообщений

ВАЖНО: doc_compliance_check — это ТОЛЬКО проверка полей карточки EDMS документа.
Это НЕ юридический анализ и НЕ проверка содержимого файла на соответствие законам.
ЗАПРЕЩЕНО вызывать doc_compliance_check в ответ на:
- «Проверь на ошибки», «найди нарушения», «проверь на соответствие» после анализа файла.
- Любой follow-up вопрос если в истории уже есть результат doc_summarize_text.
В этих случаях: can_answer_directly: true — ответ из истории диалога.

Параллельное выполнение (parallel_capable: true):
- Используй когда несколько tools не зависят друг от друга
- Пример: doc_get_details + doc_get_file_content можно запустить одновременно

Верни ТОЛЬКО валидный JSON без markdown блоков."""

_PLANNING_USER_TEMPLATE = """
Запрос пользователя: {message}

Контекст:
- Активный документ в EDMS: {document_id}
- Загруженный файл: {file_path}
- История диалога содержит сообщений: {history_length}

{tools_description}

Верни JSON следующей структуры:
{{
  "can_answer_directly": bool,
  "parallel_capable": bool,
  "steps": [
    {{"tool_name": "...", "reason": "...", "args_hint": {{}}}}
    // или для параллельной группы:
    // {{"steps": [...], "reason": "..."}}
    // или для прямого ответа:
    // {{"reason": "..."}}
  ],
  "reasoning": "краткое объяснение",
  "estimated_complexity": "simple|medium|complex"
}}
"""


class IntentPlanner:
    """
    LLM-based планировщик, заменяющий NLP keyword routing.

    Производит ExecutionPlan из запроса пользователя.
    Модель сама решает нужны ли tools и какие.
    tools передаются через DI — описание генерируется автоматически
    и всегда соответствует реальному реестру инструментов.
    """

    def __init__(self, llm: BaseLanguageModel, tools: list[Any] | None = None) -> None:
        self._llm = llm
        self._tools_description = _build_tools_description(tools or [])
        logger.debug(
            "IntentPlanner initialized with %d tools",
            len(tools) if tools else 0,
        )

    async def plan(
        self,
        message: str,
        context: ContextParams,
        history_length: int = 0,
    ) -> ExecutionPlan:
        """
        Составить план выполнения для запроса.

        Args:
            message: Текст запроса пользователя.
            context: Контекст выполнения (document_id, file_path и тд).
            history_length: Количество сообщений в истории диалога.

        Returns:
            Типизированный ExecutionPlan.
        """
        safe_message = message.replace("{", "{{").replace("}", "}}")
        user_content = _PLANNING_USER_TEMPLATE.format(
            message=safe_message,
            document_id=context.document_id or "не выбран",
            file_path=context.file_path or "нет",
            history_length=history_length,
            tools_description=self._tools_description,
        )

        messages = [
            SystemMessage(content=_PLANNING_SYSTEM),
            HumanMessage(content=user_content),
        ]

        try:
            # response_format=json_object для надёжного JSON
            response = await self._llm.ainvoke(
                messages,
                response_format={"type": "json_object"},
            )
            raw = str(response.content).strip()
            raw = _extract_json(raw)
            data = json.loads(raw)
            plan = self._parse_plan(data)
            logger.info(
                "Plan created: direct=%s steps=%d complexity=%s reasoning=%s",
                plan.can_answer_directly,
                len(plan.steps),
                plan.estimated_complexity,
                plan.reasoning[:80],
            )
            return plan

        except Exception as exc:
            logger.warning(
                "Planning failed (%s) — falling back to full tools",
                exc,
            )
            # Fallback: безопасный план с полным набором tools
            return ExecutionPlan(
                can_answer_directly=False,
                parallel_capable=False,
                steps=[],
                reasoning="Planning failed — using full tool set as fallback",
                estimated_complexity="complex",
            )

    def _parse_plan(self, data: dict[str, Any]) -> ExecutionPlan:
        """Парсим JSON от LLM в типизированный ExecutionPlan."""
        steps = []
        for step_data in data.get("steps", []):
            if "tool_name" in step_data:
                steps.append(ToolCallStep(**step_data))
            elif "steps" in step_data:
                sub_steps = [ToolCallStep(**s) for s in step_data["steps"]]
                steps.append(ParallelGroup(
                    steps=sub_steps,
                    reason=step_data.get("reason", ""),
                ))
            else:
                steps.append(DirectAnswerStep(
                    reason=step_data.get("reason", "")
                ))

        return ExecutionPlan(
            can_answer_directly=data.get("can_answer_directly", False),
            parallel_capable=data.get("parallel_capable", False),
            steps=steps,
            reasoning=data.get("reasoning", ""),
            estimated_complexity=data.get("estimated_complexity", "simple"),
        )
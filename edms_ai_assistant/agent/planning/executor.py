# edms_ai_assistant/agent/planning/executor.py
"""
PlanExecutor — параллельное и последовательное выполнение ExecutionPlan.

Ключевые возможности:
- asyncio.gather для ParallelGroup шагов
- Инжекция token/document_id через ToolCallInjector
- Накопление результатов для передачи в Synthesizer
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from langchain_core.tools import BaseTool

from edms_ai_assistant.agent.context import ContextParams
from edms_ai_assistant.agent.planning.models import (
    DirectAnswerStep,
    ExecutionPlan,
    ParallelGroup,
    ToolCallStep,
)

logger = logging.getLogger(__name__)


class StepResult:
    """Результат выполнения одного шага плана."""

    __slots__ = ("tool_name", "result", "error", "duration_ms")

    def __init__(
        self,
        tool_name: str,
        result: Any = None,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        self.tool_name = tool_name
        self.result = result
        self.error = error
        self.duration_ms = duration_ms


class PlanExecutor:
    """
    Выполняет ExecutionPlan: последовательно или параллельно.

    Используется в OrchestrationLoop когда план содержит явные шаги.
    При can_answer_directly=True — executor не вызывается.
    """

    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools: dict[str, BaseTool] = {
            t.name: t for t in tools
        }

    async def execute(
        self,
        plan: ExecutionPlan,
        context: ContextParams,
    ) -> list[StepResult]:
        """
        Выполнить все шаги плана.

        Параллельные группы выполняются через asyncio.gather.
        Последовательные шаги — один за другим.
        """
        results: list[StepResult] = []

        for step in plan.steps:
            if isinstance(step, ParallelGroup):
                # Параллельное выполнение группы
                parallel_results = await asyncio.gather(
                    *[
                        self._execute_tool_step(s, context)
                        for s in step.steps
                    ],
                    return_exceptions=True,
                )
                for s, r in zip(step.steps, parallel_results):
                    if isinstance(r, Exception):
                        results.append(StepResult(
                            tool_name=s.tool_name,
                            error=str(r),
                        ))
                    else:
                        results.append(r)

            elif isinstance(step, ToolCallStep):
                result = await self._execute_tool_step(step, context)
                results.append(result)

            elif isinstance(step, DirectAnswerStep):
                # Прямой ответ — executor ничего не делает
                logger.debug("Direct answer step: %s", step.reason)

        return results

    async def _execute_tool_step(
        self,
        step: ToolCallStep,
        context: ContextParams,
    ) -> StepResult:
        """Выполнить один tool call с инжекцией контекста."""
        tool = self._tools.get(step.tool_name)
        if tool is None:
            logger.warning("Tool '%s' not found in registry", step.tool_name)
            return StepResult(
                tool_name=step.tool_name,
                error=f"Tool '{step.tool_name}' not found",
            )

        # Инжектируем обязательные поля
        args: dict[str, Any] = dict(step.args_hint)
        args["token"] = context.user_token
        if context.document_id and "document_id" not in args:
            args["document_id"] = context.document_id
        if context.file_path and "file_path" not in args:
            from edms_ai_assistant.agent.context import is_valid_uuid
            if not is_valid_uuid(context.file_path):
                args["file_path"] = context.file_path

        start = time.monotonic()
        try:
            result = await tool.ainvoke(args)
            duration = (time.monotonic() - start) * 1000
            logger.info(
                "Tool '%s' executed in %.0fms",
                step.tool_name,
                duration,
            )
            return StepResult(
                tool_name=step.tool_name,
                result=result,
                duration_ms=duration,
            )
        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            logger.error(
                "Tool '%s' failed after %.0fms: %s",
                step.tool_name,
                duration,
                exc,
                exc_info=True,
            )
            return StepResult(
                tool_name=step.tool_name,
                error=str(exc),
                duration_ms=duration,
            )
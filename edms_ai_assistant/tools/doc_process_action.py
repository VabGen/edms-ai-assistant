# edms_ai_assistant/tools/doc_process_action.py
"""
EDMS AI Assistant — Document Process Action Tool.
Инструменты для выполнения действий в процессе (Подписание, Согласование и т.д.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

from langchain_core.tools import InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field

from edms_ai_assistant.agent.runnable_utils import (
    get_document_id_from_config,
    get_token_from_config,
)
from edms_ai_assistant.domain.document import ProcessActionWithSign, SimpleProcessAction
from edms_ai_assistant.domain.enums import DocumentProcessType
from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from edms_ai_assistant.core.deps import AppDeps

logger = logging.getLogger(__name__)


class ProcessActionInput(BaseModel):
    action_type: DocumentProcessType = Field(
        ...,
        description="Тип действия: AGREEMENT (Согласовать), SIGNING (Подписать), STATEMENT (Утвердить).",
    )
    result: bool = Field(
        True,
        description="Результат действия (True - положительный, False - отрицательный/отклонить).",
    )
    comment: str | None = Field(None, description="Комментарий к действию.")


def create_doc_process_action_tool(deps: AppDeps) -> StructuredTool:
    """Инструмент для выполнения действий в бизнес-процессе документа."""

    async def doc_process_action(
        action_type: DocumentProcessType,
        result: bool = True,
        comment: str | None = None,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Выполняет действие (Согласование, Подписание, Утверждение) по текущему этапу документа."""
        try:
            document_id = get_document_id_from_config(config)
            token = get_token_from_config(config)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Ошибка авторизации или контекста: {e}",
            }

        try:
            # 1. Получаем процесс, чтобы найти текущий этап
            process = await deps.document_process_client.get_process(
                token, UUID(document_id)
            )
            if not process or not process.current_id:
                return {
                    "status": "error",
                    "message": "Документ не находится в активном процессе.",
                }

            current_item = process.current
            if not current_item or current_item.type != action_type:
                expected = (
                    action_type.value
                    if hasattr(action_type, "value")
                    else str(action_type)
                )
                actual = (
                    current_item.type.value
                    if current_item and hasattr(current_item.type, "value")
                    else str(current_item.type if current_item else "None")
                )
                return {
                    "status": "error",
                    "message": f"Некорректное действие. Текущий этап: {actual}, а запрошено: {expected}.",
                }

            # 2. Получаем ID текущего пользователя
            user = await deps.employee_client.get_current_user(token)

            # 3. Выполняем действие в зависимости от типа
            if action_type == DocumentProcessType.AGREEMENT:
                body = SimpleProcessAction(
                    result=result, employee_id=user.id, comment=comment
                )
                await deps.document_process_client.agreement(
                    token, UUID(document_id), process.current_id, body
                )
            elif action_type == DocumentProcessType.SIGNING:
                body = ProcessActionWithSign(
                    result=result, employee_id=user.id, comment=comment
                )
                await deps.document_process_client.signing(
                    token, UUID(document_id), process.current_id, body
                )
            elif action_type == DocumentProcessType.STATEMENT:
                body = ProcessActionWithSign(
                    result=result, employee_id=user.id, comment=comment
                )
                await deps.document_process_client.statement(
                    token, UUID(document_id), process.current_id, body
                )
            elif action_type == DocumentProcessType.REVIEW:
                body = SimpleProcessAction(
                    result=result, employee_id=user.id, comment=comment
                )
                await deps.document_process_client.review(
                    token, UUID(document_id), process.current_id, body
                )
            else:
                return {
                    "status": "error",
                    "message": f"Действие {action_type} пока не поддерживается через чат.",
                }

            return {
                "status": "success",
                "message": f"Действие «{action_type}» успешно выполнено (результат: {'ОК' if result else 'Отклонено'}).",
                "requires_reload": True,
            }

        except Exception as exc:
            logger.error(f"doc_process_action error: {exc}", exc_info=True)
            return {
                "status": "error",
                "message": f"Ошибка при выполнении действия: {exc}",
            }

    return StructuredTool.from_function(
        coroutine=doc_process_action,
        name="doc_process_action",
        description="Выполняет действие в процессе (Согласовать, Подписать, Утвердить, Рассмотреть).",
        args_schema=ProcessActionInput,
    )

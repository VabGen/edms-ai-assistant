# edms_ai_assistant/tools/__init__.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from edms_ai_assistant.tools.document import create_doc_get_details_tool
    from edms_ai_assistant.tools.doc_search import create_doc_search_tool
    from edms_ai_assistant.tools.attachment import create_attachment_fetch_tool
    from edms_ai_assistant.tools.create_document_from_file import create_document_from_file_tool
    from edms_ai_assistant.tools.document_comparison import create_doc_compare_documents_tool
    from edms_ai_assistant.tools.doc_compliance_check import create_doc_compliance_check_tool
    from edms_ai_assistant.tools.document_versions import create_doc_get_versions_tool
    from edms_ai_assistant.tools.introduction import create_introduction_tool
    from edms_ai_assistant.tools.doc_update_field import create_doc_update_field_tool
    from edms_ai_assistant.tools.doc_next_process import create_doc_next_process_tool
    from edms_ai_assistant.tools.doc_control import create_doc_control_tool
    from edms_ai_assistant.tools.local_file_tool import create_local_file_reader_tool
    from edms_ai_assistant.tools.summarization import create_doc_summarize_text_tool
    from edms_ai_assistant.tools.task import create_task_tool
    from edms_ai_assistant.tools.file_compare_tool import create_file_compare_tool
    from edms_ai_assistant.tools.access_grief_tool import create_access_grief_tool
    from edms_ai_assistant.tools.employee_search import create_employee_search_tool
    from edms_ai_assistant.tools.appeal_autofill import create_appeal_autofill_tool
    from edms_ai_assistant.tools.ask_user_select import ask_user_to_select
    from edms_ai_assistant.tools.report_tool import create_report_tools
    from edms_ai_assistant.tools.doc_process_action import create_doc_process_action_tool
    from edms_ai_assistant.tools.correspondent_tool import create_correspondent_tools
    from edms_ai_assistant.tools.acting_officer import create_acting_officer_tools
    from langchain_core.tools import StructuredTool
    from edms_ai_assistant.core.deps import AppDeps

logger = logging.getLogger(__name__)


def init_tools(deps: AppDeps, llm: Any) -> list[StructuredTool]:
    """Инициализирует все инструменты агента с внедрением зависимостей.

    Args:
        deps: Контейнер зависимостей приложения.
        llm: Модель чата для инструментов, требующих LLM.

    Returns:
        Список инструментов StructuredTool, готовых к использованию в LangGraph.
    """
    from edms_ai_assistant.tools.document import create_doc_get_details_tool
    from edms_ai_assistant.tools.doc_search import create_doc_search_tool
    from edms_ai_assistant.tools.attachment import create_attachment_fetch_tool
    from edms_ai_assistant.tools.create_document_from_file import create_document_from_file_tool
    from edms_ai_assistant.tools.document_comparison import create_doc_compare_documents_tool
    from edms_ai_assistant.tools.doc_compliance_check import create_doc_compliance_check_tool
    from edms_ai_assistant.tools.document_versions import create_doc_get_versions_tool
    from edms_ai_assistant.tools.introduction import create_introduction_tool
    from edms_ai_assistant.tools.doc_update_field import create_doc_update_field_tool
    from edms_ai_assistant.tools.doc_next_process import create_doc_next_process_tool
    from edms_ai_assistant.tools.doc_control import create_doc_control_tool
    from edms_ai_assistant.tools.local_file_tool import create_local_file_reader_tool
    from edms_ai_assistant.tools.summarization import create_doc_summarize_text_tool
    from edms_ai_assistant.tools.task import create_task_tool
    from edms_ai_assistant.tools.file_compare_tool import create_file_compare_tool
    from edms_ai_assistant.tools.access_grief_tool import create_access_grief_tool
    from edms_ai_assistant.tools.employee_search import create_employee_search_tool
    from edms_ai_assistant.tools.appeal_autofill import create_appeal_autofill_tool
    from edms_ai_assistant.tools.ask_user_select import ask_user_to_select
    from edms_ai_assistant.tools.report_tool import create_report_tools
    from edms_ai_assistant.tools.doc_process_action import create_doc_process_action_tool
    from edms_ai_assistant.tools.correspondent_tool import create_correspondent_tools
    from edms_ai_assistant.tools.acting_officer import create_acting_officer_tools

    tools = [
        create_doc_get_details_tool(deps.document_service),
        create_doc_search_tool(deps.document_client),
        create_attachment_fetch_tool(deps),
        create_document_from_file_tool(deps),
        create_doc_compare_documents_tool(deps.document_client, llm),
        create_doc_compliance_check_tool(deps),
        create_doc_get_versions_tool(deps.document_client),
        create_introduction_tool(deps),
        create_doc_update_field_tool(deps.document_client),
        create_doc_next_process_tool(deps.base_client, deps.employee_client),
        create_doc_control_tool(deps.control_client, deps.employee_client),
        create_local_file_reader_tool(deps.file_processor_service),
        create_doc_summarize_text_tool(deps.summarization_service, llm),
        create_task_tool(deps),
        create_file_compare_tool(deps.document_client, deps.attachment_client),
        create_access_grief_tool(deps.access_grief_client, deps.employee_client),
        create_employee_search_tool(deps),
        create_appeal_autofill_tool(deps),
        ask_user_to_select,
        *create_report_tools(deps),
        create_doc_process_action_tool(deps),
        *create_correspondent_tools(deps),
        *create_acting_officer_tools(deps),
    ]

    logger.info("Initialized %d tools with DI factories.", len(tools))
    return tools


# Legacy export for backward compatibility
all_tools: list[StructuredTool] = []

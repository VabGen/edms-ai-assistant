# edms_ai_assistant/tools/__init__.py
"""
EDMS AI Assistant — Tool Registry.

Единая точка регистрации всех LangChain-совместимых инструментов агента.

Категории:
  Documents     — метаданные, версии, поиск, сравнение
  Content       — вложения, локальные файлы, сравнение файлов
  Analysis      — суммаризация, резолюции
  Workflow      — ознакомление, поручения, обращения
  People        — поиск сотрудников
  Notifications — уведомления и напоминания
"""

from .appeal_autofill import autofill_appeal_document

# Content
from .attachment import doc_get_file_content

# Notifications
from .doc_notification import doc_send_notification
from .doc_search import doc_search_tool

# Documents
from .document import doc_get_details
from .document_comparison import doc_compare
from .document_versions import doc_get_versions

# People
from .employee_search import employee_search_tool
from .file_compare_tool import doc_compare_with_local

# Workflow
from .introduction import introduction_create_tool
from .local_file_tool import read_local_file_content

# Analysis
from .summarization import doc_summarize_text
from .task import task_create_tool

# ── Полный реестр инструментов агента ────────────────────────────────────────
all_tools = [
    # Documents
    doc_get_details,
    doc_get_versions,
    doc_compare,
    doc_search_tool,
    # Content
    doc_get_file_content,
    read_local_file_content,
    doc_compare_with_local,
    # Analysis
    doc_summarize_text,
    # Workflow
    introduction_create_tool,
    task_create_tool,
    autofill_appeal_document,
    # People
    employee_search_tool,
    # Notifications
    doc_send_notification,
]

__all__ = [
    "all_tools",
    # Documents
    "doc_get_details",
    "doc_get_versions",
    "doc_compare",
    "doc_search_tool",
    # Content
    "doc_get_file_content",
    "read_local_file_content",
    "doc_compare_with_local",
    # Analysis
    "doc_summarize_text",
    # Workflow
    "introduction_create_tool",
    "task_create_tool",
    "autofill_appeal_document",
    # People
    "employee_search_tool",
    # Notifications
    "doc_send_notification",
]

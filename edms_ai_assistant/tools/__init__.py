# edms_ai_assistant/tools/__init__.py
from .document import doc_get_details
from .attachment import doc_get_file_content
from .summarization import doc_summarize_text
from .local_file_tool import read_local_file_content
from .employee import employee_search_tool
from .appeal_autofill import autofill_appeal_document
from .introduction import introduction_create_tool
from .task import task_create_tool
from .document_versions import doc_get_versions
from .document_comparison import doc_compare

all_tools = [
    doc_get_details,
    doc_get_file_content,
    doc_summarize_text,
    read_local_file_content,
    employee_search_tool,
    autofill_appeal_document,
    introduction_create_tool,
    task_create_tool,
    doc_get_versions,
    doc_compare,
]

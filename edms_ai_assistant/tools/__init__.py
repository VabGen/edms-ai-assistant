# edms_ai_assistant/tools/__init__.py
from .document import doc_get_details
from .attachment import doc_get_file_content
from .summarization import doc_summarize_text
from .local_file_tool import read_local_file_content
from .employee import employee_search_tool

all_tools = [
    doc_get_details,
    doc_get_file_content,
    doc_summarize_text,
    read_local_file_content,
    employee_search_tool,
]

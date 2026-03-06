# edms_ai_assistant/model.py
"""
Pydantic models for EDMS AI Assistant API.
"""
from typing import List, Optional, Annotated, Literal
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ── User Context & Input ─────────────────────────────────────────────────────
class UserContext(BaseModel):
    """User context from EDMS Employee API."""

    firstName: Optional[str] = None
    lastName: Optional[str] = None
    middleName: Optional[str] = None
    departmentName: Optional[str] = None
    postName: Optional[str] = None
    role: Optional[str] = None


class UserInput(BaseModel):
    """
    Validated user input for chat/summarize endpoints.

    Fields:
        message: User message text (1-8000 chars)
        user_token: JWT bearer token (min 20 chars)
        context_ui_id: Document UUID (optional)
        context: User context from Employee API (optional)
        file_path: Local file path or attachment UUID (optional)
        human_choice: Summary type: "факты"|"пересказ"|"тезисы"|"1"|"2"|"3"
        thread_id: LangGraph session ID for conversation history (optional)
    """

    message: str = Field(..., min_length=1, max_length=8000)
    user_token: str = Field(..., min_length=20)
    context_ui_id: Optional[str] = Field(None, pattern=r"^[0-9a-f-]{36}$|^$")
    context: Optional[UserContext] = None
    file_path: Optional[str] = Field(None, max_length=500)
    human_choice: Optional[str] = Field(None, max_length=100)
    thread_id: Optional[str] = Field(None, max_length=255)

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        return v.strip()

    @field_validator("human_choice")
    @classmethod
    def normalize_human_choice(cls, v: Optional[str]) -> Optional[str]:
        """Normalize human_choice to canonical summary type."""
        if not v:
            return None
        v_lower = v.strip().lower()
        mapping = {
            "1": "extractive",
            "факты": "extractive",
            "ключевые факты": "extractive",
            "2": "abstractive",
            "пересказ": "abstractive",
            "краткий пересказ": "abstractive",
            "3": "thesis",
            "тезисы": "thesis",
            "тезисный план": "thesis",
        }
        return mapping.get(v_lower, v_lower)


# ── API Responses ────────────────────────────────────────────────────────────
class AssistantResponse(BaseModel):
    """
    Standardized API response for chat/summarize endpoints.
    """

    status: Literal["success", "error", "requires_action", "requires_choice"]
    response: Optional[str] = None
    action_type: Optional[str] = None
    message: Optional[str] = None
    thread_id: Optional[str] = None

    elapsed_ms: Optional[float] = None
    iterations: Optional[int] = None


class FileUploadResponse(BaseModel):
    """Response for file upload endpoint."""

    file_path: str
    file_name: str
    size_bytes: Optional[int] = None


class NewChatRequest(BaseModel):
    """Request to create a new conversation thread."""

    user_token: str = Field(..., min_length=20)


# ── LangGraph State ─────────────────────────────────────────────────────────
class AgentState(TypedDict):
    """
    LangGraph state with message reducer and iteration counter.

    Fields:
        messages: Accumulated message list (add_messages reducer)
        graph_iterations: Counter to prevent infinite loops
        total_tokens: Optional token usage tracking
        start_time: Optional timestamp for performance metrics
    """

    messages: Annotated[List[BaseMessage], add_messages]
    graph_iterations: int
    total_tokens: Optional[int]
    start_time: Optional[float]

from typing import List, Optional, Annotated
from pydantic import BaseModel
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class UserContext(BaseModel):
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    role: Optional[str] = None


class UserInput(BaseModel):
    message: str
    user_token: str
    context_ui_id: Optional[str] = None
    context: Optional[UserContext] = None
    file_path: Optional[str] = None
    human_choice: Optional[str] = None
    thread_id: Optional[str] = None


class AssistantResponse(BaseModel):
    status: str
    response: Optional[str] = None
    action_type: Optional[str] = None
    message: Optional[str] = None
    thread_id: Optional[str] = None


class FileUploadResponse(BaseModel):
    file_path: str
    file_name: str


class NewChatRequest(BaseModel):
    user_token: str


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

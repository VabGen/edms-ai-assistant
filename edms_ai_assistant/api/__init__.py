from edms_ai_assistant.api.routes.actions import router as actions_router
from edms_ai_assistant.api.routes.chat import router as chat_router
from edms_ai_assistant.api.routes.files import router as files_router
from edms_ai_assistant.api.routes.settings import router as settings_router
from edms_ai_assistant.api.routes.system import router as system_router

__all__ = [
    "actions_router",
    "chat_router",
    "files_router",
    "settings_router",
    "system_router",
]

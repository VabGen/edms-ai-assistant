# edms_ai_assistant/__init__.py

from . import config
from .llm import get_chat_model, get_embedding_model

llm = get_chat_model
embedding_model = get_embedding_model

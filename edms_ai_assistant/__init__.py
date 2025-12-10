# edms_ai_assistant/__init__.py

from .llm import get_chat_model, get_embedding_model_to_use

llm = get_chat_model
embedding_model = get_embedding_model_to_use

from . import config

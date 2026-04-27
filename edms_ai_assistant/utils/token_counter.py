import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class TokenCounter(ABC):
    @abstractmethod
    def count(self, text: str) -> int: ...

class CharRatioTokenCounter(TokenCounter):
    """Fallback для русского языка. Llama/Qwen в среднем жрут 1 токен на 3.5-4 символа."""
    RATIO = 3.5

    def count(self, text: str) -> int:
        return int(len(text) / self.RATIO)

class HuggingFaceTokenCounter(TokenCounter):
    """Идеальный вариант. Требует pip install transformers"""
    def __init__(self, model_name: str = "Qwen/Qwen2.5-72B-Instruct"):
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            logger.info("HuggingFace tokenizer loaded for %s", model_name)
        except Exception as e:
            logger.error("Failed to load HF tokenizer, falling back to CharRatio: %s", e)
            self._tokenizer = None

    def count(self, text: str) -> int:
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return CharRatioTokenCounter().count(text)

token_counter = CharRatioTokenCounter()
"""
Request-scoped logging context.

Хранит `request_id` в contextvar и предоставляет logging.Filter,
который автоматически добавляет его как extra-поле в каждую запись
лога внутри обрабатываемого запроса.

Использование:
    from edms_ai_assistant.summarizer.observability.logging_ctx import (
        request_id_var, install_request_id_filter,
    )

    install_request_id_filter()      # один раз при старте приложения
    request_id_var.set("req-123")    # в начале запроса
    logger.info("...")               # автоматически получит request_id="req-123"

Чтобы это поле увидеть в выводе — добавьте `%(request_id)s` в LOGGING_FORMAT.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Делает request_id доступным как `record.request_id` для всех логгеров."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def install_request_id_filter(logger: logging.Logger | None = None) -> None:
    """Устанавливает RequestIdFilter на root-логгер (или указанный)."""
    target = logger or logging.getLogger()
    if not any(isinstance(f, RequestIdFilter) for f in target.filters):
        target.addFilter(RequestIdFilter())


__all__ = ["RequestIdFilter", "install_request_id_filter", "request_id_var"]

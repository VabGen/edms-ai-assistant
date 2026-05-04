"""Async retry decorator with exponential back-off.

Extracted from the legacy ``retry_utils.py`` into the transport layer so it
carries no external dependencies and is straightforward to tests in isolation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, TypeVar

import httpx

logger = logging.getLogger(__name__)

# Status codes that must NOT be retried — they represent deterministic errors.
_NO_RETRY_STATUSES: frozenset[int] = frozenset({400, 401, 403, 404, 405, 409, 410, 422})

_F = TypeVar("_F", bound=Callable[..., Coroutine[Any, Any, Any]])


def _should_retry(exc: BaseException) -> bool:
    """Return True only for transient / server-side failures worth retrying."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code not in _NO_RETRY_STATUSES
    if isinstance(exc, httpx.RequestError):
        return True
    return False


def async_retry(
    *,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (
        httpx.RequestError,
        httpx.HTTPStatusError,
    ),
) -> Callable[[_F], _F]:
    """Decorator: retry an async function with exponential back-off.

    Args:
        max_attempts: Total number of attempts (including the first call).
        delay:        Initial sleep between retries in seconds.
        backoff:      Multiplier applied to ``delay`` after each failure.
        exceptions:   Tuple of exception types that trigger a retry.

    Non-retriable HTTP errors (4xx except 408/429) are re-raised immediately
    without sleeping to avoid wasting time on deterministic failures.
    """

    def decorator(fn: _F) -> _F:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:
                    is_last = attempt == max_attempts
                    if not _should_retry(exc):
                        # Business error — re-raise immediately, no sleep.
                        logger.debug(
                            "Non-retriable error in %s (attempt %d/%d): %s",
                            fn.__qualname__,
                            attempt,
                            max_attempts,
                            exc,
                        )
                        raise
                    if is_last:
                        logger.error(
                            "All %d attempts exhausted for %s: %s",
                            max_attempts,
                            fn.__qualname__,
                            exc,
                        )
                        raise
                    logger.warning(
                        "Attempt %d/%d failed for %s — retrying in %.2fs. Error: %s",
                        attempt,
                        max_attempts,
                        fn.__qualname__,
                        current_delay,
                        exc,
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            # Should never be reached.
            return None  # pragma: no cover

        return wrapper  # type: ignore[return-value]

    return decorator

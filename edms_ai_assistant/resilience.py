# edms_ai_assistant/resilience.py
"""
Resilience — retry with exponential backoff and circuit breaker.

Best Practice (2026): LLM calls are inherently unreliable — network
failures, rate limits, and transient API errors are routine.  A structured
retry strategy with jitter prevents thundering-herd effects and gracefully
degrades under load.

Inspired by Anthropic's internal retry stack and Google's SRE handbook
("Handling Overload Gracefully").
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ─── Retry classification ──────────────────────────────────────────────────────


class ErrorCategory(str, Enum):
    """Classification of errors for retry decision-making."""

    TRANSIENT = "transient"  # Network blip, 429, 503 — retry with backoff
    FATAL = "fatal"  # Auth error, invalid request — do not retry
    UNKNOWN = "unknown"  # Unclassified — retry once, then give up


# Error signatures that indicate transient vs fatal failures
_TRANSIENT_SIGNALS: tuple[str, ...] = (
    "rate_limit",
    "429",
    "503",
    "502",
    "timeout",
    "connection",
    "timed out",
    "server error",
    "overloaded",
    "capacity",
)

_FATAL_SIGNALS: tuple[str, ...] = (
    "401",
    "403",
    "invalid_api_key",
    "authentication",
    "invalid_request",
    "context_length_exceeded",
    "max_tokens",
    "model_not_found",
)


def classify_error(error: Exception) -> ErrorCategory:
    """
    Classify an exception into a retry category.

    Args:
        error: The exception to classify.

    Returns:
        ErrorCategory indicating whether to retry.
    """
    err_str = str(error).lower()
    if any(sig in err_str for sig in _FATAL_SIGNALS):
        return ErrorCategory.FATAL
    if any(sig in err_str for sig in _TRANSIENT_SIGNALS):
        return ErrorCategory.TRANSIENT
    return ErrorCategory.UNKNOWN


# ─── Retry with exponential backoff + jitter ────────────────────────────────────


@dataclass
class RetryConfig:
    """
    Configuration for the retry strategy.

    Attributes:
        max_attempts: Maximum number of attempts (including the initial call).
        base_delay: Base delay in seconds for exponential backoff.
        max_delay: Maximum delay cap in seconds.
        jitter_factor: Random jitter multiplier (0.0–1.0) to prevent
            thundering herd.  0.0 = no jitter, 1.0 = full jitter.
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter_factor: float = 0.5


def _compute_delay(attempt: int, config: RetryConfig) -> float:
    """
    Compute the delay for the given attempt using exponential backoff + jitter.

    Formula::

        delay = min(base_delay * 2^attempt, max_delay)
        jitter = delay * jitter_factor * random()
        total = delay + jitter

    Args:
        attempt: Zero-based attempt index.
        config: Retry configuration.

    Returns:
        Delay in seconds.
    """
    exponential = config.base_delay * (2 ** attempt)
    capped = min(exponential, config.max_delay)
    jitter = capped * config.jitter_factor * random.random()
    return capped + jitter


async def retry_with_backoff(
    fn: Callable[..., Any],
    *,
    config: RetryConfig | None = None,
    retryable: Callable[[Exception], bool] | None = None,
    operation_name: str = "operation",
) -> Any:
    """
    Execute an async callable with exponential backoff retry.

    Args:
        fn: Async callable to execute.
        config: Retry configuration (uses defaults if None).
        retryable: Optional predicate that returns True if the error is
            retryable.  Defaults to ``classify_error != FATAL``.
        operation_name: Human-readable name for logging.

    Returns:
        The result of ``fn()``.

    Raises:
        Exception: The last exception if all attempts are exhausted,
            or immediately if the error is classified as FATAL.
    """
    if config is None:
        config = RetryConfig()

    if retryable is None:
        retryable = lambda exc: classify_error(exc) != ErrorCategory.FATAL

    last_exception: Exception | None = None

    for attempt in range(config.max_attempts):
        try:
            return await fn()
        except Exception as exc:
            last_exception = exc

            if not retryable(exc):
                logger.error(
                    "Retry [%s]: fatal error on attempt %d — not retrying: %s",
                    operation_name,
                    attempt + 1,
                    exc,
                )
                raise

            if attempt < config.max_attempts - 1:
                delay = _compute_delay(attempt, config)
                category = classify_error(exc)
                logger.warning(
                    "Retry [%s]: %s error on attempt %d/%d — retrying in %.1fs: %s",
                    operation_name,
                    category.value,
                    attempt + 1,
                    config.max_attempts,
                    delay,
                    str(exc)[:200],
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "Retry [%s]: all %d attempts exhausted: %s",
                    operation_name,
                    config.max_attempts,
                    str(exc)[:200],
                )

    # Should not reach here, but just in case
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("retry_with_backoff: unexpected state")


# ─── Simple circuit breaker ────────────────────────────────────────────────────


@dataclass
class CircuitBreaker:
    """
    Simple circuit breaker that prevents repeated calls to a failing service.

    States:
    - CLOSED: Normal operation — calls pass through.
    - OPEN: Failure threshold exceeded — calls are rejected immediately.
    - HALF_OPEN: After recovery timeout — one probe call is allowed.

    Attributes:
        failure_threshold: Number of consecutive failures before opening.
        recovery_timeout: Seconds before transitioning from OPEN to HALF_OPEN.
        half_open_max: Number of successful calls in HALF_OPEN to close the circuit.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max: int = 1

    _failure_count: int = field(default=0, init=False)
    _state: str = field(default="closed", init=False)  # closed | open | half_open
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_successes: int = field(default=0, init=False)

    @property
    def state(self) -> str:
        """Current circuit breaker state."""
        if self._state == "open":
            if time.monotonic() - self._last_failure_time > self.recovery_timeout:
                self._state = "half_open"
                self._half_open_successes = 0
        return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        current = self.state
        if current == "closed":
            return True
        if current == "half_open":
            return True  # Allow probe calls
        return False  # open

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == "half_open":
            self._half_open_successes += 1
            if self._half_open_successes >= self.half_open_max:
                self._state = "closed"
                self._failure_count = 0
                logger.info("Circuit breaker: CLOSED (recovered)")
        else:
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == "half_open":
            self._state = "open"
            logger.warning("Circuit breaker: OPEN (probe failed)")
        elif self._failure_count >= self.failure_threshold:
            self._state = "open"
            logger.warning(
                "Circuit breaker: OPEN (%d consecutive failures)",
                self._failure_count,
            )
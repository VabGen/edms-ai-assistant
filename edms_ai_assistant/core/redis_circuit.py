"""Redis circuit breaker implementation for resilient caching.

Provides circuit breaker pattern for Redis operations to prevent cascading failures
and provide graceful degradation when Redis is unavailable.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, TypeVar, Callable, Awaitable, Any

from edms_ai_assistant.config import settings

if TYPE_CHECKING:
    import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()  # Normal operation
    OPEN = auto()    # Circuit is open, calls fail fast
    HALF_OPEN = auto()  # Testing if recovery is possible


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds before trying recovery
    half_open_max_calls: int = 3  # Max calls in half-open state
    
    # Redis-specific settings
    operation_timeout: float = 5.0  # Timeout for individual Redis operations


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Calls rejected when circuit is open
    
    # Per-operation metrics
    operation_metrics: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"hits": 0, "misses": 0, "errors": 0})
    )


class RedisCircuitBreaker:
    """Circuit breaker for Redis operations with metrics and fallback."""
    
    def __init__(
        self,
        redis_client: aioredis.Redis,
        config: CircuitBreakerConfig | None = None,
    ):
        self._redis = redis_client
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()
        
    async def execute_operation(
        self,
        operation_name: str,
        operation: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T | None:
        """Execute Redis operation with circuit breaker protection.
        
        Args:
            operation_name: Name of the operation for metrics
            operation: Async function to execute
            *args, **kwargs: Arguments to pass to operation
            
        Returns:
            Result of operation, or None if circuit is open or operation fails
        """
        self._metrics.total_calls += 1
        
        async with self._lock:
            # Check circuit state
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    self._metrics.rejected_calls += 1
                    logger.debug(
                        "Circuit breaker OPEN - rejecting Redis call: %s",
                        operation_name
                    )
                    return None
        
        # Execute operation with timeout
        try:
            result = await asyncio.wait_for(
                operation(*args, **kwargs),
                timeout=self._config.operation_timeout,
            )
            
            await self._on_success(operation_name)
            return result
            
        except asyncio.TimeoutError:
            await self._on_failure(operation_name, "timeout")
            return None
        except Exception as exc:
            await self._on_failure(operation_name, str(exc))
            return None
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt circuit recovery."""
        return (
            time.time() - self._last_failure_time
            >= self._config.recovery_timeout
        )
    
    async def _on_success(self, operation_name: str) -> None:
        """Handle successful operation."""
        async with self._lock:
            self._metrics.successful_calls += 1
            self._metrics.operation_metrics[operation_name]["hits"] += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self._config.half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info("Circuit breaker CLOSED - recovery successful")
    
    async def _on_failure(self, operation_name: str, error: str) -> None:
        """Handle failed operation."""
        async with self._lock:
            self._metrics.failed_calls += 1
            self._metrics.operation_metrics[operation_name]["errors"] += 1
            
            if self._state == CircuitState.HALF_OPEN:
                # Reset to OPEN if we fail during recovery testing
                self._state = CircuitState.OPEN
                self._last_failure_time = time.time()
                logger.warning(
                    "Circuit breaker OPEN - recovery test failed for %s: %s",
                    operation_name,
                    error,
                )
            else:
                self._failure_count += 1
                self._last_failure_time = time.time()
                
                if self._failure_count >= self._config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.error(
                        "Circuit breaker OPEN - %d failures threshold reached. "
                        "Last error: %s",
                        self._failure_count,
                        error,
                    )
    
    def get_metrics(self) -> dict[str, Any]:
        """Get current circuit breaker metrics."""
        return {
            "state": self._state.name,
            "failure_count": self._failure_count,
            "last_failure_time": self._last_failure_time,
            "total_calls": self._metrics.total_calls,
            "successful_calls": self._metrics.successful_calls,
            "failed_calls": self._metrics.failed_calls,
            "rejected_calls": self._metrics.rejected_calls,
            "operation_metrics": dict(self._metrics.operation_metrics),
        }
    
    def get_health_status(self) -> dict[str, Any]:
        """Get health status for monitoring."""
        success_rate = (
            self._metrics.successful_calls / self._metrics.total_calls
            if self._metrics.total_calls > 0
            else 1.0
        )
        
        return {
            "status": "healthy" if self._state == CircuitState.CLOSED else "degraded",
            "state": self._state.name,
            "success_rate": round(success_rate, 4),
            "is_available": self._state != CircuitState.OPEN,
        }


class CircuitBreakerRedisCache:
    """Redis cache wrapper with circuit breaker protection."""
    
    def __init__(self, circuit_breaker: RedisCircuitBreaker):
        self._cb = circuit_breaker
    
    async def get(self, key: str) -> str | None:
        """Get value from cache with circuit breaker protection."""
        result = await self._cb.execute_operation(
            "get",
            self._cb._redis.get,
            key,
        )
        if result is None:
            self._cb._metrics.operation_metrics["get"]["misses"] += 1
        return result
    
    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        """Set value in cache with circuit breaker protection."""
        try:
            result = await self._cb.execute_operation(
                "set",
                self._cb._redis.set,
                key,
                value,
                ex=ex,
            )
            return result is not None
        except Exception:
            return False
    
    async def delete(self, *keys: str) -> int:
        """Delete keys from cache with circuit breaker protection."""
        try:
            result = await self._cb.execute_operation(
                "delete",
                self._cb._redis.delete,
                *keys,
            )
            return result if result is not None else 0
        except Exception:
            return 0
    
    async def get_circuit_breaker_metrics(self) -> dict[str, Any]:
        """Get circuit breaker metrics for monitoring."""
        return self._cb.get_metrics()
    
    async def get_health_status(self) -> dict[str, Any]:
        """Get health status for monitoring."""
        return self._cb.get_health_status()
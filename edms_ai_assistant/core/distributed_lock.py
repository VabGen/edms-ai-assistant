"""Distributed locking implementation using Redis for thread-safe operations.

Provides distributed locks for preventing race conditions in concurrent
operations on shared resources like LangGraph thread states.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator

from edms_ai_assistant.config import settings

if TYPE_CHECKING:
    import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


@dataclass
class DistributedLockConfig:
    """Configuration for distributed lock behavior."""
    lock_ttl: int = 30  # Lock TTL in seconds
    acquire_timeout: float = 5.0  # Max time to wait for lock acquisition
    retry_interval: float = 0.1  # Interval between lock acquisition retries


class DistributedLock:
    """Distributed lock implementation using Redis SET with NX option.
    
    Uses Redis for distributed locking to prevent race conditions across
    multiple application instances when modifying shared state like
    LangGraph thread states.
    """
    
    def __init__(
        self,
        redis_client: aioredis.Redis,
        config: DistributedLockConfig | None = None,
    ):
        self._redis = redis_client
        self._config = config or DistributedLockConfig()
        
    async def acquire(
        self,
        lock_key: str,
        lock_value: str | None = None,
    ) -> bool:
        """Acquire a distributed lock.
        
        Args:
            lock_key: Unique key for the lock (e.g., "thread_state:{thread_id}")
            lock_value: Unique value for this lock attempt (defaults to UUID)
            
        Returns:
            True if lock was acquired, False otherwise
        """
        if lock_value is None:
            lock_value = str(uuid.uuid4())
        
        lock_key_with_prefix = f"lock:{lock_key}"
        
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < self._config.acquire_timeout:
            try:
                # Redis SET with NX (only if not exists) and EX (expiration)
                result = await self._redis.set(
                    lock_key_with_prefix,
                    lock_value,
                    nx=True,
                    ex=self._config.lock_ttl,
                )
                
                if result:
                    logger.debug("Acquired distributed lock: %s", lock_key_with_prefix)
                    return True
                    
            except Exception as exc:
                logger.error("Failed to acquire lock %s: %s", lock_key_with_prefix, exc)
                return False
            
            # Wait before retry
            await asyncio.sleep(self._config.retry_interval)
        
        logger.warning("Failed to acquire lock %s after timeout", lock_key_with_prefix)
        return False
    
    async def release(self, lock_key: str, lock_value: str) -> bool:
        """Release a distributed lock.
        
        Uses Lua script to ensure only the lock owner can release it.
        
        Args:
            lock_key: Unique key for the lock
            lock_value: Value that was used when acquiring the lock
            
        Returns:
            True if lock was released, False otherwise
        """
        lock_key_with_prefix = f"lock:{lock_key}"
        
        # Lua script for safe lock release
        # Only deletes the key if it exists and has the expected value
        lua_script = """
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("DEL", KEYS[1])
        else
            return 0
        end
        """
        
        try:
            result = await self._redis.eval(
                lua_script,
                1,
                lock_key_with_prefix,
                lock_value,
            )
            
            if result:
                logger.debug("Released distributed lock: %s", lock_key_with_prefix)
                return True
            else:
                logger.warning(
                    "Failed to release lock %s - either not owned or expired",
                    lock_key_with_prefix,
                )
                return False
                
        except Exception as exc:
            logger.error("Failed to release lock %s: %s", lock_key_with_prefix, exc)
            return False
    
    @asynccontextmanager
    async def lock_context(self, lock_key: str) -> AsyncIterator[None]:
        """Context manager for distributed lock.
        
        Args:
            lock_key: Unique key for the lock
            
        Yields:
            None when lock is acquired
            
        Raises:
            RuntimeError: If lock cannot be acquired
        """
        lock_value = str(uuid.uuid4())
        acquired = await self.acquire(lock_key, lock_value)
        
        if not acquired:
            raise RuntimeError(f"Failed to acquire distributed lock for {lock_key}")
        
        try:
            yield
        finally:
            await self.release(lock_key, lock_value)


class DistributedLockManager:
    """Manager for distributed locks with typed lock key generation."""
    
    def __init__(self, redis_client: aioredis.Redis):
        self._redis = redis_client
        self._lock = DistributedLock(redis_client)
    
    @asynccontextmanager
    async def thread_state_lock(self, thread_id: str) -> AsyncIterator[None]:
        """Lock for preventing concurrent modifications to thread state.
        
        Args:
            thread_id: LangGraph thread identifier
            
        Yields:
            None when lock is acquired
        """
        lock_key = f"thread_state:{thread_id}"
        async with self._lock.lock_context(lock_key):
            yield
    
    @asynccontextmanager
    async def document_cache_lock(self, document_id: str) -> AsyncIterator[None]:
        """Lock for preventing concurrent cache operations on documents.
        
        Args:
            document_id: Document identifier
            
        Yields:
            None when lock is acquired
        """
        lock_key = f"document_cache:{document_id}"
        async with self._lock.lock_context(lock_key):
            yield
    
    @asynccontextmanager
    async def user_session_lock(self, user_id: str) -> AsyncIterator[None]:
        """Lock for preventing concurrent operations on user sessions.
        
        Args:
            user_id: User identifier
            
        Yields:
            None when lock is acquired
        """
        lock_key = f"user_session:{user_id}"
        async with self._lock.lock_context(lock_key):
            yield
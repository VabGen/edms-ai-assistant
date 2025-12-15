# edms_ai_assistant/utils/retry_utils.py

import asyncio
import logging
from functools import wraps
from typing import Callable, Any, Tuple, Dict, Type

logger = logging.getLogger(__name__)


def async_retry(
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Асинхронный декоратор с логикой повторных попыток.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    is_last_attempt = (attempt == max_attempts - 1)

                    if is_last_attempt:
                        logger.error(
                            f"Fatal error after {max_attempts} attempts calling {func.__name__}: {type(e).__name__}: {e}"
                        )
                        raise

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}. Retrying in {current_delay:.2f}s."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            return None

        return wrapper

    return decorator
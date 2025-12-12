# edms_ai_assistant/utils/retry_utils.py
import asyncio
import logging
from functools import wraps
from typing import Callable, Type, Tuple, Any, Awaitable, Optional

logger = logging.getLogger(__name__)

AsyncCallable = Callable[..., Awaitable[Any]]


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Декоратор для выполнения асинхронной функции с повторными попытками (экспоненциальный откат).

    Args:
        max_attempts: Максимальное количество попыток (включая первую).
        delay: Начальная задержка между попытками в секундах.
        backoff: Множитель для увеличения задержки после каждой неудачной попытки.
        exceptions: Кортеж типов исключений, при которых следует повторять попытку.
    """

    def decorator(func: AsyncCallable) -> AsyncCallable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception: Optional[Exception] = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {type(e).__name__}: {e}. Retrying in {current_delay:.2f}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}. Last error: {type(e).__name__}: {e}"
                        )

            if last_exception is not None:
                raise last_exception

            raise RuntimeError(
                f"Retry mechanism failed for {func.__name__} without capturing an exception."
            )

        return wrapper

    return decorator

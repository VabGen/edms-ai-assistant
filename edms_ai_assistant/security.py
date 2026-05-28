# edms_ai_assistant/security.py
import logging
import jwt
from typing import Any
from edms_ai_assistant.config import settings

logger = logging.getLogger(__name__)


def _sanitize_token(token: str) -> str:
    """Удаляет префикс Bearer и лишние пробелы."""
    token = token.strip()
    if token.lower().startswith("bearer "):
        return token[7:].strip()
    return token


def extract_user_id_from_token(user_token: str) -> str:
    """
    Декодирует и валидирует JWT для извлечения ID пользователя ('id' или 'sub').

    Использует JWT_SECRET_KEY и JWT_ALGORITHM из настроек приложения.
    В режиме DEBUG=True допускает использование невалидной подписи.
    """
    try:
        payload = decode_token(user_token)
        return _extract_id_from_payload(payload)
    except jwt.ExpiredSignatureError:
        logger.error("Срок действия токена истек")
        raise ValueError("Срок действия токена истек") from None
    except jwt.InvalidTokenError as e:
        logger.error(f"Невалидный JWT токен: {e}")
        raise ValueError(f"Невалидный токен: {e}") from e
    except Exception as e:
        logger.error(f"Ошибка при обработке JWT: {e}")
        raise ValueError("Внутренняя ошибка при проверке токена.") from e


def decode_token(user_token: str) -> dict[str, Any]:
    """
    Полное декодирование и валидация токена.
    """
    token = _sanitize_token(user_token)
    try:
        return jwt.decode(
            token,
            settings.JWT_SECRET_KEY.get_secret_value(),
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.InvalidTokenError:
        if settings.DEBUG:
            return jwt.decode(token, options={"verify_signature": False})
        raise


def _extract_id_from_payload(payload: dict[str, Any]) -> str:
    user_id = str(payload.get("id") or payload.get("sub") or "")
    if not user_id:
        raise ValueError("User ID ('id' или 'sub') не найдены в полезной нагрузке JWT.")
    return user_id

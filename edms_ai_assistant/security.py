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
        extracted = token[7:].strip()
        if not extracted:
            logger.warning("Empty Bearer token detected (header contains 'Bearer' with no token)")
            raise ValueError("Bearer token is empty")
        return extracted
    if not token:
        logger.warning("Empty authorization token detected")
        raise ValueError("Authorization token is empty")
    return token


def extract_user_id_from_token(user_token: str) -> str:
    """
    Декодирует и валидирует JWT для извлечения ID пользователя ('id' или 'sub').

    Использует JWT_SECRET_KEY и JWT_ALGORITHM из настроек приложения.
    SECURITY: Signature verification is always enforced.
    
    Args:
        user_token: JWT токен из Authorization header
        
    Returns:
        ID пользователя из payload
        
    Raises:
        ValueError: Если токен невалиден или отсутствует user_id
    """
    try:
        payload = decode_token(user_token)
        return _extract_id_from_payload(payload)
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired during user_id extraction")
        raise ValueError("Срок действия токена истек") from None
    except jwt.InvalidTokenError as e:
        logger.warning("Invalid token during user_id extraction", extra={"error": str(e)})
        raise ValueError(f"Невалидный токен: {e}") from e
    except Exception as e:
        logger.error("Unexpected error during user_id extraction", exc_info=True)
        raise ValueError("Внутренняя ошибка при проверке токена.") from e


def decode_token(user_token: str) -> dict[str, Any]:
    """
    Полное декодирование и валидация токена.
    
    SECURITY: Signature verification is ALWAYS enforced regardless of environment.
    In development mode, invalid tokens will still raise proper security errors.
    """
    token = _sanitize_token(user_token)
    
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY.get_secret_value(),
            algorithms=[settings.JWT_ALGORITHM],
            options={
                "verify_signature": True,  # Always verify signature
                "require": ["exp"],  # Require expiration
            }
        )

        user_id = _extract_id_from_payload(payload)
        logger.info("JWT token decoded successfully", extra={"user_id": user_id})
        
        return payload
        
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired", extra={"token_prefix": token[:20]})
        raise
    except jwt.InvalidSignatureError:
        logger.error(
            "JWT token has invalid signature - possible forgery attempt",
            extra={"token_prefix": token[:20]}
        )
        raise
    except jwt.InvalidTokenError as exc:
        logger.warning(
            "JWT token validation failed",
            extra={"error": str(exc), "token_prefix": token[:20]}
        )
        raise
    except Exception as exc:
        logger.error(
            "Unexpected error during JWT token validation",
            extra={"error": str(exc), "token_prefix": token[:20]},
            exc_info=True
        )
        raise


def _extract_id_from_payload(payload: dict[str, Any]) -> str:
    user_id = str(payload.get("id") or payload.get("sub") or "")
    if not user_id:
        raise ValueError("User ID ('id' или 'sub') не найдены в полезной нагрузке JWT.")
    return user_id

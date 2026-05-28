# edms_ai_assistant/security.py
import logging
import jwt
from edms_ai_assistant.config import settings

logger = logging.getLogger(__name__)


def extract_user_id_from_token(user_token: str) -> str:
    """
    Декодирует и валидирует JWT для извлечения ID пользователя ('id' или 'sub').

    Использует JWT_SECRET_KEY и JWT_ALGORITHM из настроек приложения.
    """
    try:
        token = user_token.strip()
        if token.startswith("Bearer "):
            token = token[7:]
            logger.debug("Удален префикс 'Bearer ' из токена")

        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY.get_secret_value(),
            algorithms=[settings.JWT_ALGORITHM],
        )

        user_id = str(payload.get("id") or payload.get("sub"))

        if not user_id:
            raise ValueError(
                "User ID ('id' или 'sub') не найдены в полезной нагрузке JWT."
            )

        logger.debug(f"Успешно извлечен и валидирован user_id: {user_id}")
        return user_id

    except jwt.ExpiredSignatureError:
        logger.error("Срок действия токена истек")
        raise ValueError("Срок действия токена истек") from None
    except jwt.InvalidTokenError as e:
        logger.error(f"Невалидный JWT токен: {e}")
        raise ValueError(f"Невалидный токен: {e}") from e
    except Exception as e:
        logger.error(f"Ошибка при обработке JWT: {e}")
        raise ValueError("Внутренняя ошибка при проверке токена.") from e


def decode_token(user_token: str) -> dict:
    """
    Полное декодирование и валидация токена.

    Token comes directly from FastAPI HTTPBearer (no Bearer prefix).
    """
    return jwt.decode(
        user_token.strip(),
        settings.JWT_SECRET_KEY.get_secret_value(),
        algorithms=[settings.JWT_ALGORITHM],
    )

# edms_ai_assistant/utils/api_utilss.py
import uuid
import httpx
import logging
from typing import Optional, Dict, Any, List, Union

from fastapi import HTTPException

logger = logging.getLogger(__name__)


# --- Обработка ошибок API ---
async def handle_api_error(response: httpx.Response, operation_name: str = "API call"):
    """
    Вспомогательная функция для обработки ошибок HTTP-ответа API.

    Если ответ содержит ошибку (4xx или 5xx), детали ошибки логируются,
    и затем вызывается response.raise_for_status(), что поднимает httpx.HTTPStatusError.

    Args:
        response: Объект httpx.Response.
        operation_name: Имя операции для логирования.
    """
    if response.is_error:
        error_details = "No details available."

        try:
            error_json = response.json()
            error_details = f" Error details: {error_json}"
        except httpx.JSONDecodeError:
            # 2. Если JSON невалиден, используем текстовое тело ответа
            error_details = f" Error response text: {response.text[:200]}..."
        except Exception:
            error_details = (
                f" Failed to decode error body, status code {response.status_code}."
            )

        logger.error(
            f"{operation_name} failed with status {response.status_code}."
            f"{error_details}"
        )
        response.raise_for_status()
    else:
        logger.debug(f"{operation_name} successful, status {response.status_code}.")


# -------------------------------------------------------------
# --- Подготовка заголовков и Валидация ---
# -------------------------------------------------------------


def prepare_auth_headers(token: str) -> Dict[str, str]:
    """
    Подготавливает заголовки авторизации в формате Bearer.

    Args:
        token: Токен аутентификации.

    Returns:
        Словарь заголовков.
    """
    return {"Authorization": f"Bearer {token}"}


def validate_document_data(data: Dict[str, Any]) -> bool:
    """
    Простая валидация данных документа перед отправкой в API.

    Args:
        data: Словарь с данными документа.

    Returns:
        True, если данные валидны, иначе False.
    """
    required_fields = ["name", "type"]
    for field in required_fields:
        if field not in data or not data[field]:
            logger.error(
                f"Validation failed: Missing or empty required field '{field}'."
            )
            return False
    return True


def validate_document_id(doc_id: Optional[str]) -> Optional[uuid.UUID]:
    """
    Валидация идентификатора документа. Если невалиден, поднимает HTTPException (для интеграции с FastAPI).
    """
    if doc_id is None:
        return None
    try:
        return uuid.UUID(doc_id)
    except ValueError:
        # В контексте сервиса, вызываемого API, лучше поднять HTTPException
        raise HTTPException(
            status_code=400, detail="Invalid document_id format. Must be a valid UUID."
        )


# -------------------------------------------------------------
# --- Обработка пагинации ---
# -------------------------------------------------------------


async def fetch_all_pages(
    client: httpx.AsyncClient,
    base_url: str,
    endpoint: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Union[str, int, float]]] = None,
    page_param: str = "page",
    page_start: int = 0,
    size_param: str = "size",
    default_page_size: int = 20,
) -> List[Dict[str, Any]]:
    """
    Асинхронно извлекает все страницы paginated API-ответа (пагинация "по страницам").

    Args:
        client: Экземпляр httpx.AsyncClient.
        base_url: Базовый URL API.
        endpoint: Конечная точка API.
        headers: Заголовки для запроса.
        params: Базовые параметры запроса.
        page_param: Имя параметра для номера страницы (по умолчанию "page").
        page_start: Стартовый номер страницы (обычно 0 или 1).
        size_param: Имя параметра размера страницы.
        default_page_size: Размер страницы.

    Returns:
        Список всех элементов из всех страниц.
    """
    all_items: List[Dict[str, Any]] = []
    base_params = params.copy() if params else {}

    if size_param not in base_params:
        base_params[size_param] = default_page_size

    page = page_start

    while True:
        request_params = {**base_params, page_param: page}
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        operation_name = f"Fetch {page_param}={page} of {endpoint}"

        try:
            response = await client.get(url, headers=headers, params=request_params)
            await handle_api_error(response, operation_name)
            data: Dict[str, Any] = response.json()

            items = data.get("content", data.get("items", data.get("results", [])))

            all_items.extend(items)

            if not items:
                logger.debug(f"{operation_name}: No more items found. Stopping.")
                break

            total_pages = data.get("totalPages")
            total_elements = data.get("totalElements")

            if total_pages is not None:
                if page >= total_pages - 1:
                    logger.debug(
                        f"{operation_name}: Reached total pages limit ({total_pages}). Stopping."
                    )
                    break
            elif total_elements is not None:
                fetched = len(all_items)
                if fetched >= total_elements:
                    logger.debug(
                        f"{operation_name}: Fetched all {total_elements} elements. Stopping."
                    )
                    break
            elif len(items) < base_params[size_param]:
                logger.debug(
                    f"{operation_name}: Last page detected (partial size). Stopping."
                )
                break

            page += 1

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Terminating pagination for {endpoint} due to HTTP error on page {page}."
            )
            break
        except Exception as e:
            logger.error(
                f"Unexpected error fetching {page_param}={page} of {endpoint}: {type(e).__name__}: {e}"
            )
            break

    return all_items

"""Low-level async HTTP session wrapping ``httpx.AsyncClient``.

This module owns the single ``httpx.AsyncClient`` instance and all
wire-level concerns: auth headers, timeout selection, status-code
validation, JSON/bytes deserialisation, and retry orchestration.

Higher-level clients receive a ``HttpSession`` via dependency injection and
never instantiate ``httpx`` directly.
"""

from __future__ import annotations

import json as _json
import logging
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from edms_ai_assistant.clients._transport.retry import async_retry
from edms_ai_assistant.clients.config import EdmsClientConfig
from edms_ai_assistant.clients.exceptions import (
    EdmsAuthError,
    EdmsFileNotFoundError,
    EdmsHttpError,
    EdmsNotFoundError,
    EdmsSerializationError,
    EdmsTransportError,
)

logger = logging.getLogger(__name__)

# Params type accepted by httpx for query strings.
# httpx handles list values correctly (multi-value params).
QueryParams = dict[str, Any] | list[tuple[str, Any]] | None


def _build_auth_headers(token: str) -> dict[str, str]:
    """Return standard Bearer-auth + JSON content-type headers."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _raise_for_status(response: httpx.Response, endpoint: str) -> None:
    """Convert HTTP error status codes to typed exceptions."""
    if response.is_success:
        return
    body_preview = response.text[:300] if response.content else ""
    kwargs: dict[str, Any] = {
        "status_code": response.status_code,
        "endpoint": endpoint,
        "response_body": body_preview,
    }
    if response.status_code in (401, 403):
        raise EdmsAuthError(
            f"Authentication/authorisation failure ({response.status_code})",
            **kwargs,
        )
    if response.status_code == 404:
        raise EdmsNotFoundError(f"Resource not found: {endpoint}", **kwargs)
    raise EdmsHttpError(
        f"HTTP {response.status_code} from {endpoint}",
        **kwargs,
    )


class HttpSession:
    """Async HTTP session configured for the EDMS API.

    Lifecycle
    ---------
    Use as an async context manager::

        async with HttpSession(config) as session:
            data = await session.request_json("GET", "api/document/...", token=token)

    Or manage manually::

        session = HttpSession(config)
        await session.open()
        ...
        await session.close()

    The session is **not** thread-safe but is safe to share across coroutines
    within the same event loop (httpx.AsyncClient is coroutine-safe).
    """

    def __init__(self, config: EdmsClientConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def open(self) -> None:
        """Initialise the underlying httpx client if not already open."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._config.timeout,
                # Keep-alive is httpx default; set limits to avoid fd exhaustion.
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=30,
                ),
            )

    async def close(self) -> None:
        """Gracefully close the underlying httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "HttpSession":
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.close()

    # ── Public request methods ────────────────────────────────────────────────

    async def request_json(
        self,
        method: str,
        endpoint: str,
        *,
        token: str,
        params: QueryParams = None,
        json: Any = None,
        long_timeout: bool = False,
    ) -> Any:
        """Execute a request and return the deserialised JSON body.

        Returns an empty dict ``{}`` for 204 No Content responses.

        Raises:
            EdmsAuthError: 401/403.
            EdmsNotFoundError: 404.
            EdmsHttpError: Any other 4xx/5xx.
            EdmsTransportError: Network-level failures.
            EdmsSerializationError: Response body is not valid JSON.
        """
        response = await self._raw_request(
            method,
            endpoint,
            token=token,
            params=params,
            json=json,
            long_timeout=long_timeout,
        )
        if response.status_code == 204 or not response.content:
            return {}
        try:
            return response.json()
        except _json.JSONDecodeError as exc:
            raise EdmsSerializationError(
                f"Response from {endpoint} is not valid JSON",
                endpoint=endpoint,
            ) from exc

    async def request_bytes(
        self,
        method: str,
        endpoint: str,
        *,
        token: str,
        params: QueryParams = None,
        long_timeout: bool = False,
    ) -> bytes:
        """Execute a request and return the raw response bytes.

        Raises:
            EdmsAuthError, EdmsNotFoundError, EdmsHttpError, EdmsTransportError.
        """
        response = await self._raw_request(
            method,
            endpoint,
            token=token,
            params=params,
            long_timeout=long_timeout,
        )
        return response.content

    async def upload_multipart(
        self,
        endpoint: str,
        *,
        token: str,
        file_path: str,
        file_name: str | None = None,
    ) -> dict[str, Any]:
        """Upload a local file as ``multipart/form-data``.

        Args:
            endpoint:  Relative API path.
            token:     JWT bearer token.
            file_path: Absolute local path to the file.
            file_name: Display name override (defaults to the file's basename).

        Returns:
            Parsed JSON response body, or ``{}`` for 204.

        Raises:
            EdmsFileNotFoundError: If ``file_path`` does not exist.
            EdmsAuthError, EdmsNotFoundError, EdmsHttpError, EdmsTransportError.
        """
        path = Path(file_path)
        if not path.exists():
            raise EdmsFileNotFoundError(file_path)

        display_name = (file_name or path.name).strip()
        content_type, _ = mimetypes.guess_type(display_name)
        content_type = content_type or "application/octet-stream"

        url = self._url(endpoint)
        # Strip Content-Type so httpx sets multipart boundary automatically.
        headers = {
            k: v
            for k, v in _build_auth_headers(token).items()
            if k.lower() != "content-type"
        }
        timeout = self._config.timeout

        logger.info("Uploading '%s' (%s) → %s", display_name, content_type, endpoint)

        client = self._assert_open()
        with open(file_path, "rb") as fh:
            response = await client.post(
                url,
                headers=headers,
                files={"file": (display_name, fh, content_type)},
                timeout=timeout,
            )

        _raise_for_status(response, endpoint)

        if response.status_code == 204 or not response.content:
            return {}

        try:
            return response.json()  # type: ignore[no-any-return]
        except _json.JSONDecodeError as exc:
            raise EdmsSerializationError(
                f"Multipart upload response from {endpoint} is not valid JSON",
                endpoint=endpoint,
            ) from exc

    # ── Internal helpers ──────────────────────────────────────────────────────

    @async_retry()
    async def _raw_request(
        self,
        method: str,
        endpoint: str,
        *,
        token: str,
        params: QueryParams,
        json: Any = None,
        long_timeout: bool = False,
    ) -> httpx.Response:
        """Send the HTTP request with retry semantics; return the raw Response."""
        url = self._url(endpoint)
        headers = _build_auth_headers(token)
        timeout = (
            self._config.timeout + self._config.long_timeout_extra
            if long_timeout
            else self._config.timeout
        )

        try:
            client = self._assert_open()
            response = await client.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json,
                timeout=timeout,
            )
        except httpx.RequestError as exc:
            raise EdmsTransportError(
                f"Network error on {method} {endpoint}: {exc}",
                endpoint=endpoint,
            ) from exc

        _raise_for_status(response, endpoint)
        return response

    def _url(self, endpoint: str) -> str:
        return f"{self._config.base_url_str}/{endpoint.lstrip('/')}"

    def _assert_open(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "HttpSession is not open. "
                "Use 'async with HttpSession(config) as session:' or call open() first."
            )
        return self._client

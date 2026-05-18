"""
EDMS AI Assistant — Access Grief HTTP Client.

Клиент для AccessGriefController.java.

  GET  /api/access-grief                    → search_griefs()    (SliceDto<AccessGriefDto>)
  GET  /api/access-grief/{id}               → get_grief()        (AccessGriefDto)
  GET  /api/access-grief/{id}/employees     → get_grief_employees() (SliceDto<EmployeeAccessGriefDto>)

Намеренно НЕ реализованы:
  - POST / PUT / DELETE — управление грифами не входит в сценарии агента
"""

from __future__ import annotations

import logging
from typing import Any

from .base_client import EdmsBaseClient

logger = logging.getLogger(__name__)

_DEFAULT_PAGE: int = 0
_DEFAULT_SIZE: int = 50


class AccessGriefClient(EdmsBaseClient):
    """Async HTTP client for EDMS Access Grief API."""

    async def search_griefs(
        self,
        token: str,
        name: str | None = None,
        pageable: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Searches access griefs. GET /api/access-grief.

        Args:
            token: JWT bearer token.
            name: Filter by grief name (partial match).
            pageable: {'page': int, 'size': int, 'sort': str}.

        Returns:
            List of AccessGriefDto dicts.
        """
        params: dict[str, Any] = {}
        if name:
            params["name"] = name
        params.update(_build_pageable_params(pageable))

        logger.debug(
            "Searching access griefs",
            extra={"name": name},
        )

        result = await self._make_request(
            "GET", "api/access-grief", token=token, params=params
        )
        return _extract_slice_content(result, endpoint="GET api/access-grief")

    async def get_grief(self, token: str, grief_id: str) -> dict[str, Any] | None:
        """Gets a single access grief by UUID. GET /api/access-grief/{id}.

        Args:
            token: JWT bearer token.
            grief_id: AccessGrief UUID string.

        Returns:
            AccessGriefDto dict, or None if not found.
        """
        try:
            result = await self._make_request(
                "GET", f"api/access-grief/{grief_id}", token=token
            )
            return result if isinstance(result, dict) and result else None
        except Exception:
            logger.warning("Failed to fetch grief %s", grief_id[:8], exc_info=True)
            return None

    async def get_grief_employees(
        self,
        token: str,
        grief_id: str,
        pageable: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Gets employees with a specific access grief. GET /api/access-grief/{id}/employees.

        Args:
            token: JWT bearer token.
            grief_id: AccessGrief UUID string.
            pageable: Pagination params.

        Returns:
            List of EmployeeAccessGriefDto dicts.
        """
        params = _build_pageable_params(pageable)

        logger.debug(
            "Fetching employees for grief",
            extra={"grief_id": grief_id[:8]},
        )

        result = await self._make_request(
            "GET",
            f"api/access-grief/{grief_id}/employees",
            token=token,
            params=params,
        )
        return _extract_slice_content(
            result, endpoint="GET api/access-grief/{id}/employees"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════════════


def _build_pageable_params(pageable: dict[str, Any] | None) -> dict[str, Any]:
    effective: dict[str, Any] = {"page": _DEFAULT_PAGE, "size": _DEFAULT_SIZE}
    if pageable:
        effective.update(pageable)
    return effective


def _extract_slice_content(
    result: Any,
    endpoint: str = "api/access-grief",
) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            return content
        # Single object returned directly
        if result.get("id"):
            return [result]
        logger.warning(
            "Unexpected response from %s: dict without 'content'",
            endpoint,
            extra={"response_keys": list(result.keys())},
        )
        return []

    if isinstance(result, list):
        return result

    return []

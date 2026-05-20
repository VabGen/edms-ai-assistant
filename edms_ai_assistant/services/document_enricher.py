# edms_ai_assistant/services/document_enricher.py
"""
EDMS AI Assistant — Document Enricher Service.

Покрываемые связи:
    correspondentId               → GET /api/correspondent/{id}
    journalId                     → GET /api/reg-journal/{id}
    documentTypeId                → GET /api/document-type/{id}
    currencyId                    → GET /api/currency/{id}
    control.controlTypeId         → GET /api/control-type/{id}
    doc.id (introduction)         → GET /api/document/{id}/introduction
    doc.id (appeal)               → GET /api/document-appeal/document/{id}
    doc.id (recipientList)        → GET /api/document/{id}/recipient
    doc.id (taskList+executors)   → GET /api/document/{id}/task?fetchExecutors=true
    doc.id (responsible)          → GET /api/document/{id}/responsible  (CONTRACT only)
    recipientList[N].correspondentId → GET /api/correspondent/{id}  (batch)

Использование:
    enricher = DocumentEnricher(base_client=http_client)
    enriched_doc_dict = await enricher.enrich(raw_doc_dict, token=jwt_token)
    # Затем enriched_doc_dict парсится в DocumentDto на уровне сервиса
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from edms_ai_assistant.clients.base_client import EdmsBaseClient
from edms_ai_assistant.core.exceptions import EdmsNotFoundError

logger = logging.getLogger(__name__)

RawDoc = dict[str, Any]


class DocumentEnricher:
    """Enriches a raw DocumentDto dict with data from secondary API endpoints.

    Берёт сырой dict из DocumentClient.get_document_metadata() и параллельно
    дозапрашивает связанные объекты по UUID-полям через базовый HTTP клиент.

    Все запросы выполняются через asyncio.gather — максимальный параллелизм,
    минимальная задержка.

    Принцип «не ломать»: если любой доп. запрос упал — поле остаётся как есть
    (UUID-строка), enrich() не бросает исключений наружу.
    """

    def __init__(self, base_client: EdmsBaseClient) -> None:
        self._client = base_client

    # ── Внутренние хелперы для запросов ────────────────────────────────────

    async def _fetch_single(self, token: str, endpoint: str) -> dict[str, Any] | None:
        """Делает GET запрос и возвращает JSON или None (если 404 или ошибка)."""
        try:
            response = await self._client._make_request("GET", endpoint, token=token)
            if isinstance(response, dict) and response:
                return response
            return None
        except EdmsNotFoundError:
            return None
        except Exception as exc:
            logger.warning("Enrichment sub-request failed for %s: %s", endpoint, exc)
            return None

    async def _fetch_list(self, token: str, endpoint: str) -> list[dict[str, Any]]:
        """Делает GET запрос и возвращает список."""
        try:
            response = await self._client._make_request("GET", endpoint, token=token)
            if isinstance(response, list):
                return response
            if isinstance(response, dict):
                # Поддержка Spring Page/Slice оберток
                content = response.get("content") or response.get("items")
                if isinstance(content, list):
                    return content
            return []
        except EdmsNotFoundError:
            return []
        except Exception as exc:
            logger.warning("Enrichment list request failed for %s: %s", endpoint, exc)
            return []

    # ── Основной метод обогащения ──────────────────────────────────────────

    async def enrich(self, doc: RawDoc, token: str) -> RawDoc:
        """Enrich a raw DocumentDto dict with all secondary API data.

        Args:
            doc: Raw DocumentDto dict from DocumentClient.
            token: JWT bearer token for all sub-requests.

        Returns:
            Same dict mutated in-place with enriched nested objects.
        """
        if not doc:
            return doc

        doc_id: str | None = str(doc.get("id", "")) or None
        category: str = str(doc.get("docCategoryConstant", "") or "")

        tasks: dict[str, Any] = {}

        # ── Планирование запросов ─────────────────────────────────────────

        if doc.get("correspondentId") and not _has_nested(doc, "correspondent"):
            tasks["correspondent"] = self._fetch_single(token, f"api/correspondent/{doc['correspondentId']}")

        if doc.get("journalId") and not _has_nested(doc, "registrationJournal"):
            tasks["registrationJournal"] = self._fetch_single(token, f"api/reg-journal/{doc['journalId']}")

        if doc.get("documentTypeId") and not _has_nested(doc, "documentType"):
            tasks["documentType"] = self._fetch_single(token, f"api/document-type/{doc['documentTypeId']}")

        if doc.get("currencyId") and not _has_nested(doc, "currency"):
            tasks["currency"] = self._fetch_single(token, f"api/currency/{doc['currencyId']}")

        control = doc.get("control") or {}
        if isinstance(control, dict) and control.get("controlTypeId") and not _has_nested(control, "controlType"):
            tasks["_controlType"] = self._fetch_single(token, f"api/control-type/{control['controlTypeId']}")

        if doc_id and not doc.get("introduction"):
            tasks["introduction"] = self._fetch_list(token, f"api/document/{doc_id}/introduction")

        if doc_id and category == "APPEAL" and not doc.get("documentAppeal"):
            tasks["documentAppeal"] = self._fetch_single(token, f"api/document-appeal/document/{doc_id}")

        if doc_id and not doc.get("recipientList"):
            tasks["recipientList"] = self._fetch_list(token, f"api/document/{doc_id}/recipient")

        if doc_id and not doc.get("taskList"):
            tasks["taskList"] = self._fetch_list(token, f"api/document/{doc_id}/task?fetchExecutors=true")

        if doc_id and category == "CONTRACT" and not doc.get("contractResponsible"):
            tasks["contractResponsible"] = self._fetch_list(token, f"api/document/{doc_id}/responsible")

        # ── Параллельное выполнение ───────────────────────────────────────

        if tasks:
            task_keys = list(tasks.keys())
            task_coros = [tasks[k] for k in task_keys]

            results = await asyncio.gather(*task_coros, return_exceptions=True)

            for key, result in zip(task_keys, results):
                if isinstance(result, Exception):
                    logger.warning("Enrichment task %s failed: %s", key, result)
                    continue

                if key == "_controlType":
                    if isinstance(doc.get("control"), dict) and result:
                        doc["control"]["controlType"] = result
                elif result is not None:
                    doc[key] = result

        # ── Корреспонденты внутри адресатов (batch) ──────────────────────
        recipient_list: list[dict[str, Any]] = doc.get("recipientList") or []
        recipient_corr_tasks = []
        for idx, recipient in enumerate(recipient_list):
            if isinstance(recipient, dict) and recipient.get("correspondentId") and not _has_nested(recipient, "correspondent"):
                recipient_corr_tasks.append(
                    (idx, self._fetch_single(token, f"api/correspondent/{recipient['correspondentId']}"))
                )

        if recipient_corr_tasks:
            indices = [idx for idx, _ in recipient_corr_tasks]
            corr_coros = [coro for _, coro in recipient_corr_tasks]
            corr_results = await asyncio.gather(*corr_coros, return_exceptions=True)

            for idx, result in zip(indices, corr_results):
                if isinstance(result, dict) and isinstance(recipient_list[idx], dict):
                    recipient_list[idx]["correspondent"] = result

        return doc


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _has_nested(obj: dict[str, Any], key: str) -> bool:
    """Return True if *obj* already has a non-empty nested object at *key*.

    Prevents redundant API calls when includes already populated the field.
    """
    val = obj.get(key)
    if isinstance(val, dict):
        return bool(val)
    if isinstance(val, list):
        return bool(val)
    return False
# edms_ai_assistant/interactive_status_detector.py
"""
InteractiveStatusDetector — Single Responsibility: detect requires_choice /
requires_disambiguation / requires_action from the last ToolMessage and
convert it into an AgentResponse dict.

Extracted from EdmsDocumentAgent._detect_interactive_status (~250 lines).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

from edms_ai_assistant.agent import ActionType, AgentResponse, AgentStatus

logger = logging.getLogger(__name__)


class InteractiveStatusDetector:
    """
    Implements IInteractiveStatusDetector.

    Scans the last ToolMessage for interactive statuses and converts
    them into serialized AgentResponse dicts ready for the HTTP layer.

    Does NOT build final content responses — that is ResponseAssembler's job.
    """

    def detect(self, messages: list[BaseMessage]) -> dict[str, Any] | None:
        last_tool_msg = next(
            (m for m in reversed(messages) if isinstance(m, ToolMessage)),
            None,
        )
        if last_tool_msg is None:
            return None

        raw = str(last_tool_msg.content).strip()
        if not raw.startswith("{"):
            return None

        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            return None

        status = data.get("status", "")
        if status not in ("requires_choice", "requires_disambiguation", "requires_action"):
            return None

        logger.info(
            "Interactive status detected: status=%s keys=%s",
            status,
            list(data.keys()),
        )

        if status == "requires_choice":
            return self._handle_requires_choice(data)
        if status == "requires_disambiguation":
            return self._handle_requires_disambiguation(data)
        if status == "requires_action":
            return self._handle_requires_action(data)

        return None

    # ── Handlers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _handle_requires_choice(data: dict[str, Any]) -> dict[str, Any]:
        options = data.get("options", [])
        hint = data.get("hint", "extractive")
        hint_reason = data.get("hint_reason", "")
        msg = data.get("message", "Выберите формат анализа:")

        options_lines = [
            f"- **{opt['key']}** — {opt['label']}: {opt['description']}"
            for opt in options
            if isinstance(opt, dict)
        ]
        hint_text = (
            f"\n\n💡 *Рекомендация:* **{hint}** — {hint_reason}" if hint_reason else ""
        )
        full_message = (
            f"{msg}\n\n" + "\n".join(options_lines) + hint_text
            + "\n\nОтветьте: **extractive**, **abstractive** или **thesis**."
        )
        return AgentResponse(
            status=AgentStatus.REQUIRES_ACTION,
            action_type=ActionType.SUMMARIZE_SELECTION,
            message=full_message,
        ).model_dump()

    @staticmethod
    def _handle_requires_disambiguation(data: dict[str, Any]) -> dict[str, Any]:
        _KNOWN_LIST_KEYS = (
            "available_attachments",
            "available_employees",
            "candidates",
            "employees",
            "results",
            "items",
            "users",
        )

        available: list[dict[str, Any]] = next(
            (v for k in _KNOWN_LIST_KEYS if isinstance(v := data.get(k), list) and v),
            [],
        )

        # Fallback: unwrap nested "matches"
        if not available:
            for _k, _v in data.items():
                if _k == "options" or not isinstance(_v, list) or not _v:
                    continue
                first = _v[0]
                if isinstance(first, dict) and "matches" in first and not first.get("id"):
                    flat: list[dict[str, Any]] = []
                    for group in _v:
                        flat.extend(group.get("matches", []))
                    if flat:
                        available = flat
                        break
                else:
                    available = _v
                    break

        base_msg = (
            re.sub(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                "",
                data.get("message", "Уточните выбор:"),
                flags=re.I,
            )
            .strip()
            .rstrip("с «»")
            .strip()
        ) or "Уточните выбор:"

        candidates = [
            {
                "id": str(
                    item.get("id") or item.get("uuid")
                    or item.get("employeeId") or item.get("userId") or "?"
                ),
                "name": (
                    item.get("fullName") or item.get("full_name") or item.get("fio")
                    or " ".join(filter(None, [
                        item.get("lastName", ""),
                        item.get("firstName", ""),
                        item.get("middleName", ""),
                    ])).strip()
                    or item.get("name") or item.get("email", "").split("@")[0]
                    or "Без имени"
                ).strip(),
                "dept": (
                    item.get("department") or item.get("departmentName")
                    or item.get("post") or item.get("position") or ""
                ).strip(),
            }
            for item in available
            if isinstance(item, dict)
        ]

        candidates_json = json.dumps(candidates, ensure_ascii=False)
        full_msg = f"{base_msg}\n\n<!--CANDIDATES:{candidates_json}-->"

        return AgentResponse(
            status=AgentStatus.REQUIRES_ACTION,
            action_type=ActionType.DISAMBIGUATION,
            message=full_msg,
        ).model_dump()

    @staticmethod
    def _handle_requires_action(data: dict[str, Any]) -> dict[str, Any] | None:
        choices: list[dict[str, Any]] = data.get("choices", [])
        base_msg = data.get("message", "Выберите сотрудника:")

        candidates = [
            {
                "id": str(item.get("id", "?")),
                "name": (
                    item.get("full_name") or item.get("fullName")
                    or item.get("name") or "Без имени"
                ).strip(),
                "dept": (item.get("department") or item.get("post") or "").strip(),
            }
            for item in choices
            if isinstance(item, dict)
        ]

        if not candidates:
            return None

        candidates_json = json.dumps(candidates, ensure_ascii=False)
        full_msg = base_msg + "\n\n<!--CANDIDATES:" + candidates_json + "-->"

        logger.info(
            "requires_action/select_employee → %d candidates", len(candidates)
        )
        return AgentResponse(
            status=AgentStatus.REQUIRES_ACTION,
            action_type=ActionType.DISAMBIGUATION,
            message=full_msg,
        ).model_dump()
# edms_ai_assistant/interactive_status_detector.py
"""
InteractiveStatusDetector — Single Responsibility:
Scan the last ToolMessage for interactive statuses that require user input.

Statuses handled:
  requires_choice        → summarize type selection
  requires_disambiguation → employee / attachment disambiguation
  requires_action        → employee_search multiple matches

The <!--CANDIDATES:[...]---> marker is parsed by AssistantWidget.tsx to render
clickable candidate cards in the frontend.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

from edms_ai_assistant.model import ActionType, AgentResponse, AgentStatus

logger = logging.getLogger(__name__)

# Keys searched (in priority order) for candidate list in disambiguation payloads
_CANDIDATE_LIST_KEYS = (
    "available_attachments",
    "available_employees",
    "candidates",
    "employees",
    "results",
    "items",
    "users",
    "ambiguous_matches",  # task_create_tool / introduction_create_tool
    "matches",
)


class InteractiveStatusDetector:
    """
    Scans the LAST ToolMessage for interactive statuses.

    Only the last ToolMessage is checked — if it doesn't contain an interactive
    status the graph is still running and the LLM will produce a final answer.
    """

    def detect(self, messages: list[BaseMessage]) -> dict[str, Any] | None:
        """
        Returns a serialised AgentResponse dict if an interactive status is found,
        otherwise returns None.
        """
        last_tool_msg = self._last_tool_message(messages)
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

        if status in ("requires_disambiguation", "requires_action"):
            return self._handle_disambiguation(data, status)

        return None

    # ── Private handlers ──────────────────────────────────────────────────────

    @staticmethod
    def _last_tool_message(messages: list[BaseMessage]) -> ToolMessage | None:
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                return m
        return None

    @staticmethod
    def _handle_requires_choice(data: dict[str, Any]) -> dict[str, Any]:
        """Summarize type selection — returns three buttons."""
        options = data.get("options", [])
        hint = data.get("hint", "extractive")
        hint_reason = data.get("hint_reason", "")
        msg = data.get("message", "Выберите формат анализа:")

        options_lines = [
            f"- **{opt['key']}** — {opt['label']}: {opt['description']}"
            for opt in options
            if isinstance(opt, dict) and "key" in opt
        ]
        options_text = "\n".join(options_lines)
        hint_text = (
            f"\n\n💡 *Рекомендация:* **{hint}** — {hint_reason}" if hint_reason else ""
        )
        full_message = (
            f"{msg}\n\n{options_text}{hint_text}\n\n"
            "Ответьте: **extractive**, **abstractive** или **thesis**."
        )
        return AgentResponse(
            status=AgentStatus.REQUIRES_ACTION,
            action_type=ActionType.SUMMARIZE_SELECTION,
            message=full_message,
        ).model_dump()

    @classmethod
    def _handle_disambiguation(
        cls, data: dict[str, Any], status: str
    ) -> dict[str, Any]:
        """
        Employee / attachment disambiguation — renders <!--CANDIDATES:[...]-->.

        Handles multiple payload shapes from different tools:

        Shape A (introduction_create_tool / task_create_tool):
          {
            "status": "requires_disambiguation",
            "ambiguous_matches": [
              {
                "search_term": "Петров",
                "matches": [
                  {"id": "uuid", "firstName": "Иван", "lastName": "Петров", ...},
                  ...
                ]
              }
            ]
          }

        Shape B (doc_compare_attachment_with_local):
          {
            "status": "requires_disambiguation",
            "available_attachments": [
              {"id": "uuid", "name": "file.docx"},
              ...
            ]
          }

        Shape C (employee_search_tool requires_action):
          {
            "status": "requires_action",
            "action_type": "select_employee",
            "choices": [
              {"id": "uuid", "full_name": "...", "department": "..."},
              ...
            ]
          }
        """
        available = cls._extract_candidates(data)
        candidates = [cls._normalise_candidate(item) for item in available if isinstance(item, dict)]
        candidates = [c for c in candidates if c["id"] != "?"]

        base_msg = cls._clean_base_message(data.get("message", "Уточните выбор:"))

        if not candidates:
            # No structured candidates — return plain message
            return AgentResponse(
                status=AgentStatus.REQUIRES_ACTION,
                action_type=ActionType.DISAMBIGUATION,
                message=base_msg,
            ).model_dump()

        candidates_json = json.dumps(candidates, ensure_ascii=False)
        full_msg = f"{base_msg}\n\n<!--CANDIDATES:{candidates_json}-->"

        logger.info(
            "Disambiguation: %d candidates prepared for frontend", len(candidates)
        )

        return AgentResponse(
            status=AgentStatus.REQUIRES_ACTION,
            action_type=ActionType.DISAMBIGUATION,
            message=full_msg,
        ).model_dump()

    @classmethod
    def _extract_candidates(cls, data: dict[str, Any]) -> list[dict]:
        """
        Extracts a flat list of candidate dicts from any known payload shape.
        """
        # Shape A: ambiguous_matches → list of {search_term, matches:[...]}
        ambiguous = data.get("ambiguous_matches")
        if isinstance(ambiguous, list) and ambiguous:
            flat: list[dict] = []
            for group in ambiguous:
                if isinstance(group, dict):
                    matches = group.get("matches", [])
                    if isinstance(matches, list):
                        flat.extend(matches)
                    elif isinstance(group, dict) and "id" in group:
                        # group IS a candidate directly
                        flat.append(group)
            if flat:
                return flat

        # Shape C: choices (employee_search requires_action)
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            return choices

        # Generic: iterate known list keys
        for key in _CANDIDATE_LIST_KEYS:
            val = data.get(key)
            if isinstance(val, list) and val:
                first = val[0]
                if isinstance(first, dict):
                    return val

        # Last resort: find any list value with dict items
        for k, v in data.items():
            if k in ("status", "message", "instruction", "action_type"):
                continue
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v

        return []

    @staticmethod
    def _normalise_candidate(item: dict) -> dict[str, str]:
        """
        Normalises a candidate dict into {id, name, dept} for the frontend.
        Handles all known field name variants from EDMS employee / attachment APIs.
        """
        # ── ID ────────────────────────────────────────────────────────────────
        item_id = str(
            item.get("id")
            or item.get("uuid")
            or item.get("employeeId")
            or item.get("employee_id")
            or item.get("userId")
            or item.get("user_id")
            or item.get("personId")
            or item.get("person_id")
            or "?"
        )

        # ── Name ──────────────────────────────────────────────────────────────
        first = (
            item.get("firstName") or item.get("first_name")
            or item.get("firstname") or item.get("givenName") or ""
        ).strip()
        last = (
            item.get("lastName") or item.get("last_name")
            or item.get("lastname") or item.get("surname")
            or item.get("familyName") or ""
        ).strip()
        middle = (
            item.get("middleName") or item.get("middle_name")
            or item.get("patronymic") or ""
        ).strip()

        display_name = (
            item.get("fullName") or item.get("full_name")
            or item.get("fio") or item.get("FIO")
            or item.get("name")
            or " ".join(filter(None, [last, first, middle]))
            or item.get("username") or item.get("login")
            or (item.get("email", "").split("@")[0] if item.get("email") else "")
            or "Без имени"
        ).strip()

        # ── Department / position ─────────────────────────────────────────────
        dept = (
            item.get("department") or item.get("departmentName")
            or item.get("department_name") or item.get("division")
            or item.get("post") or item.get("position")
            or item.get("jobTitle") or item.get("job_title")
            or item.get("role") or ""
        ).strip()

        return {"id": item_id, "name": display_name, "dept": dept}

    @staticmethod
    def _clean_base_message(msg: str) -> str:
        """Strip embedded UUIDs and trailing junk from message text."""
        cleaned = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "",
            msg,
            flags=re.I,
        ).strip().rstrip("с «»").strip()
        return cleaned if cleaned else "Уточните выбор:"
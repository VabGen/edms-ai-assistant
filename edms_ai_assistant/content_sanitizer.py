# edms_ai_assistant/content_sanitizer.py
"""
ContentSanitizer — Single Responsibility: remove technical artifacts
(UUIDs, file system paths, token strings) from user-visible response text.

Extracted from EdmsDocumentAgent._sanitize_technical_content.
"""
from __future__ import annotations

import re

from edms_ai_assistant.agent import ContextParams
from edms_ai_assistant.utils.regex_utils import UUID_RE


def _is_valid_uuid(value: str) -> bool:
    return bool(UUID_RE.match(value.strip()))


class ContentSanitizer:
    """
    Removes technical artifacts from agent responses before delivery.

    Implements ISanitizer protocol.
    Single Responsibility: text sanitization only.
    """

    _FS_PATH_RE = re.compile(
        r"[A-Za-z]:\\[^\s,;)'\"]{3,}|/(?:tmp|var|home|uploads)/[^\s,;)'\"]{3,}"
    )
    _HASH_FILENAME_RE = re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        r"_[0-9a-f]{32}\.[a-zA-Z]{2,5}",
        re.I,
    )
    _SHORT_HASH_RE = re.compile(r"_?[0-9a-f]{32}\.[a-zA-Z]{2,5}\b", re.I)
    _DOC_PLACEHOLDER_RE = re.compile(r"«документ»\s*(?=«)|«документ»_\s*")

    def sanitize(self, content: str, context: ContextParams) -> str:
        file_label = (
            f"«{context.uploaded_file_name}»"
            if context.uploaded_file_name
            else "«загруженный файл»"
        )
        lines = content.split("\n")
        return "\n".join(
            line if line.strip().startswith("|")
            else self._sanitize_line(line, context, file_label)
            for line in lines
        )

    def _sanitize_line(
        self,
        line: str,
        context: ContextParams,
        file_label: str,
    ) -> str:
        line = self._FS_PATH_RE.sub(file_label, line)
        line = self._HASH_FILENAME_RE.sub(file_label, line)
        line = self._SHORT_HASH_RE.sub(file_label, line)

        if context.file_path and _is_valid_uuid(str(context.file_path).strip()):
            line = line.replace(str(context.file_path).strip(), file_label)

        if context.document_id and _is_valid_uuid(context.document_id):
            line = line.replace(context.document_id, "«текущего документа»")

        line = self._DOC_PLACEHOLDER_RE.sub("", line)
        return line
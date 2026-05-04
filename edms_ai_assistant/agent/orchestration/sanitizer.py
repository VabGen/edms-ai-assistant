"""
sanitizer.py — очистка технических данных из финального ответа агента.

Убирает из пользовательского ответа:
- UUID
- Пути файловой системы (/tmp/..., C:\\...)
- Хэш-имена файлов (uuid_md5hash.ext)
"""

from __future__ import annotations

import re

from edms_ai_assistant.agent.context import ContextParams, is_valid_uuid

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
_FS_PATH_RE = re.compile(
    r"[A-Za-z]:\\[^\s,;)'\"]{3,}|/(?:tmp|var|home|uploads)/[^\s,;)'\"]{3,}"
)
_HASH_FILENAME_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"_[0-9a-f]{32}\.[a-zA-Z]{2,5}",
    re.IGNORECASE,
)
_SHORT_HASH_RE = re.compile(r"_?[0-9a-f]{32}\.[a-zA-Z]{2,5}\b", re.IGNORECASE)


def _sanitize_line(line: str, context: ContextParams, file_label: str) -> str:
    line = _FS_PATH_RE.sub(file_label, line)
    line = _HASH_FILENAME_RE.sub(file_label, line)
    line = _SHORT_HASH_RE.sub(file_label, line)
    if context.file_path and is_valid_uuid(context.file_path.strip()):
        line = line.replace(context.file_path.strip(), file_label)
    if context.document_id and is_valid_uuid(context.document_id):
        line = line.replace(context.document_id, "«текущего документа»")
    line = re.sub(r"«документ»\s*(?=«)", "", line)
    line = re.sub(r"«документ»_\s*", "", line)
    return line


def sanitize_technical_content(content: str, context: ContextParams) -> str:
    """
    Убирает технические данные (UUID, пути, хэши) из пользовательского ответа.

    Строки таблиц (начинающиеся с |) не затрагиваются — они намеренно
    содержат колонку id для навигации по документам во фронтенде.
    """
    file_label = (
        f"«{context.uploaded_file_name}»"
        if context.uploaded_file_name
        else "«загруженный файл»"
    )
    lines = content.split("\n")
    result: list[str] = []
    for line in lines:
        if line.strip().startswith("|"):
            result.append(line)
        else:
            result.append(_sanitize_line(line, context, file_label))
    return "\n".join(result)

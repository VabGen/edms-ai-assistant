# edms_ai_assistant\utils\format_utils.py
from typing import Any
import re


def clean_dict(d: Any) -> Any:
    """Recursively remove None, empty lists, empty dicts and empty strings.

    Returns None (not an empty collection) when the top-level container
    becomes empty after cleaning, so callers can use ``or`` chaining.
    """
    if isinstance(d, dict):
        cleaned = {k: clean_dict(v) for k, v in d.items()}
        cleaned = {k: v for k, v in cleaned.items() if v not in (None, [], {}, "")}
        return cleaned or None
    if isinstance(d, list):
        cleaned_list = [clean_dict(i) for i in d]
        cleaned_list = [i for i in cleaned_list if i not in (None, [], {}, "")]
        return cleaned_list or None
    return d


def format_document_response(text_content: str) -> str:
    formatted_content = (
        text_content.replace(r"\n", "\n").replace(r"\t", "    ").replace(r"\"", '"')
    )

    junk_phrases = [
        r"Похоже, произошла ошибка при попытке извлечь содержание вложения\.",
        r"Я буду использовать другой подход для предоставления информации о документе\.",
        r"Для получения более подробного содержания файла необходимо его извлечь и проанализировать\.",
        r"Для получения более подробной информации о содержании вложения необходимо обратиться к соответствующему инструменту или сервису, который поддерживает извлечение содержимого документов\.",
        r"Для получения более подробного содержания файла необходимо использовать дополнительный инструмент 'summarize_attachment_tool_wrapped'\.",
    ]

    cleaned_content = formatted_content
    for phrase in junk_phrases:
        cleaned_content = re.sub(
            phrase, "", cleaned_content, flags=re.IGNORECASE | re.DOTALL
        ).strip()

    lines = cleaned_content.split("\n")
    filtered_lines = []

    unwanted_keywords = [
        "ID документа:",
        "ID вложения:",
        "Размер:",
        "Дата загрузки:",
        "ID:",
        "UUID",
    ]

    for line in lines:
        is_unwanted = False

        is_junk_header = False
        if line.strip().startswith("## Информация о Документе"):
            if any(phrase in line for phrase in ["Похоже", "ошибка"]):
                is_junk_header = True

        if is_junk_header:
            continue

        for keyword in unwanted_keywords:
            if line.strip().startswith(f"- **{keyword}**") or line.strip().startswith(
                f"- {keyword}"
            ):
                is_unwanted = True
                break

        if not is_unwanted and line.strip():
            filtered_lines.append(line.rstrip())

    formatted_content = "\n".join(filtered_lines)

    if not formatted_content.strip().startswith(
        "##"
    ) and not formatted_content.strip().startswith("#"):
        formatted_content = "## Информация о Документе\n\n" + formatted_content
    formatted_content = re.sub(r"\n\s*\n", "\n\n", formatted_content).strip()

    return formatted_content

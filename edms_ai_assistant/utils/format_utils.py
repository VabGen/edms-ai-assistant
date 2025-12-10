# edms_ai_assistant\utils\format_utils.py


def format_document_response(text_content: str) -> str:
    formatted_content = text_content.replace(r'\n', '\n').replace(r'\t', '    ').replace(r'\"', '"')

    lines = formatted_content.split('\n')
    filtered_lines = []

    unwanted_keywords = ["ID документа:", "ID вложения:", "Размер:", "Дата загрузки:", "ID:"]

    for line in lines:
        is_unwanted = False
        for keyword in unwanted_keywords:
            if line.strip().startswith(f"- **{keyword}**") or line.strip().startswith(f"- {keyword}"):
                is_unwanted = True
                break

        if not is_unwanted and line.strip():
            filtered_lines.append(line)

    formatted_content = '\n'.join(filtered_lines)

    if not formatted_content.strip().startswith('##') and not formatted_content.strip().startswith('#'):
        formatted_content = "## Информация о Документе\n\n" + formatted_content

    formatted_content = formatted_content.replace('\n', ' ')

    return formatted_content

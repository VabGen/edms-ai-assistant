# edms-ai-assistant
AI-Powered EDMS Task Automation Assistant

uv venv
uv sync устанавливает зависимости из pyproject.toml
uv pip install -r requirements.txt
.venv\Scripts\activate

uv pip install --upgrade setuptools
uv sync
uv sync --all-extras

black edms_ai_assistant/

uvicorn edms_ai_assistant.app:app --host 127.0.0.1 --port 8000 --reload
D:\project\edms-ai-assistant\.venv\Scripts\python.exe D:\project\edms-ai-assistant\edms_ai_assistant\app.py
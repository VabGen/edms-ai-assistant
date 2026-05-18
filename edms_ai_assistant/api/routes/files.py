# edms_ai_assistant/api/routes/files.py
"""
File upload API routes.

Endpoints:
    POST /upload-file — save an uploaded file to the temp directory for in-chat analysis
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from edms_ai_assistant.api.deps import UPLOAD_DIR
from edms_ai_assistant.model import FileUploadResponse
from edms_ai_assistant.security import extract_user_id_from_token

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Files"])


@router.post(
    "/upload-file",
    response_model=FileUploadResponse,
    summary="Upload a file for in-chat analysis",
)
async def upload_file(
    user_token: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
) -> FileUploadResponse:
    try:
        extract_user_id_from_token(user_token)
        if not file.filename:
            raise HTTPException(status_code=400, detail="Имя файла не указано")

        original_path = Path(file.filename or "file")
        suffix = original_path.suffix.lower()
        if not suffix:
            ct = file.content_type or ""
            suffix = {
                "application/pdf": ".pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                "application/msword": ".doc",
                "text/plain": ".txt",
            }.get(ct, "")

        safe_stem = re.sub(r"[^\w\-.]", "_", original_path.stem[:80])
        safe_stem = re.sub(r"_+", "_", safe_stem).strip("_")
        dest_path = UPLOAD_DIR / f"{safe_stem}{suffix}"
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(dest_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                await out_file.write(chunk)

        logger.info(
            "File uploaded",
            extra={"orig_filename": file.filename, "dest": str(dest_path)},
        )
        return FileUploadResponse(file_path=str(dest_path), file_name=file.filename)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("File upload failed", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Ошибка при сохранении файла"
        ) from exc

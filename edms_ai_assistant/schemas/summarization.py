import datetime
from enum import Enum

from pydantic import BaseModel


class SummaryFormat(str, Enum):
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    THESIS = "thesis"


class SummarizeRequest(BaseModel):
    text: str
    summary_type: SummaryFormat = SummaryFormat.EXTRACTIVE
    file_identifier: str | None = None


class SummarizationResult(BaseModel):
    status: str = "success"
    content: str
    format_used: SummaryFormat
    text_length: int = 0
    processing_time_ms: int | None = None
    chunks_processed: int = 1
    pipeline: str = "direct"
    quality_score: float | None = None
    confidence: str | None = None
    degraded: bool = False
    warnings: list[str] = []
    created_at: datetime.datetime = datetime.datetime.utcnow()

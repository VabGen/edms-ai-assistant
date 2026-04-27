import datetime
from sqlalchemy import Column, String, Text, DateTime
from edms_ai_assistant.db.database import Base


class SummarizationCache(Base):
    __tablename__ = "summarization_cache"

    __table_args__ = {"schema": "edms"}

    file_identifier = Column(String(255), primary_key=True)
    summary_type = Column(String(50), primary_key=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
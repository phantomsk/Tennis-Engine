import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, String

from tennis.db.database import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String, nullable=False, default="UPLOADED")
    completed_stages = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def stage_completed(self, stage: str) -> bool:
        return stage in (self.completed_stages or [])

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text

from tennis.db.database import Base


def _now() -> datetime:  # naive UTC: SQLite stores datetimes without a zone
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=_now)


class JobStage(Base):
    """One row per (job, stage or source). Job status is derived from these rows.

    status: QUEUED -> RUNNING -> DONE | FAILED. Sources (UPLOAD, CALIBRATION, ...)
    are written by the UI and recorded directly as DONE.
    input_hash: fingerprint of the inputs the stored output was computed from;
    a stage re-runs only when its current fingerprint differs.
    """

    __tablename__ = "job_stages"

    job_id = Column(String, ForeignKey("jobs.id"), primary_key=True)
    stage = Column(String, primary_key=True)
    status = Column(String, nullable=False, default="QUEUED")
    input_hash = Column(String)
    output_hash = Column(String)
    attempts = Column(Integer, nullable=False, default=0)
    lease_until = Column(DateTime)
    error = Column(Text)
    progress = Column(Float, nullable=False, default=0.0)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)

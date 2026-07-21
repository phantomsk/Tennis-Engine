"""In-process stage runner. Real CV/ML stages replace run_stage_stub in later phases.

The loop body here is what a distributed worker would run per SQS message —
splitting it out later means the same _handle_stage_message logic runs behind
a queue consumer instead of InProcessQueue.process_next.
"""

import json
import time
from typing import Callable

from sqlalchemy.orm import Session

from tennis.core.config import QUEUE, STORAGE
from tennis.db.models import Job

STAGES = [
    "EXTRACTING_FRAMES",
    "POSE_TRACKING",
    "BALL_TRACKING",
    "DETECTING_EVENTS",
    "BUILDING_STATES",
    "RUNNING_INFERENCE",
]

ProgressCallback = Callable[[str], None]


def create_job(db: Session, upload_bytes: bytes) -> Job:
    job = Job()
    db.add(job)
    db.flush()  # assigns job.id via the column default
    STORAGE.write_upload(job.id, upload_bytes)
    db.commit()
    db.refresh(job)
    return job


def _prev_stage(stage: str) -> str | None:
    idx = STAGES.index(stage)
    return STAGES[idx - 1] if idx > 0 else None


def run_stage_stub(stage: str, prev: bytes | None) -> bytes:
    """Placeholder for real CV/ML work (added in later phases)."""
    time.sleep(0.3)
    payload = {"stage": stage, "stub": True}
    if prev is not None:
        payload["prev"] = json.loads(prev)
    if stage == "RUNNING_INFERENCE":
        payload["recommendation"] = {"shot": "forehand_cross_deep", "confidence": 0.42}
    return json.dumps(payload).encode()


def _handle_stage_message(job_id: str, stage: str, progress: ProgressCallback | None, db: Session) -> None:
    job = db.get(Job, job_id)
    if job.stage_completed(stage):
        return  # idempotent: redelivered message for already-completed stage

    prev_stage = _prev_stage(stage)
    prev = STORAGE.read_artifact(job_id, prev_stage) if prev_stage else None
    result = run_stage_stub(stage, prev)
    STORAGE.write_artifact(job_id, stage, result)

    job.completed_stages = [*job.completed_stages, stage]
    job.status = "COMPLETED" if stage == STAGES[-1] else stage
    db.commit()

    if progress:
        progress(stage)

    next_idx = STAGES.index(stage) + 1
    if next_idx < len(STAGES):
        QUEUE.enqueue(job_id, STAGES[next_idx])


def run_pipeline(job_id: str, db: Session, progress: ProgressCallback | None = None) -> None:
    job = db.get(Job, job_id)

    first_pending = next((s for s in STAGES if not job.stage_completed(s)), None)
    if first_pending is None:
        return  # already complete: idempotent no-op

    job.status = "QUEUED"
    db.commit()

    QUEUE.enqueue(job_id, first_pending)

    def handler(msg_job_id: str, msg_stage: str) -> None:
        _handle_stage_message(msg_job_id, msg_stage, progress, db)

    while QUEUE.process_next(handler):
        pass

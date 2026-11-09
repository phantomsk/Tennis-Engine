"""DAG stage runner (PLAN.md §7).

- `record_source` stores a UI-written input (upload, clip settings, calibration).
- `schedule` enqueues every stage whose deps are fresh and whose stored output
  was built from different inputs (fingerprint mismatch).
- `handle_stage` is the per-message handler a worker runs. It claims the stage
  with a conditional UPDATE, so a duplicate delivery is a no-op, then runs it,
  stores the output and calls `schedule` to fan out to newly ready stages.

A failed stage is marked FAILED with its error and is only retried by an
explicit run (`schedule(..., retry_failed=True)`), never by automatic
rescheduling.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from sqlalchemy import or_, update
from sqlalchemy.orm import Session

from tennis.core.config import QUEUE, STORAGE
from tennis.db.models import Job, JobStage
from tennis.pipeline.graph import ORDER, SOURCES, SPECS, fingerprint

log = logging.getLogger(__name__)

LEASE = timedelta(minutes=30)
ACTIVE = ("QUEUED", "RUNNING")

ProgressCallback = Callable[[str], None]


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)  # SQLite stores naive UTC


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact(name: str) -> str:
    return SOURCES[name] if name in SOURCES else SPECS[name].output


def _rows(db: Session, job_id: str) -> dict[str, JobStage]:
    return {r.stage: r for r in db.query(JobStage).filter_by(job_id=job_id)}


@dataclass
class _Context:
    db: Session
    job_id: str
    row: JobStage
    params: dict

    def read(self, name: str) -> bytes:
        return STORAGE.read_artifact(self.job_id, _artifact(name))

    def path(self, name: str):
        return STORAGE.local_path(self.job_id, _artifact(name))

    def report(self, fraction: float) -> None:
        self.row.progress = max(0.0, min(1.0, float(fraction)))
        self.db.commit()


def create_job(db: Session, upload_bytes: bytes, time_scale: float = 1.0) -> Job:
    job = Job()
    db.add(job)
    db.commit()
    record_source(db, job.id, "UPLOAD", upload_bytes)
    record_source(db, job.id, "CLIP_SETTINGS", f'{{"time_scale": {float(time_scale)}}}'.encode())
    return job


def record_source(db: Session, job_id: str, name: str, data: bytes) -> None:
    STORAGE.write_artifact(job_id, SOURCES[name], data)
    row = db.get(JobStage, (job_id, name)) or JobStage(job_id=job_id, stage=name)
    row.status = "DONE"
    row.output_hash = _sha256(data)
    row.finished_at = utcnow()
    db.add(row)
    db.commit()


def _walk(rows: dict[str, JobStage]):
    """Yield (name, row, fingerprint|None, fresh) in topological order.

    fingerprint is None when a dep is not fresh yet (stage not ready).
    """
    out = {s: r.output_hash for s, r in rows.items() if r.status == "DONE"}
    fresh = {s for s in SOURCES if s in out}
    for name in ORDER:
        spec = SPECS[name]
        row = rows.get(name)
        if not all(d in fresh for d in spec.deps):
            yield name, row, None, False
            continue
        fp = fingerprint(spec, out)
        is_fresh = row is not None and row.status == "DONE" and row.input_hash == fp
        if is_fresh:
            fresh.add(name)
        yield name, row, fp, is_fresh


def schedule(db: Session, job_id: str, retry_failed: bool = False) -> list[str]:
    now = utcnow()
    rows = _rows(db, job_id)
    to_enqueue = []
    for name, row, fp, is_fresh in _walk(rows):
        if fp is None or is_fresh:
            continue
        if row is not None:
            if row.status in ACTIVE and row.lease_until is not None and row.lease_until > now:
                continue  # already queued/running and its lease is live
            if row.status == "FAILED" and not retry_failed:
                continue
        else:
            row = JobStage(job_id=job_id, stage=name)
            db.add(row)
        row.status = "QUEUED"
        row.lease_until = now + LEASE
        row.progress = 0.0
        row.error = None
        to_enqueue.append(name)
    db.commit()  # commit before enqueueing so the worker sees the QUEUED rows
    for name in to_enqueue:
        QUEUE.enqueue(job_id, name)
    return to_enqueue


def handle_stage(db: Session, job_id: str, stage: str) -> bool:
    """Run one stage. Returns True if it actually ran; raises if it failed."""
    spec = SPECS[stage]
    now = utcnow()
    claimed = db.execute(
        update(JobStage)
        .where(
            JobStage.job_id == job_id,
            JobStage.stage == stage,
            or_(JobStage.status.in_(("QUEUED", "DONE", "FAILED")), JobStage.lease_until < now),
        )
        .values(status="RUNNING", lease_until=now + LEASE, attempts=JobStage.attempts + 1,
                started_at=now, progress=0.0)
    ).rowcount
    db.commit()
    if not claimed:
        return False  # another worker holds it
    db.expire_all()

    rows = _rows(db, job_id)
    row = rows[stage]
    out = {s: r.output_hash for s, r in rows.items() if r.status == "DONE"}
    if not all(d in out for d in spec.deps):
        row.status, row.lease_until = "QUEUED", None  # not ready; schedule() re-enqueues it
        db.commit()
        return False
    fp = fingerprint(spec, out)
    if row.input_hash == fp and row.output_hash:
        row.status, row.lease_until = "DONE", None  # duplicate delivery of finished work
        db.commit()
        return False

    try:
        data = spec.run(_Context(db, job_id, row, spec.params))
    except Exception as exc:
        row.status = "FAILED"
        row.error = f"{type(exc).__name__}: {exc}"
        row.lease_until = None
        row.finished_at = utcnow()
        db.commit()
        raise

    STORAGE.write_artifact(job_id, spec.output, data)
    row.status = "DONE"
    row.input_hash = fp  # if a dep changed mid-run, schedule() sees the mismatch and re-runs
    row.output_hash = _sha256(data)
    row.progress = 1.0
    row.lease_until = None
    row.error = None
    row.finished_at = utcnow()
    db.commit()
    schedule(db, job_id)
    return True


def job_status(db: Session, job_id: str) -> str:
    rows = _rows(db, job_id)
    stage_rows = [rows[n] for n in ORDER if n in rows]
    statuses = {r.status for r in stage_rows}
    if "FAILED" in statuses:
        return "FAILED"
    if "RUNNING" in statuses:
        return "RUNNING"
    if "QUEUED" in statuses:
        return "QUEUED"
    if not stage_rows:
        return "UPLOADED"
    walked = list(_walk(rows))
    if all(is_fresh for *_, is_fresh in walked):
        return "COMPLETED"
    if all(row is not None and row.status == "DONE" for _, row, _, _ in walked):
        return "OUTDATED"  # inputs changed since the last run
    return "WAITING"  # e.g. waiting for calibration


def stage_rows(db: Session, job_id: str) -> list[JobStage]:
    rows = _rows(db, job_id)
    return [rows[n] for n in ORDER if n in rows]


def run_pipeline(job_id: str, db: Session, progress: ProgressCallback | None = None) -> None:
    """Synchronous drain of the in-process queue (explicit Run/Retry)."""
    schedule(db, job_id, retry_failed=True)

    def handler(msg_job_id: str, msg_stage: str) -> None:
        try:
            ran = handle_stage(db, msg_job_id, msg_stage)
        except Exception:
            log.exception("Stage %s failed for job %s", msg_stage, msg_job_id)
            return
        if ran and progress:
            progress(msg_stage)

    while QUEUE.process_next(handler):
        pass

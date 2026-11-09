import dataclasses
import json
from datetime import timedelta

import pytest

from tennis.core.config import QUEUE
from tennis.db.database import SessionLocal, init_db
from tennis.db.models import JobStage
from tennis.pipeline import runner
from tennis.worker import process_next, reap_expired

CALIBRATION = json.dumps({"homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}).encode()


@pytest.fixture
def db():
    init_db()
    QUEUE.clear()
    session = SessionLocal()
    yield session
    session.close()
    QUEUE.clear()


def _drain(db):
    while process_next(db):
        pass


def _calibrated_job(db, video_bytes):
    job = runner.create_job(db, video_bytes)
    runner.record_source(db, job.id, "CALIBRATION", CALIBRATION)
    return job


def test_starting_a_job_only_enqueues_and_the_worker_runs_it(db, video_bytes):
    job = _calibrated_job(db, video_bytes)

    runner.start_job(db, job.id)
    assert runner.job_status(db, job.id) == "QUEUED"  # the UI returns immediately

    _drain(db)

    assert runner.job_status(db, job.id) == "COMPLETED"


def test_a_failing_stage_is_retried_then_marked_failed(db, video_bytes, monkeypatch):
    job = _calibrated_job(db, video_bytes)
    real = runner.SPECS["BALL_TRACKING"]

    def boom(ctx):
        raise RuntimeError("tracker exploded")

    monkeypatch.setitem(runner.SPECS, "BALL_TRACKING", dataclasses.replace(real, run=boom))
    runner.start_job(db, job.id)
    _drain(db)

    row = db.get(JobStage, (job.id, "BALL_TRACKING"))
    assert row.attempts == runner.MAX_ATTEMPTS
    assert "tracker exploded" in row.error
    assert runner.job_status(db, job.id) == "FAILED"

    monkeypatch.setitem(runner.SPECS, "BALL_TRACKING", real)
    runner.start_job(db, job.id)  # the UI's Retry button
    _drain(db)

    assert runner.job_status(db, job.id) == "COMPLETED"


def test_a_message_taken_by_a_crashed_worker_is_redelivered(db, video_bytes):
    job = _calibrated_job(db, video_bytes)
    runner.start_job(db, job.id)

    crashed = QUEUE.receive(visibility_timeout_s=0)  # taken, never acked
    assert crashed.stage == "PROBE_VIDEO"

    _drain(db)

    assert runner.job_status(db, job.id) == "COMPLETED"


def test_reaper_restarts_a_stage_whose_worker_died_mid_run(db, video_bytes):
    job = _calibrated_job(db, video_bytes)
    runner.start_job(db, job.id)
    QUEUE.clear()  # the message is gone (acked by a worker that then died)
    row = db.get(JobStage, (job.id, "PROBE_VIDEO"))
    row.status = "RUNNING"
    row.lease_until = runner.utcnow() - timedelta(seconds=1)  # lease ran out
    db.commit()

    assert reap_expired(db) == [job.id]
    _drain(db)

    assert runner.job_status(db, job.id) == "COMPLETED"

import dataclasses
import json
from collections import Counter
from datetime import timedelta

import pytest

from tennis.core.config import QUEUE, STORAGE
from tennis.db.database import SessionLocal, init_db
from tennis.db.models import JobStage
from tennis.pipeline import runner
from tennis.pipeline.graph import ORDER, SPECS


@pytest.fixture
def db():
    init_db()
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def run_counts(monkeypatch):
    """Count how many times each stage's run function actually executes."""
    counts = Counter()
    patched = {}
    for name, spec in SPECS.items():
        def counting(ctx, _run=spec.run, _name=name):
            counts[_name] += 1
            return _run(ctx)
        patched[name] = dataclasses.replace(spec, run=counting)
    monkeypatch.setattr(runner, "SPECS", patched)
    return counts


CALIBRATION_V1 = json.dumps({"homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "v": 1}).encode()
CALIBRATION_V2 = json.dumps({"homography": [[2, 0, 0], [0, 2, 0], [0, 0, 1]], "v": 2}).encode()


def _new_calibrated_job(db, video_bytes):
    job = runner.create_job(db, video_bytes)
    runner.record_source(db, job.id, "CALIBRATION", CALIBRATION_V1)
    return job


def test_pipeline_runs_every_stage_to_completion(db, video_bytes, run_counts):
    job = _new_calibrated_job(db, video_bytes)
    assert runner.job_status(db, job.id) == "UPLOADED"

    runner.run_pipeline(job.id, db)

    assert runner.job_status(db, job.id) == "COMPLETED"
    assert set(run_counts) == set(ORDER)
    assert all(n == 1 for n in run_counts.values())
    rec = json.loads(STORAGE.read_artifact(job.id, "recommendations.json"))
    assert "recommendation" in rec


def test_probe_video_writes_real_metadata(db, video_bytes):
    job = _new_calibrated_job(db, video_bytes)

    runner.run_pipeline(job.id, db)

    probe = json.loads(STORAGE.read_artifact(job.id, "probe.json"))
    assert probe["n_frames"] == 5
    assert probe["fps"] == pytest.approx(10.0)
    assert probe["time_scale"] == 1.0


def test_rerunning_a_completed_job_runs_nothing(db, video_bytes, run_counts):
    job = _new_calibrated_job(db, video_bytes)
    runner.run_pipeline(job.id, db)
    run_counts.clear()

    runner.run_pipeline(job.id, db)

    assert runner.job_status(db, job.id) == "COMPLETED"
    assert sum(run_counts.values()) == 0


def test_recalibrating_reruns_player_tracking_downstream_but_not_ball_tracking(db, video_bytes, run_counts):
    job = _new_calibrated_job(db, video_bytes)
    runner.run_pipeline(job.id, db)
    run_counts.clear()

    runner.record_source(db, job.id, "CALIBRATION", CALIBRATION_V2)
    runner.run_pipeline(job.id, db)

    assert run_counts["PLAYER_TRACKING"] == 1
    assert run_counts["BALL_TRACKING"] == 0
    assert run_counts["PROBE_VIDEO"] == 0
    assert runner.job_status(db, job.id) == "COMPLETED"


def test_status_reports_outdated_after_recalibration_until_rerun(db, video_bytes):
    job = _new_calibrated_job(db, video_bytes)
    runner.run_pipeline(job.id, db)

    runner.record_source(db, job.id, "CALIBRATION", CALIBRATION_V2)

    assert runner.job_status(db, job.id) == "OUTDATED"


def test_rewriting_identical_calibration_reruns_nothing(db, video_bytes, run_counts):
    job = _new_calibrated_job(db, video_bytes)
    runner.run_pipeline(job.id, db)
    run_counts.clear()

    runner.record_source(db, job.id, "CALIBRATION", CALIBRATION_V1)
    runner.run_pipeline(job.id, db)

    assert sum(run_counts.values()) == 0


def test_a_failing_stage_marks_the_job_failed_and_retry_resumes(db, video_bytes, monkeypatch):
    job = _new_calibrated_job(db, video_bytes)
    real = runner.SPECS["BALL_TRACKING"]

    def boom(ctx):
        raise RuntimeError("tracker exploded")

    monkeypatch.setitem(runner.SPECS, "BALL_TRACKING", dataclasses.replace(real, run=boom))
    runner.run_pipeline(job.id, db)  # records the failure; does not raise or loop

    assert runner.job_status(db, job.id) == "FAILED"
    row = db.get(JobStage, (job.id, "BALL_TRACKING"))
    assert "tracker exploded" in row.error
    assert row.attempts == runner.MAX_ATTEMPTS  # retried once, then left FAILED

    monkeypatch.setitem(runner.SPECS, "BALL_TRACKING", real)
    runner.run_pipeline(job.id, db)  # explicit Run/Retry

    assert runner.job_status(db, job.id) == "COMPLETED"


def test_duplicate_delivery_of_a_finished_stage_is_a_no_op(db, video_bytes, run_counts):
    job = _new_calibrated_job(db, video_bytes)
    runner.run_pipeline(job.id, db)
    run_counts.clear()

    runner.handle_stage(db, job.id, "BALL_TRACKING")

    assert run_counts["BALL_TRACKING"] == 0
    assert runner.job_status(db, job.id) == "COMPLETED"


def test_schedule_reenqueues_a_stage_whose_lease_expired(db, video_bytes):
    job = _new_calibrated_job(db, video_bytes)
    enqueued = runner.schedule(db, job.id)
    assert enqueued == ["PROBE_VIDEO"]
    QUEUE.clear()  # simulate the message being lost (e.g. crash after commit)

    row = db.get(JobStage, (job.id, "PROBE_VIDEO"))
    row.lease_until = runner.utcnow() - timedelta(seconds=1)
    db.commit()

    assert runner.schedule(db, job.id) == ["PROBE_VIDEO"]
    QUEUE.clear()


def test_stages_wait_for_calibration_before_running(db, video_bytes, run_counts):
    job = runner.create_job(db, video_bytes)

    runner.run_pipeline(job.id, db)

    assert run_counts["PROBE_VIDEO"] == 1
    assert run_counts["BALL_TRACKING"] == 1
    assert run_counts["PLAYER_TRACKING"] == 0
    assert runner.job_status(db, job.id) != "COMPLETED"

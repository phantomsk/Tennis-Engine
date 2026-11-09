import dataclasses
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tennis.core.config import QUEUE
from tennis.db.database import SessionLocal, init_db
from tennis.pipeline import runner
from tennis.worker import process_next

APP = str(Path(__file__).resolve().parents[1] / "app.py")
CALIBRATION = json.dumps(
    {"homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "image_points": [[0, 0]] * 8}
).encode()


@pytest.fixture
def db():
    init_db()
    QUEUE.clear()
    session = SessionLocal()
    yield session
    session.close()
    QUEUE.clear()


def _finished_job(db, video_bytes):
    job = runner.create_job(db, video_bytes)
    runner.record_source(db, job.id, "CALIBRATION", CALIBRATION)
    runner.start_job(db, job.id)
    while process_next(db):
        pass
    return job


def _all_text(at) -> str:
    parts = [m.value for m in at.markdown] + [s.value for s in at.success] + [e.value for e in at.error]
    parts += [str(j.value) for j in at.json]
    return "\n".join(parts)


def test_app_starts_without_errors_and_prompts_for_upload():
    at = AppTest.from_file(APP, default_timeout=30).run()

    assert not at.exception
    assert at.title[0].value == "Tennis Strategy Engine"


def test_a_job_in_the_url_reopens_with_its_results_after_a_refresh(db, video_bytes):
    job = _finished_job(db, video_bytes)

    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["job"] = job.id
    at.run()

    assert not at.exception
    text = _all_text(at)
    assert "COMPLETED" in text
    assert "CROSS_DEEP" in text


def test_a_failed_job_shows_its_error_and_a_retry_button(db, video_bytes, monkeypatch):
    real = runner.SPECS["BALL_TRACKING"]

    def boom(ctx):
        raise RuntimeError("tracker exploded")

    monkeypatch.setitem(runner.SPECS, "BALL_TRACKING", dataclasses.replace(real, run=boom))
    job = _finished_job(db, video_bytes)

    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["job"] = job.id
    at.run()

    assert not at.exception
    assert "tracker exploded" in _all_text(at)
    assert any(b.label == "Retry" for b in at.button)

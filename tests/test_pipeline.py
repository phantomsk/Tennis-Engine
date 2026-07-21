from tennis.db.database import SessionLocal, init_db
from tennis.pipeline.runner import STAGES, create_job, run_pipeline


def test_pipeline_runs_all_stages_to_completion():
    init_db()
    db = SessionLocal()
    try:
        job = create_job(db, b"fake video bytes")
        assert job.status == "UPLOADED"

        seen_stages = []
        run_pipeline(job.id, db, progress=seen_stages.append)

        db.refresh(job)
        assert job.status == "COMPLETED"
        assert seen_stages == STAGES
        assert job.completed_stages == STAGES
    finally:
        db.close()


def test_pipeline_is_idempotent_on_rerun():
    init_db()
    db = SessionLocal()
    try:
        job = create_job(db, b"fake video bytes")
        run_pipeline(job.id, db)
        db.refresh(job)
        assert job.status == "COMPLETED"

        # re-running a completed job must be a no-op, not reprocess stages
        run_pipeline(job.id, db)
        db.refresh(job)
        assert job.status == "COMPLETED"
        assert job.completed_stages == STAGES
    finally:
        db.close()

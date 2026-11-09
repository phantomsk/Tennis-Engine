"""Pipeline worker: `python -m tennis.worker`.

Receives stage messages from the queue and runs them, outside Streamlit's
script thread. Streamlit auto-starts one of these per server (see app.py
`ensure_worker`); set TENNIS_WORKER=external to run it yourself instead. At
scale-up this same loop runs in an ECS task against SqsQueue.
"""

import argparse
import logging
import os
import time

from sqlalchemy import distinct
from sqlalchemy.orm import Session

from tennis.core.config import QUEUE
from tennis.db.database import SessionLocal, init_db
from tennis.db.models import JobStage
from tennis.pipeline.runner import ACTIVE, LEASE, process_message, schedule, utcnow

log = logging.getLogger("tennis.worker")

REAP_EVERY_S = 60.0


def process_next(db: Session) -> bool:
    """Handle at most one message. Returns False when the queue is empty."""
    msg = QUEUE.receive(int(LEASE.total_seconds()))
    if msg is None:
        return False
    process_message(db, msg)
    return True


def reap_expired(db: Session) -> list[str]:
    """Re-schedule jobs with a stage whose lease ran out (its worker died)."""
    job_ids = [
        job_id
        for (job_id,) in db.query(distinct(JobStage.job_id)).filter(
            JobStage.status.in_(ACTIVE), JobStage.lease_until < utcnow()
        )
    ]
    for job_id in job_ids:
        schedule(db, job_id)
    return job_ids


def run_worker(poll_interval_s: float = 1.0, parent_pid: int | None = None) -> None:
    init_db()
    last_reap = 0.0
    while True:
        if parent_pid is not None and os.getppid() != parent_pid:
            log.info("Parent process %d exited; worker stopping.", parent_pid)
            return
        db = SessionLocal()
        try:
            if time.monotonic() - last_reap > REAP_EVERY_S:
                reap_expired(db)
                last_reap = time.monotonic()
            busy = process_next(db)
        except Exception:
            log.exception("Worker loop error")
            busy = False
        finally:
            db.close()
        if not busy:
            time.sleep(poll_interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, default=None,
                        help="Exit when this process is no longer our parent (set by the app).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_worker(parent_pid=args.parent_pid)


if __name__ == "__main__":
    main()

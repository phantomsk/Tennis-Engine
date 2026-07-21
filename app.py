import json

import streamlit as st

from tennis.core.config import STORAGE
from tennis.db.database import SessionLocal, init_db
from tennis.pipeline.runner import STAGES, create_job, run_pipeline

st.set_page_config(page_title="Tennis Strategy Engine", layout="centered")
init_db()

st.title("Tennis Strategy Engine")
st.caption("Phase 1 skeleton — pipeline stages are stubbed, no real CV/ML yet.")

uploaded = st.file_uploader("Upload a rally clip (10-30s)", type=["mp4", "mov"])

if uploaded is not None and st.button("Run analysis"):
    db = SessionLocal()
    try:
        job = create_job(db, uploaded.getvalue())

        progress_bar = st.progress(0.0, text="Starting...")

        def on_progress(stage: str) -> None:
            done = STAGES.index(stage) + 1
            progress_bar.progress(done / len(STAGES), text=f"{stage} ({done}/{len(STAGES)})")

        run_pipeline(job.id, db, progress=on_progress)
        db.refresh(job)

        st.success(f"Job `{job.id}` — status: **{job.status}**")

        if job.status == "COMPLETED":
            recommendation = json.loads(STORAGE.read_artifact(job.id, "RUNNING_INFERENCE"))
            st.subheader("Recommendation (stub)")
            st.json(recommendation["recommendation"])
    finally:
        db.close()

import json

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from tennis.core.config import STORAGE
from tennis.cv.court_calibration import CORNER_LABELS, compute_homography, serialize_calibration
from tennis.cv.video_loader import extract_first_frame
from tennis.db.database import SessionLocal, init_db
from tennis.db.models import Job
from tennis.pipeline.runner import STAGES, create_job, run_pipeline

st.set_page_config(page_title="Tennis Strategy Engine", layout="centered")
init_db()

st.title("Tennis Strategy Engine")
st.caption("Phase 1-3 skeleton — court calibration is real; downstream pipeline stages are still stubbed.")

st.session_state.setdefault("job_id", None)
st.session_state.setdefault("upload_file_id", None)
st.session_state.setdefault("calib_points", [])

db = SessionLocal()


@st.cache_data(show_spinner=False)
def _cached_first_frame(video_bytes: bytes):
    return extract_first_frame(video_bytes)


def _draw_markers(frame_rgb, points) -> Image.Image:
    image = Image.fromarray(frame_rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    radius = 8
    for i, (x, y) in enumerate(points, start=1):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="lime", width=3)
        draw.text((x + radius + 2, y - radius), str(i), fill="lime")
    return image


uploaded = st.file_uploader("Upload a rally clip (10-30s)", type=["mp4", "mov"])

if uploaded is not None:
    if uploaded.file_id != st.session_state.upload_file_id:
        job = create_job(db, uploaded.getvalue())
        st.session_state.job_id = job.id
        st.session_state.upload_file_id = uploaded.file_id
        st.session_state.calib_points = []

    job_id = st.session_state.job_id
    calib_points = st.session_state.calib_points

    st.subheader("Court calibration")

    if len(calib_points) < 4:
        st.info(f"Click the **{CORNER_LABELS[len(calib_points)]}**.")
        frame_rgb = _cached_first_frame(uploaded.getvalue())
        marked_image = _draw_markers(frame_rgb, calib_points)
        value = streamlit_image_coordinates(
            marked_image,
            width=700,
            key=f"corner_{job_id}_{len(calib_points)}",
        )
        if value is not None:
            scale = frame_rgb.shape[1] / 700
            st.session_state.calib_points.append((value["x"] * scale, value["y"] * scale))
            st.rerun()
    else:
        homography = compute_homography(calib_points)
        STORAGE.write_artifact(job_id, "CALIBRATION", serialize_calibration(calib_points, homography))

        frame_rgb = _cached_first_frame(uploaded.getvalue())
        st.image(_draw_markers(frame_rgb, calib_points), width=700)
        st.success("Court calibrated.")
        if st.button("Recalibrate"):
            st.session_state.calib_points = []
            st.rerun()

    if len(calib_points) == 4 and st.button("Run analysis"):
        progress_bar = st.progress(0.0, text="Starting...")

        def on_progress(stage: str) -> None:
            done = STAGES.index(stage) + 1
            progress_bar.progress(done / len(STAGES), text=f"{stage} ({done}/{len(STAGES)})")

        run_pipeline(job_id, db, progress=on_progress)
        job = db.get(Job, job_id)

        st.success(f"Job `{job.id}` — status: **{job.status}**")

        if job.status == "COMPLETED":
            recommendation = json.loads(STORAGE.read_artifact(job.id, "RUNNING_INFERENCE"))
            st.subheader("Recommendation (stub)")
            st.json(recommendation["recommendation"])

db.close()

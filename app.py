import json

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from tennis.core.config import STORAGE
from tennis.cv.court_calibration import (
    POINT_LABELS,
    CalibrationError,
    compute_homography,
    court_lines_px,
    serialize_calibration,
    validate_calibration,
)
from tennis.cv.video_loader import extract_first_frame
from tennis.db.database import SessionLocal, init_db
from tennis.pipeline.graph import ORDER
from tennis.pipeline.runner import create_job, job_status, record_source, run_pipeline, stage_rows

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


def _draw_markers(frame_rgb, points, lines=None, bad_points=()) -> Image.Image:
    image = Image.fromarray(frame_rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    for (a, b) in (lines or {}).values():
        draw.line((a, b), fill="cyan", width=2)
    radius = 8
    for i, (x, y) in enumerate(points):
        color = "red" if i in bad_points else "lime"
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
        draw.text((x + radius + 2, y - radius), str(i + 1), fill=color)
    return image


uploaded = st.file_uploader("Upload a rally clip (10-30s)", type=["mp4", "mov"])

if uploaded is not None:
    if uploaded.file_id != st.session_state.upload_file_id:
        job = create_job(db, uploaded.getvalue())
        st.session_state.job_id = job.id
        st.session_state.upload_file_id = uploaded.file_id
        st.session_state.calib_points = []
        st.session_state.saved_calibration = None
        st.session_state.saved_time_scale = 1.0

    job_id = st.session_state.job_id
    calib_points = st.session_state.calib_points

    slowmo = st.selectbox(
        "Recorded in slow motion?",
        ["No (normal speed)", "2x slow-mo", "4x slow-mo", "8x slow-mo"],
        key=f"slowmo_{job_id}",
        help="Slow-motion clips are converted back to real time before analysis.",
    )
    time_scale = {"No": 1.0, "2x": 2.0, "4x": 4.0, "8x": 8.0}[slowmo.split(" ")[0]]
    if time_scale != st.session_state.get("saved_time_scale", 1.0):
        record_source(db, job_id, "CLIP_SETTINGS", json.dumps({"time_scale": time_scale}).encode())
        st.session_state.saved_time_scale = time_scale

    st.subheader("Court calibration")

    calibration_ok = False
    frame_rgb = _cached_first_frame(uploaded.getvalue())

    if len(calib_points) < len(POINT_LABELS):
        st.info(f"Point {len(calib_points) + 1}/{len(POINT_LABELS)}: click the **{POINT_LABELS[len(calib_points)]}**.")
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
        if calib_points and st.button("Undo last point"):
            st.session_state.calib_points = calib_points[:-1]
            st.rerun()
    else:
        try:
            homography = compute_homography(calib_points)
        except CalibrationError as exc:
            st.error(str(exc))
            homography = None

        if homography is not None:
            check = validate_calibration(calib_points, homography)
            st.image(
                _draw_markers(frame_rgb, calib_points, court_lines_px(homography), check.bad_points),
                width=700,
                caption="Cyan lines are projected from your clicks -- they should sit on the real court lines.",
            )
            st.caption(f"Reprojection error: {check.rms_px:.2f}px RMS, sideline bow {check.sideline_bow_px:.1f}px")
            for issue in check.issues:
                st.error(issue)

            col_near, col_far, col_me = st.columns(3)
            near_hand = col_near.selectbox("Near player hand", ["right", "left"], key=f"near_hand_{job_id}")
            far_hand = col_far.selectbox("Far player hand", ["right", "left"], key=f"far_hand_{job_id}")
            user_side = col_me.selectbox("Which player are you?", ["near", "far"], key=f"user_side_{job_id}")

            if check.ok:
                calibration_ok = True
                data = serialize_calibration(
                    calib_points, homography, check, {"near": near_hand, "far": far_hand}, user_side
                )
                # Write only when something changed: rewriting identical bytes on
                # every rerun would churn the artifact for no reason.
                if st.session_state.get("saved_calibration") != data:
                    record_source(db, job_id, "CALIBRATION", data)
                    st.session_state.saved_calibration = data
                st.success("Court calibrated.")

        if st.button("Recalibrate"):
            st.session_state.calib_points = []
            st.rerun()

    if calibration_ok and st.button("Run analysis"):
        progress_bar = st.progress(0.0, text="Starting...")

        def on_progress(stage: str) -> None:
            done = ORDER.index(stage) + 1
            progress_bar.progress(done / len(ORDER), text=f"{stage} ({done}/{len(ORDER)})")

        run_pipeline(job_id, db, progress=on_progress)
        status = job_status(db, job_id)

        st.success(f"Job `{job_id}` — status: **{status}**")
        for row in stage_rows(db, job_id):
            if row.status == "FAILED":
                st.error(f"{row.stage} failed: {row.error}")

        if status == "COMPLETED":
            recommendation = json.loads(STORAGE.read_artifact(job_id, "recommendations.json"))
            st.subheader("Recommendation (stub)")
            st.json(recommendation["recommendation"])

db.close()

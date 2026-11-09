"""Streamlit UI. It never runs pipeline work itself: it enqueues stages and polls
job status, while a worker subprocess (tennis.worker) does the processing.
The job id lives in the URL (?job=...), so a refresh or a return visit reopens
the same job."""

import json
import os
import subprocess
import sys

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
from tennis.db.models import Job
from tennis.pipeline.graph import ORDER
from tennis.pipeline.runner import create_job, job_status, record_source, stage_rows, start_job

ACTIVE_STATUSES = ("QUEUED", "RUNNING")
SLOWMO_OPTIONS = {"No (normal speed)": 1.0, "2x slow-mo": 2.0, "4x slow-mo": 4.0, "8x slow-mo": 8.0}
STAGE_ICONS = {"DONE": "✅", "RUNNING": "⏳", "QUEUED": "🕒", "FAILED": "❌"}

st.set_page_config(page_title="Tennis Strategy Engine", layout="centered")
init_db()


@st.cache_resource
def ensure_worker():
    """Start one pipeline worker per Streamlit server (TENNIS_WORKER=external disables)."""
    if os.environ.get("TENNIS_WORKER") == "external":
        return None
    return subprocess.Popen([sys.executable, "-m", "tennis.worker", "--parent-pid", str(os.getpid())])


ensure_worker()

st.title("Tennis Strategy Engine")
st.caption("Calibration and video probing are real; the CV/ML stages after them are still stubs.")

st.session_state.setdefault("upload_file_id", None)
st.session_state.setdefault("loaded_job", None)

db = SessionLocal()


@st.cache_data(show_spinner=False)
def _first_frame(job_id: str):
    return extract_first_frame(STORAGE.read_artifact(job_id, "upload.mp4"))


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


def _load_job_into_session(job_id: str) -> None:
    """Restore saved calibration and clip settings, e.g. after a page refresh."""
    st.session_state.loaded_job = job_id
    try:
        saved = STORAGE.read_artifact(job_id, "calibration.json")
        cal = json.loads(saved)
        st.session_state.calib_points = [tuple(p) for p in cal.get("image_points", [])]
        st.session_state.saved_calibration = saved
        st.session_state.saved_hands = cal.get("handedness", {})
        st.session_state.saved_user_side = cal.get("user_side", "near")
    except FileNotFoundError:
        st.session_state.calib_points = []
        st.session_state.saved_calibration = None
        st.session_state.saved_hands = {}
        st.session_state.saved_user_side = "near"
    try:
        st.session_state.saved_time_scale = json.loads(STORAGE.read_artifact(job_id, "clip.json"))["time_scale"]
    except FileNotFoundError:
        st.session_state.saved_time_scale = 1.0


def _open_job(job_id: str | None) -> None:
    if job_id:
        st.query_params["job"] = job_id
    else:
        st.query_params.clear()
    st.session_state.loaded_job = None


uploaded = st.file_uploader("Upload a rally clip (10-30s of real time)", type=["mp4", "mov"])
if uploaded is not None and uploaded.file_id != st.session_state.upload_file_id:
    st.session_state.upload_file_id = uploaded.file_id
    _open_job(create_job(db, uploaded.getvalue()).id)

with st.sidebar:  # after upload handling, so a just-created job is listed immediately
    st.header("Recent jobs")
    if st.button("New job"):
        _open_job(None)
        st.rerun()
    for recent in db.query(Job).order_by(Job.created_at.desc()).limit(10):
        label = f"{recent.created_at:%b %d %H:%M} · {job_status(db, recent.id)}"
        if st.button(label, key=f"open_{recent.id}"):
            _open_job(recent.id)
            st.rerun()

job_id = st.query_params.get("job")
if job_id and db.get(Job, job_id) is None:
    st.error(f"Unknown job `{job_id}`.")
    job_id = None


def _calibration_section(job_id: str) -> bool:
    """Render settings + calibration; returns True when a valid calibration is saved."""
    labels = list(SLOWMO_OPTIONS)
    saved_scale = st.session_state.saved_time_scale
    slowmo = st.selectbox(
        "Recorded in slow motion?", labels,
        index=list(SLOWMO_OPTIONS.values()).index(saved_scale) if saved_scale in SLOWMO_OPTIONS.values() else 0,
        key=f"slowmo_{job_id}",
        help="Slow-motion clips are converted back to real time before analysis.",
    )
    if SLOWMO_OPTIONS[slowmo] != saved_scale:
        record_source(db, job_id, "CLIP_SETTINGS", json.dumps({"time_scale": SLOWMO_OPTIONS[slowmo]}).encode())
        st.session_state.saved_time_scale = SLOWMO_OPTIONS[slowmo]

    st.subheader("Court calibration")
    calib_points = st.session_state.calib_points
    frame_rgb = _first_frame(job_id)

    if len(calib_points) < len(POINT_LABELS):
        st.info(f"Point {len(calib_points) + 1}/{len(POINT_LABELS)} — click: **{POINT_LABELS[len(calib_points)]}**")
        value = streamlit_image_coordinates(
            _draw_markers(frame_rgb, calib_points), width=700, key=f"corner_{job_id}_{len(calib_points)}"
        )
        if value is not None:
            scale = frame_rgb.shape[1] / 700
            st.session_state.calib_points = [*calib_points, (value["x"] * scale, value["y"] * scale)]
            st.rerun()
        if calib_points and st.button("Undo last point"):
            st.session_state.calib_points = calib_points[:-1]
            st.rerun()
        return False

    ok = False
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

        hands = st.session_state.saved_hands
        col_near, col_far, col_me = st.columns(3)
        near_hand = col_near.selectbox("Near player hand", ["right", "left"],
                                       index=["right", "left"].index(hands.get("near", "right")),
                                       key=f"near_hand_{job_id}")
        far_hand = col_far.selectbox("Far player hand", ["right", "left"],
                                     index=["right", "left"].index(hands.get("far", "right")),
                                     key=f"far_hand_{job_id}")
        user_side = col_me.selectbox("Which player are you?", ["near", "far"],
                                     index=["near", "far"].index(st.session_state.saved_user_side),
                                     key=f"user_side_{job_id}")

        if check.ok:
            ok = True
            data = serialize_calibration(calib_points, homography, check, {"near": near_hand, "far": far_hand},
                                         user_side)
            # Write only when something changed, so reruns don't invalidate stages.
            if st.session_state.saved_calibration != data:
                record_source(db, job_id, "CALIBRATION", data)
                st.session_state.saved_calibration = data
            st.success("Court calibrated.")

    if st.button("Recalibrate"):
        st.session_state.calib_points = []
        st.rerun()
    return ok


def _status_panel(job_id: str, was_active: bool) -> None:
    panel_db = SessionLocal()
    try:
        status = job_status(panel_db, job_id)
        st.markdown(f"**Status:** {status}")
        rows = {r.stage: r for r in stage_rows(panel_db, job_id)}
        for name in ORDER:
            row = rows.get(name)
            if row is None:
                st.markdown(f"▫️ `{name}`")
            elif row.status == "RUNNING":
                st.progress(row.progress, text=f"⏳ {name}")
            else:
                st.markdown(f"{STAGE_ICONS.get(row.status, '▫️')} `{name}`")

        if status == "FAILED":
            for row in rows.values():
                if row.status == "FAILED":
                    st.error(f"{row.stage} failed: {row.error}")
            if st.button("Retry"):
                start_job(panel_db, job_id)
                st.rerun()
        elif status == "COMPLETED":
            recommendation = json.loads(STORAGE.read_artifact(job_id, "recommendations.json"))
            st.subheader("Recommendation (stub)")
            st.json(recommendation["recommendation"])

        if was_active and status not in ACTIVE_STATUSES:
            st.rerun()  # job finished: refresh the whole page once and stop polling
    finally:
        panel_db.close()


if job_id is None:
    st.info("Upload a clip, or open a recent job from the sidebar.")
else:
    if st.session_state.loaded_job != job_id:
        _load_job_into_session(job_id)

    st.caption(f"Job `{job_id}` — bookmark this page to come back to it.")
    calibration_ok = _calibration_section(job_id)

    status = job_status(db, job_id)
    active = status in ACTIVE_STATUSES
    if calibration_ok and not active:
        label = "Re-run analysis" if status in ("COMPLETED", "OUTDATED") else "Run analysis"
        if st.button(label, disabled=status == "COMPLETED"):
            start_job(db, job_id)
            st.rerun()

    st.subheader("Analysis")
    st.fragment(run_every=2 if active else None)(_status_panel)(job_id, active)

db.close()

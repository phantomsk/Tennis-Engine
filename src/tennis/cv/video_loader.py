"""Video I/O (OpenCV)."""

import tempfile

import cv2
import numpy as np


def extract_first_frame(video_bytes: bytes) -> np.ndarray:
    """Decode the first frame of a video as an RGB array.

    cv2.VideoCapture only reads from a filesystem path, so the bytes are
    materialized to a temp file for the duration of the read.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        ok, frame_bgr = cap.read()
        cap.release()

    if not ok:
        raise ValueError("Could not decode a frame from the uploaded video")

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


HIGH_FRAME_RATE_FPS = 100.0
VFR_REL_STD = 0.05


def probe_video(path, time_scale: float = 1.0) -> dict:
    """Video metadata plus a per-frame timestamp table; no frames are kept.

    `time_scale` is how many seconds of video one real second plays over
    (1 = normal, 4 = 4x slow motion). Downstream stages must use the `real_*`
    fields so speed limits, flight times and TrackNet's motion cue hold for
    slow-motion clips too.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)

    timestamps_ms = []
    while cap.grab():  # grab() skips decoding: cheap
        timestamps_ms.append(round(float(cap.get(cv2.CAP_PROP_POS_MSEC)), 3))
    cap.release()

    if not timestamps_ms:
        raise ValueError(f"No decodable frames in: {path}")

    deltas = np.diff(timestamps_ms)
    vfr = bool(deltas.size > 1 and deltas.mean() > 0 and deltas.std() / deltas.mean() > VFR_REL_STD)
    n = len(timestamps_ms)
    duration_s = n / fps if fps else (timestamps_ms[-1] / 1000.0)

    return {
        "fps": fps,
        "n_frames": n,
        "width": width,
        "height": height,
        "rotation": rotation,
        "timestamps_ms": timestamps_ms,
        "variable_frame_rate": vfr,
        "high_frame_rate": fps >= HIGH_FRAME_RATE_FPS,
        "time_scale": time_scale,
        "real_fps": fps * time_scale,
        "real_timestamps_ms": [round(t / time_scale, 3) for t in timestamps_ms],
        "real_duration_s": duration_s / time_scale,
    }

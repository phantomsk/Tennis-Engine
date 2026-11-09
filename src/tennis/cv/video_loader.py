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

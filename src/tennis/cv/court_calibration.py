"""Manual court calibration: 4 user-clicked image points -> homography to court meters.

Fixed-camera, planar-court assumption (see CLAUDE.md). Click order is
near-left, near-right, far-left, far-right, where "near" is the baseline
closest to the camera.
"""

import json

import cv2
import numpy as np

# Singles court in meters: 8.23m (sideline-to-sideline) x 23.77m (baseline-to-baseline).
SINGLES_COURT_POINTS = [
    (0.0, 0.0),
    (8.23, 0.0),
    (0.0, 23.77),
    (8.23, 23.77),
]

CORNER_LABELS = [
    "Near-left baseline corner (closest to the camera, left side)",
    "Near-right baseline corner (closest to the camera, right side)",
    "Far-left baseline corner (opposite baseline, left side)",
    "Far-right baseline corner (opposite baseline, right side)",
]


def compute_homography(image_points: list[tuple[float, float]]) -> np.ndarray:
    if len(image_points) != 4:
        raise ValueError(f"Expected 4 image points, got {len(image_points)}")

    src = np.array(image_points, dtype=np.float32)
    dst = np.array(SINGLES_COURT_POINTS, dtype=np.float32)
    homography, _ = cv2.findHomography(src, dst)
    return homography


def serialize_calibration(image_points: list[tuple[float, float]], homography: np.ndarray) -> bytes:
    payload = {
        "image_points": image_points,
        "court_points": SINGLES_COURT_POINTS,
        "homography": homography.tolist(),
    }
    return json.dumps(payload).encode()


def deserialize_calibration(data: bytes) -> np.ndarray:
    payload = json.loads(data)
    return np.array(payload["homography"], dtype=np.float64)

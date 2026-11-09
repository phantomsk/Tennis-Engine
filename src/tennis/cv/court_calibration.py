"""Manual court calibration: 8 user-clicked image points -> homography to court meters.

Fixed-camera, planar-court assumption. The homography is valid ONLY for points
at height 0 (feet, ball bounces) -- never project hips, wrists or an in-flight
ball through it.

Court frame (singles): x runs 0-8.23 m from the left singles sideline, y runs
0-23.77 m from the near baseline (the one closest to the camera). Click order:
the 4 baseline corners, then the 4 points where the service lines meet the
singles sidelines.
"""

import json
from dataclasses import dataclass, field

import cv2
import numpy as np

SINGLES_WIDTH = 8.23
COURT_LENGTH = 23.77
SERVICE_NEAR = 5.485
SERVICE_FAR = 18.285
NET_Y = 11.885
ALLEY = 1.37  # doubles alley width

COURT_POINTS = [
    (0.0, 0.0),
    (SINGLES_WIDTH, 0.0),
    (0.0, COURT_LENGTH),
    (SINGLES_WIDTH, COURT_LENGTH),
    (0.0, SERVICE_NEAR),
    (SINGLES_WIDTH, SERVICE_NEAR),
    (0.0, SERVICE_FAR),
    (SINGLES_WIDTH, SERVICE_FAR),
]

POINT_LABELS = [
    "Near-left baseline corner (closest to the camera, left singles sideline)",
    "Near-right baseline corner (closest to the camera, right singles sideline)",
    "Far-left baseline corner (opposite baseline, left singles sideline)",
    "Far-right baseline corner (opposite baseline, right singles sideline)",
    "Where the near service line meets the LEFT singles sideline",
    "Where the near service line meets the RIGHT singles sideline",
    "Where the far service line meets the LEFT singles sideline",
    "Where the far service line meets the RIGHT singles sideline",
]

MAX_RMS_PX = 3.0
MAX_POINT_RESIDUAL_PX = 4.0
MAX_SIDELINE_BOW_PX = 4.0

# Court lines (meters) drawn back onto the frame for visual verification. The
# doubles sidelines, centre service line and net are never clicked, so they are
# an independent check of the fit.
COURT_LINES_M = {
    "near_baseline": ((0.0, 0.0), (SINGLES_WIDTH, 0.0)),
    "far_baseline": ((0.0, COURT_LENGTH), (SINGLES_WIDTH, COURT_LENGTH)),
    "left_singles": ((0.0, 0.0), (0.0, COURT_LENGTH)),
    "right_singles": ((SINGLES_WIDTH, 0.0), (SINGLES_WIDTH, COURT_LENGTH)),
    "left_doubles": ((-ALLEY, 0.0), (-ALLEY, COURT_LENGTH)),
    "right_doubles": ((SINGLES_WIDTH + ALLEY, 0.0), (SINGLES_WIDTH + ALLEY, COURT_LENGTH)),
    "near_service": ((0.0, SERVICE_NEAR), (SINGLES_WIDTH, SERVICE_NEAR)),
    "far_service": ((0.0, SERVICE_FAR), (SINGLES_WIDTH, SERVICE_FAR)),
    "centre_service": ((SINGLES_WIDTH / 2, SERVICE_NEAR), (SINGLES_WIDTH / 2, SERVICE_FAR)),
    "net": ((-ALLEY, NET_Y), (SINGLES_WIDTH + ALLEY, NET_Y)),
}


class CalibrationError(ValueError):
    """The clicked points do not define a usable court homography."""


@dataclass
class CalibrationCheck:
    rms_px: float
    residuals_px: list[float]
    sideline_bow_px: float
    bad_points: list[int] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def _project(h: np.ndarray, points) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, h).reshape(-1, 2)


def compute_homography(image_points: list[tuple[float, float]]) -> np.ndarray:
    """Least-squares homography (image px -> court m) over all 8 points."""
    if len(image_points) != len(COURT_POINTS):
        raise ValueError(f"Expected {len(COURT_POINTS)} image points, got {len(image_points)}")

    src = np.array(image_points, dtype=np.float64)
    dst = np.array(COURT_POINTS, dtype=np.float64)
    h, _ = cv2.findHomography(src, dst, 0)
    if h is None or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        raise CalibrationError("These clicks do not define a court plane -- check the click order and re-click.")
    return h


def _distance_to_line(p, a, b) -> float:
    p, a, b = (np.asarray(v, dtype=np.float64) for v in (p, a, b))
    d = b - a
    return float(abs(d[0] * (p - a)[1] - d[1] * (p - a)[0]) / np.linalg.norm(d))


def _sideline_bow(points) -> float:
    """Max distance of the service-line clicks from the straight baseline-to-baseline sideline.

    A perspective view keeps straight lines straight, so any bow is lens
    distortion (or a misclick).
    """
    nl, nr, fl, fr, snl, snr, sfl, sfr = points
    return max(
        _distance_to_line(snl, nl, fl),
        _distance_to_line(sfl, nl, fl),
        _distance_to_line(snr, nr, fr),
        _distance_to_line(sfr, nr, fr),
    )


def _baseline_order_ok(points) -> bool:
    nl, nr, fl, fr = (np.asarray(p, dtype=np.float64) for p in points[:4])
    near_below_far = min(nl[1], nr[1]) > max(fl[1], fr[1])
    left_of_right = nl[0] < nr[0] and fl[0] < fr[0]
    quad = [nl, nr, fr, fl]
    crosses = []
    for i in range(4):
        a, b, c = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        ab, bc = b - a, c - b
        crosses.append(ab[0] * bc[1] - ab[1] * bc[0])
    convex = all(c > 0 for c in crosses) or all(c < 0 for c in crosses)
    return near_below_far and left_of_right and convex


def validate_calibration(image_points: list[tuple[float, float]], homography: np.ndarray) -> CalibrationCheck:
    expected_px = _project(np.linalg.inv(homography), COURT_POINTS)
    residuals = np.linalg.norm(expected_px - np.asarray(image_points, dtype=np.float64), axis=1)
    rms = float(np.sqrt(np.mean(residuals**2)))
    bow = _sideline_bow(image_points)

    check = CalibrationCheck(
        rms_px=rms,
        residuals_px=[float(r) for r in residuals],
        sideline_bow_px=bow,
        bad_points=[i for i, r in enumerate(residuals) if r > MAX_POINT_RESIDUAL_PX],
    )
    if not _baseline_order_ok(image_points):
        check.issues.append("Baseline corners are out of order or not convex -- re-click in the listed order.")
    if rms > MAX_RMS_PX:
        check.issues.append(f"Reprojection error {rms:.1f}px exceeds {MAX_RMS_PX:.0f}px -- re-click the highlighted points.")
    if bow > MAX_SIDELINE_BOW_PX:
        check.issues.append(
            f"Sidelines bow by {bow:.1f}px: lens distortion (disable the ultra-wide lens) or a misclick."
        )
    return check


def court_lines_px(homography: np.ndarray) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """Every court line projected into image pixels, for the verification overlay."""
    inv = np.linalg.inv(homography)
    lines = {}
    for name, (a, b) in COURT_LINES_M.items():
        pa, pb = _project(inv, [a, b])
        lines[name] = ((float(pa[0]), float(pa[1])), (float(pb[0]), float(pb[1])))
    return lines


def serialize_calibration(
    image_points: list[tuple[float, float]],
    homography: np.ndarray,
    check: CalibrationCheck,
    handedness: dict[str, str],
    user_side: str,
) -> bytes:
    payload = {
        "image_points": [list(p) for p in image_points],
        "court_points": [list(p) for p in COURT_POINTS],
        "homography": homography.tolist(),
        "rms_px": check.rms_px,
        "residuals_px": check.residuals_px,
        "handedness": handedness,
        "user_side": user_side,
    }
    return json.dumps(payload, sort_keys=True).encode()


def deserialize_calibration(data: bytes) -> np.ndarray:
    payload = json.loads(data)
    return np.array(payload["homography"], dtype=np.float64)

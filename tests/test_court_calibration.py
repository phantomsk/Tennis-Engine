import json

import numpy as np
import pytest

from tennis.cv.court_calibration import (
    COURT_POINTS,
    POINT_LABELS,
    CalibrationError,
    compute_homography,
    court_lines_px,
    deserialize_calibration,
    serialize_calibration,
    validate_calibration,
)

# A plausible baseline-camera view: court (meters) -> image (pixels).
# Built as a projective map so far-court points converge like a real camera.
COURT_TO_IMAGE = np.array(
    [
        [60.0, 25.6, 393.1],
        [0.0, -21.55, 1000.0],
        [0.0, 0.04, 1.0],
    ]
)  # near baseline at image y=1000 (~494 px wide), far baseline at y=250 (~253 px wide)


def _to_image(court_xy):
    p = COURT_TO_IMAGE @ np.array([court_xy[0], court_xy[1], 1.0])
    return (float(p[0] / p[2]), float(p[1] / p[2]))


def _clicks():
    return [_to_image(pt) for pt in COURT_POINTS]


def _apply(h, xy):
    p = h @ np.array([xy[0], xy[1], 1.0])
    return p[:2] / p[2]


def test_eight_reference_points_with_singles_geometry():
    assert len(COURT_POINTS) == 8 == len(POINT_LABELS)
    assert COURT_POINTS[:4] == [(0.0, 0.0), (8.23, 0.0), (0.0, 23.77), (8.23, 23.77)]
    assert COURT_POINTS[4:] == [(0.0, 5.485), (8.23, 5.485), (0.0, 18.285), (8.23, 18.285)]


def test_compute_homography_recovers_court_coordinates_from_eight_clicks():
    clicks = _clicks()

    h = compute_homography(clicks)

    for click, court in zip(clicks, COURT_POINTS):
        assert _apply(h, click) == pytest.approx(court, abs=1e-3)


def test_compute_homography_rejects_wrong_point_count():
    with pytest.raises(ValueError):
        compute_homography(_clicks()[:4])


def test_compute_homography_raises_calibration_error_on_degenerate_clicks():
    # All eight clicks on one image row: no homography exists. Previously this
    # returned None and crashed later on None.tolist().
    with pytest.raises(CalibrationError):
        compute_homography([(float(i * 10), 50.0) for i in range(8)])


def test_validation_passes_for_clean_clicks():
    clicks = _clicks()

    check = validate_calibration(clicks, compute_homography(clicks))

    assert check.ok
    assert check.rms_px == pytest.approx(0.0, abs=1e-3)
    assert check.bad_points == []
    assert check.issues == []


def test_validation_flags_a_misclicked_point_for_reclicking():
    clicks = _clicks()
    x, y = clicks[5]
    clicks[5] = (x + 25.0, y - 15.0)  # near-right service-line point clicked badly

    check = validate_calibration(clicks, compute_homography(clicks))

    assert not check.ok
    assert 5 in check.bad_points
    assert check.rms_px > 3.0


def test_validation_flags_swapped_baseline_corners():
    clicks = _clicks()
    clicks[0], clicks[1] = clicks[1], clicks[0]  # near-left and near-right swapped

    check = validate_calibration(clicks, compute_homography(clicks))

    assert not check.ok
    assert any("order" in issue for issue in check.issues)


def test_validation_flags_bowed_sideline_as_lens_distortion():
    # Apply barrel-like distortion: push points outward by an amount that
    # grows with the squared distance from the image centre.
    centre = np.array([640.0, 600.0])
    distorted = []
    for x, y in _clicks():
        v = np.array([x, y]) - centre
        r2 = float(v @ v) / (640.0**2)
        distorted.append(tuple(centre + v * (1 + 0.08 * r2)))

    check = validate_calibration(distorted, compute_homography(distorted))

    assert check.sideline_bow_px > 4.0
    assert not check.ok
    assert any("distortion" in issue for issue in check.issues)


def test_court_lines_px_projects_court_lines_back_into_the_image():
    clicks = _clicks()
    h = compute_homography(clicks)

    lines = court_lines_px(h)

    assert set(lines) >= {"near_baseline", "far_baseline", "net", "centre_service", "left_doubles"}
    (x1, y1), (x2, y2) = lines["near_baseline"]
    assert (x1, y1) == pytest.approx(clicks[0], abs=1e-2)
    assert (x2, y2) == pytest.approx(clicks[1], abs=1e-2)


def test_serialize_round_trip_keeps_homography_and_player_settings():
    clicks = _clicks()
    h = compute_homography(clicks)
    check = validate_calibration(clicks, h)

    data = serialize_calibration(
        clicks, h, check, handedness={"near": "right", "far": "left"}, user_side="near"
    )

    assert np.allclose(deserialize_calibration(data), h)
    payload = json.loads(data)
    assert payload["handedness"] == {"near": "right", "far": "left"}
    assert payload["user_side"] == "near"
    assert payload["rms_px"] == pytest.approx(0.0, abs=1e-3)

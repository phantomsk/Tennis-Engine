import numpy as np
import pytest

from tennis.cv.court_calibration import (
    SINGLES_COURT_POINTS,
    compute_homography,
    deserialize_calibration,
    serialize_calibration,
)


def test_compute_homography_maps_image_points_to_court_points():
    # A rectangle in image space (pixels) that is a simple scaled version of
    # the court, in the same near-left/near-right/far-left/far-right order.
    image_points = [(100.0, 500.0), (500.0, 500.0), (150.0, 100.0), (450.0, 100.0)]

    homography = compute_homography(image_points)

    for (x, y), (court_x, court_y) in zip(image_points, SINGLES_COURT_POINTS):
        projected = homography @ np.array([x, y, 1.0])
        projected /= projected[2]
        assert projected[0] == pytest.approx(court_x, abs=1e-3)
        assert projected[1] == pytest.approx(court_y, abs=1e-3)


def test_compute_homography_rejects_wrong_point_count():
    with pytest.raises(ValueError):
        compute_homography([(0.0, 0.0), (1.0, 1.0)])


def test_serialize_and_deserialize_round_trip():
    image_points = [(100.0, 500.0), (500.0, 500.0), (150.0, 100.0), (450.0, 100.0)]
    homography = compute_homography(image_points)

    data = serialize_calibration(image_points, homography)
    restored = deserialize_calibration(data)

    assert np.allclose(restored, homography)

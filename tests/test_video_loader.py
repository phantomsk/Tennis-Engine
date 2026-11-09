import cv2
import numpy as np

from tennis.cv.video_loader import extract_first_frame


def _make_video_bytes(tmp_path, first_frame_bgr, second_frame_bgr):
    path = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (64, 48))
    writer.write(first_frame_bgr)
    writer.write(second_frame_bgr)
    writer.release()
    return path.read_bytes()


def test_extract_first_frame_returns_rgb_array_of_first_frame(tmp_path):
    first_frame_bgr = np.full((48, 64, 3), (10, 20, 30), dtype=np.uint8)
    second_frame_bgr = np.full((48, 64, 3), (200, 100, 50), dtype=np.uint8)
    video_bytes = _make_video_bytes(tmp_path, first_frame_bgr, second_frame_bgr)

    frame_rgb = extract_first_frame(video_bytes)

    assert frame_rgb.shape == (48, 64, 3)
    expected_rgb = (30, 20, 10)  # BGR (10,20,30) -> RGB
    assert np.allclose(frame_rgb[0, 0], expected_rgb, atol=15)


def test_probe_video_reports_metadata_and_per_frame_timestamps(tmp_path):
    from conftest import make_video_bytes

    from tennis.cv.video_loader import probe_video

    path = tmp_path / "clip.mp4"
    make_video_bytes(path, n_frames=5, fps=10.0, size=(64, 48))

    probe = probe_video(path)

    assert probe["n_frames"] == 5
    assert probe["fps"] == 10.0
    assert (probe["width"], probe["height"]) == (64, 48)
    assert probe["timestamps_ms"] == [0.0, 100.0, 200.0, 300.0, 400.0]
    assert probe["variable_frame_rate"] is False
    assert probe["real_duration_s"] == 0.5


def test_probe_video_rescales_slow_motion_to_real_time(tmp_path):
    from conftest import make_video_bytes

    from tennis.cv.video_loader import probe_video

    path = tmp_path / "clip.mp4"
    make_video_bytes(path, n_frames=5, fps=10.0)

    probe = probe_video(path, time_scale=4.0)  # 4x slow-mo: 1 real second plays over 4

    assert probe["time_scale"] == 4.0
    assert probe["real_fps"] == 40.0
    assert probe["real_timestamps_ms"] == [0.0, 25.0, 50.0, 75.0, 100.0]
    assert probe["real_duration_s"] == 0.125


def test_extract_first_frame_raises_on_undecodable_video():
    import pytest

    with pytest.raises(ValueError):
        extract_first_frame(b"not a real video")

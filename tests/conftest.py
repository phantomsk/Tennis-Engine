import os
import tempfile

import cv2
import numpy as np
import pytest

_tmp_dir = tempfile.mkdtemp(prefix="tennis-test-")
os.environ.setdefault("ARTIFACTS_DIR", os.path.join(_tmp_dir, "artifacts"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp_dir}/test.db")


def make_video_bytes(path, n_frames=5, fps=10.0, size=(64, 48)) -> bytes:
    """Encode a tiny synthetic clip; frame i is filled with intensity 40*i."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), (40 * i) % 256, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


@pytest.fixture
def video_bytes(tmp_path) -> bytes:
    return make_video_bytes(tmp_path / "clip.mp4")

# Never spawn the background worker from app tests; tests drive it directly.
os.environ["TENNIS_WORKER"] = "external"

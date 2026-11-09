import pytest

from tennis.core.storage import LocalStorage


def test_artifacts_are_keyed_by_job_and_filename_with_explicit_extension(tmp_path):
    storage = LocalStorage(tmp_path)

    path = storage.write_artifact("job-1", "states.parquet", b"\x00\x01")

    assert path == str(tmp_path / "jobs" / "job-1" / "states.parquet")
    assert storage.read_artifact("job-1", "states.parquet") == b"\x00\x01"


def test_write_is_atomic_and_leaves_no_temp_files(tmp_path):
    storage = LocalStorage(tmp_path)

    storage.write_artifact("job-1", "ball.json", b"first")
    storage.write_artifact("job-1", "ball.json", b"second")

    assert storage.read_artifact("job-1", "ball.json") == b"second"
    assert sorted(p.name for p in (tmp_path / "jobs" / "job-1").iterdir()) == ["ball.json"]


def test_reading_a_missing_artifact_raises_without_creating_directories(tmp_path):
    storage = LocalStorage(tmp_path)

    with pytest.raises(FileNotFoundError):
        storage.read_artifact("nope", "ball.json")

    assert list(tmp_path.iterdir()) == []


def test_local_path_yields_a_readable_file_for_opencv(tmp_path):
    storage = LocalStorage(tmp_path)
    storage.write_artifact("job-1", "upload.mp4", b"video")

    with storage.local_path("job-1", "upload.mp4") as path:
        assert path.read_bytes() == b"video"

from tennis.core.storage import LocalStorage


def test_write_and_read_upload_round_trip(tmp_path):
    storage = LocalStorage(tmp_path)

    storage.write_upload("job-1", b"fake video bytes")

    assert storage.read_upload("job-1") == b"fake video bytes"

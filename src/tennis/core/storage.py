"""Storage seam: local filesystem now, S3 later behind the same interface."""

from pathlib import Path
from typing import Protocol


class Storage(Protocol):
    def write_artifact(self, job_id: str, stage: str, data: bytes) -> str: ...
    def read_artifact(self, job_id: str, stage: str) -> bytes: ...
    def write_upload(self, job_id: str, data: bytes) -> str: ...
    def read_upload(self, job_id: str) -> bytes: ...


class LocalStorage:
    """./artifacts/{job_id}/{stage}.json. MVP default."""

    def __init__(self, root: Path):
        self.root = root

    def _job_dir(self, job_id: str) -> Path:
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def write_artifact(self, job_id: str, stage: str, data: bytes) -> str:
        path = self._job_dir(job_id) / f"{stage}.json"
        path.write_bytes(data)
        return str(path)

    def read_artifact(self, job_id: str, stage: str) -> bytes:
        path = self._job_dir(job_id) / f"{stage}.json"
        return path.read_bytes()

    def write_upload(self, job_id: str, data: bytes) -> str:
        path = self._job_dir(job_id) / "upload.mp4"
        path.write_bytes(data)
        return str(path)

    def read_upload(self, job_id: str) -> bytes:
        path = self._job_dir(job_id) / "upload.mp4"
        return path.read_bytes()


# class S3Storage(Storage): ...   # scale-up, same interface

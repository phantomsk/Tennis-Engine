"""Storage seam: local filesystem now, S3 later behind the same interface.

Keys are (job_id, filename) with an explicit extension, laid out job-first
(jobs/{job_id}/{name}) so deleting or listing a job is one prefix operation --
the same keys work as S3 object keys.
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager, Iterator, Protocol


class Storage(Protocol):
    def write_artifact(self, job_id: str, name: str, data: bytes) -> str: ...
    def read_artifact(self, job_id: str, name: str) -> bytes: ...
    def local_path(self, job_id: str, name: str) -> ContextManager[Path]: ...


class LocalStorage:
    """./artifacts/jobs/{job_id}/{name}. MVP default."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, job_id: str, name: str) -> Path:
        return self.root / "jobs" / job_id / name

    def write_artifact(self, job_id: str, name: str, data: bytes) -> str:
        path = self._path(job_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file and rename, so a crash never leaves a partial
        # artifact that a later run would read.
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return str(path)

    def read_artifact(self, job_id: str, name: str) -> bytes:
        return self._path(job_id, name).read_bytes()

    @contextmanager
    def local_path(self, job_id: str, name: str) -> Iterator[Path]:
        """A filesystem path for libraries (cv2) that cannot read bytes.

        Local: the artifact itself. S3 later: download to a temp file.
        """
        path = self._path(job_id, name)
        if not path.exists():
            raise FileNotFoundError(path)
        yield path


# class S3Storage(Storage): ...   # scale-up, same interface

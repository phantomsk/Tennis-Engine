"""JobQueue seam: in-process now, SQS later behind the same interface."""

from collections import deque
from typing import Callable, Protocol

Handler = Callable[[str, str], None]


class JobQueue(Protocol):
    def enqueue(self, job_id: str, stage: str) -> None: ...
    def process_next(self, handler: Handler) -> bool: ...


class InProcessQueue:
    """Runs the next stage immediately. MVP default."""

    def __init__(self) -> None:
        self._messages: deque[tuple[str, str]] = deque()

    def enqueue(self, job_id: str, stage: str) -> None:
        self._messages.append((job_id, stage))

    def process_next(self, handler: Handler) -> bool:
        if not self._messages:
            return False
        job_id, stage = self._messages.popleft()
        handler(job_id, stage)
        return True


# class SqsQueue(JobQueue): ...   # scale-up, same interface

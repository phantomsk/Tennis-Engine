"""JobQueue seam: a SQLite table now, SQS later behind the same interface.

Semantics match SQS: at-least-once delivery. `receive` hides a message for
`visibility_timeout_s` (the lease); a message that is not acked before then is
delivered again. Consumers must therefore be idempotent.
"""

import time
import uuid
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import text


@dataclass(frozen=True)
class Message:
    id: str
    job_id: str
    stage: str
    attempts: int


class JobQueue(Protocol):
    def enqueue(self, job_id: str, stage: str) -> None: ...
    def receive(self, visibility_timeout_s: int) -> Message | None: ...
    def ack(self, msg: Message) -> None: ...
    def nack(self, msg: Message, error: str) -> None: ...


class SqliteQueue:
    """`queue_messages` table in the app's SQLite file. MVP default.

    Shared between the Streamlit process (producer) and the worker subprocess
    (consumer). `receive` leases a message with a single UPDATE ... RETURNING,
    which SQLite executes atomically under its write lock.
    """

    def __init__(self, database_url: str):
        self._url = database_url
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            from tennis.db.database import make_sqlite_engine  # lazy: avoids a config <-> db import cycle

            self._engine = make_sqlite_engine(self._url)
            with self._engine.begin() as conn:
                conn.execute(text(
                    "CREATE TABLE IF NOT EXISTS queue_messages ("
                    " id TEXT PRIMARY KEY, job_id TEXT NOT NULL, stage TEXT NOT NULL,"
                    " visible_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,"
                    " acked_at REAL, last_error TEXT, created_at REAL NOT NULL)"
                ))
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_queue_ready ON queue_messages (acked_at, visible_at)"
                ))
        return self._engine

    def enqueue(self, job_id: str, stage: str) -> None:
        now = time.time()
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO queue_messages (id, job_id, stage, visible_at, created_at)"
                     " VALUES (:id, :job_id, :stage, :now, :now)"),
                {"id": str(uuid.uuid4()), "job_id": job_id, "stage": stage, "now": now},
            )

    def receive(self, visibility_timeout_s: int) -> Message | None:
        now = time.time()
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    "UPDATE queue_messages SET visible_at = :until, attempts = attempts + 1"
                    " WHERE id = (SELECT id FROM queue_messages"
                    "             WHERE acked_at IS NULL AND visible_at <= :now"
                    "             ORDER BY created_at, rowid LIMIT 1)"
                    " RETURNING id, job_id, stage, attempts"
                ),
                {"now": now, "until": now + visibility_timeout_s},
            ).first()
        return Message(*row) if row else None

    def ack(self, msg: Message) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("UPDATE queue_messages SET acked_at = :now WHERE id = :id"),
                         {"now": time.time(), "id": msg.id})

    def nack(self, msg: Message, error: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("UPDATE queue_messages SET visible_at = :now, last_error = :err WHERE id = :id"),
                         {"now": time.time(), "err": error, "id": msg.id})

    def clear(self) -> None:
        """Delete every message (tests only)."""
        with self.engine.begin() as conn:
            conn.execute(text("DELETE FROM queue_messages"))


# class SqsQueue(JobQueue): ...   # scale-up, same interface; extend() added when needed

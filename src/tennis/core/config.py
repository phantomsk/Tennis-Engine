from pathlib import Path

from pydantic_settings import BaseSettings

from tennis.core.queue import SqliteQueue
from tennis.core.storage import LocalStorage


class Settings(BaseSettings):
    artifacts_dir: Path = Path("./artifacts")
    # Lives under artifacts/ (git-ignored). The pre-DAG ./tennis.db schema is
    # incompatible and is simply no longer used.
    database_url: str = "sqlite:///./artifacts/tennis.db"


settings = Settings()

STORAGE = LocalStorage(settings.artifacts_dir)  # swap to S3Storage() later
QUEUE = SqliteQueue(settings.database_url)  # swap to SqsQueue() later

from pathlib import Path

from pydantic_settings import BaseSettings

from tennis.core.queue import InProcessQueue
from tennis.core.storage import LocalStorage


class Settings(BaseSettings):
    artifacts_dir: Path = Path("./artifacts")
    database_url: str = "sqlite:///./tennis.db"


settings = Settings()

STORAGE = LocalStorage(settings.artifacts_dir)  # swap to S3Storage() later
QUEUE = InProcessQueue()  # swap to SqsQueue() later

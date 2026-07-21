from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from tennis.core.config import settings

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def init_db() -> None:
    from tennis.db import models  # noqa: F401  (registers models with Base)

    Base.metadata.create_all(bind=engine)

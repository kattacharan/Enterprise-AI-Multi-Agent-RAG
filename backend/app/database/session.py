from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.app.config import settings

# SQLite connection with check_same_thread=False for FastAPI multi-threaded requests
engine = create_engine(
    settings.SQLITE_DB_PATH,
    connect_args={"check_same_thread": False} if "sqlite" in settings.SQLITE_DB_PATH else {},
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency for yielding database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    from backend.app.database import models  # noqa
    Base.metadata.create_all(bind=engine)

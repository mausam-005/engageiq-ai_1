"""Database connection and session helpers."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config.settings import settings

# Create engine
engine = create_engine(settings.database_url)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Database session generator dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

"""Calibration model for per-user calibration data persistence."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base


class Calibration(Base):
    """Stores per-student calibration data in the database.

    Each student has at most one calibration record. Re-calibrating
    overwrites the existing record (upsert behaviour).
    """

    __tablename__ = "calibrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    calibration_data: Mapped[dict] = mapped_column(JSON, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    user = relationship("User", back_populates="calibration")

    def __repr__(self) -> str:
        return f"<Calibration user_id={self.user_id}>"

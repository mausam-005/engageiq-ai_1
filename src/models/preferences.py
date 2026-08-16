"""Preferences model for per-user nudge settings."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base


class Preferences(Base):
    """Stores user-specific nudge delivery preferences."""

    __tablename__ = "preferences"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    notification_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    overlay_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    audio_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    quiet_hours_start: Mapped[str | None] = mapped_column(String(5), nullable=True)
    quiet_hours_end: Mapped[str | None] = mapped_column(String(5), nullable=True)
    sensitivity: Mapped[str] = mapped_column(
        String(10), nullable=False, default="normal"
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationship back to user (optional)
    user = relationship("User", back_populates="preferences")

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"<Preferences user_id={self.user_id} sensitivity={self.sensitivity}>"

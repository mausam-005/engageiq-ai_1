"""EngagementLog model."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import Float, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.config.settings import EngagementState
from src.models.base import Base

if TYPE_CHECKING:
    from src.models.session import Session
    from src.models.user import User


class EngagementLog(Base):
    """EngagementLog model storing per-frame engagement signals."""

    __tablename__ = "engagement_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    score: Mapped[float] = mapped_column(Float, nullable=False)
    state: Mapped[EngagementState] = mapped_column(
        SQLEnum(EngagementState), nullable=False
    )

    # Core CV signals
    gaze: Mapped[str | None] = mapped_column(String(100), nullable=True)
    drowsiness: Mapped[float | None] = mapped_column(Float, nullable=True)
    expression: Mapped[str | None] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    session: Mapped["Session"] = relationship(back_populates="engagement_logs")
    user: Mapped["User"] = relationship(back_populates="engagement_logs")

    # Table arguments for composite index
    __table_args__ = (
        Index("ix_engagement_logs_session_user", "session_id", "user_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<EngagementLog id={self.id} session_id={self.session_id} "
            f"user_id={self.user_id} score={self.score}>"
        )

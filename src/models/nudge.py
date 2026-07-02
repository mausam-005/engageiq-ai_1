"""Nudge model."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import Float, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.config.settings import NudgeType
from src.models.base import Base

if TYPE_CHECKING:
    from src.models.session import Session
    from src.models.user import User


class Nudge(Base):
    """Nudge model representing intervention nudges sent to students."""

    __tablename__ = "nudges"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    nudge_type: Mapped[NudgeType] = mapped_column(SQLEnum(NudgeType), nullable=False)
    trigger_state: Mapped[str] = mapped_column(String(100), nullable=False)
    effectiveness_delta: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    session: Mapped["Session"] = relationship(back_populates="nudges")
    user: Mapped["User"] = relationship(back_populates="nudges")

    def __repr__(self) -> str:
        return (
            f"<Nudge id={self.id} session_id={self.session_id} "
            f"user_id={self.user_id} nudge_type={self.nudge_type}>"
        )

"""Track engagement change after nudges to measure effectiveness."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.config.settings import NudgeType
from src.models.nudge import Nudge


@dataclass
class EffectivenessResult:
    """Structured result summarizing effectiveness for a single nudge."""

    nudge_type: str
    timestamp: float
    pre_score: float
    post_scores: List[float] = field(default_factory=list)
    average_post_score: Optional[float] = None
    delta: float = 0.0
    effective: bool = False
    measurement_window: float = 60.0
    trigger_state: str = "unknown"
    user_id: Optional[int] = None
    session_id: Optional[int] = None
    persisted: bool = False


@dataclass
class NudgeStats:
    """Tracking totals and success rate for a nudge type."""

    total_count: int = 0
    effective_count: int = 0
    success_rate: float = 0.0

    def update(self, effective: bool) -> None:
        """Update statistics with a new nudge result."""
        self.total_count += 1
        if effective:
            self.effective_count += 1
        if self.total_count:
            self.success_rate = self.effective_count / self.total_count


@dataclass
class _PendingNudge:
    nudge_type: str
    timestamp: float
    pre_score: float
    trigger_state: str = "unknown"
    user_id: Optional[int] = None
    session_id: Optional[int] = None
    post_scores: List[float] = field(default_factory=list)
    persisted_nudge: Optional[Nudge] = None


def _normalize_nudge_type(nudge_type: Optional[str]) -> str:
    """Normalize historical nudge type labels for compatibility."""
    normalized = str(nudge_type or "").strip().lower()
    if normalized == "notification":
        return "popup"
    if normalized == "popup":
        return "popup"
    return normalized


def _db_nudge_type(nudge_type: str) -> Optional[NudgeType]:
    """Map tracker channel names to persisted NudgeType values."""
    normalized = _normalize_nudge_type(nudge_type)
    if normalized == "popup":
        return NudgeType.POPUP
    if normalized == "audio":
        return NudgeType.AUDIO
    if normalized == "email":
        return NudgeType.EMAIL
    return None


class EffectivenessTracker:
    """Tracks engagement improvement after nudges and exposes results."""

    def __init__(
        self,
        measurement_window: float = 60.0,
        observation_window: Optional[float] = None,
        db: Optional[Session] = None,
        user_id: Optional[int] = None,
        session_id: Optional[int] = None,
    ) -> None:
        """Initialize the tracker with a configurable observation window.

        Args:
            measurement_window: Window in seconds used to measure post-nudge scores.
            observation_window: Deprecated alias for measurement_window.
            db: Optional SQLAlchemy session for persistence.
            user_id: Optional user identifier for persisted history.
            session_id: Optional session identifier for persisted history.
        """
        self.measurement_window = (
            observation_window if observation_window is not None else measurement_window
        )
        self.db = db
        self.user_id = user_id
        self.session_id = session_id
        self._pending: List[_PendingNudge] = []
        self._results: List[EffectivenessResult] = []

    def record_nudge(
        self,
        timestamp: float,
        pre_nudge_score: Optional[float] = None,
        nudge_type: str = "notification",
        *,
        pre_score: Optional[float] = None,
        trigger_state: str = "unknown",
        user_id: Optional[int] = None,
        session_id: Optional[int] = None,
    ) -> None:
        """Record a nudge and prepare it for effectiveness measurement.

        Args:
            timestamp: Time when the nudge was delivered.
            pre_nudge_score: Engagement score before the nudge.
            nudge_type: Delivery channel name.
            pre_score: Backward-compatible alias for pre-nudge score.
            trigger_state: The state that triggered the nudge.
            user_id: Optional user identifier for persistence.
            session_id: Optional session identifier for persistence.
        """
        actual_pre_score = pre_nudge_score if pre_nudge_score is not None else pre_score
        if actual_pre_score is None:
            raise ValueError("pre_nudge_score or pre_score is required")

        nudge_type_normalized = nudge_type.strip().lower()
        user_id = user_id if user_id is not None else self.user_id
        session_id = session_id if session_id is not None else self.session_id

        persisted_nudge = None
        if self.db is not None and user_id is not None and session_id is not None:
            db_nudge_type = _db_nudge_type(nudge_type_normalized)
            if db_nudge_type is not None:
                persisted_nudge = Nudge(
                    session_id=session_id,
                    user_id=user_id,
                    nudge_type=db_nudge_type,
                    trigger_state=trigger_state,
                )
                self.db.add(persisted_nudge)
                self.db.flush()

        self._pending.append(
            _PendingNudge(
                nudge_type=nudge_type_normalized,
                timestamp=timestamp,
                pre_score=actual_pre_score,
                trigger_state=trigger_state,
                user_id=user_id,
                session_id=session_id,
                persisted_nudge=persisted_nudge,
            )
        )

    def record_post_score(self, timestamp: float, score: float) -> None:
        """Record a post-nudge score within the observation window."""
        for pending in sorted(self._pending, key=lambda item: item.timestamp):
            window_end = pending.timestamp + self.measurement_window
            if pending.timestamp <= timestamp <= window_end:
                pending.post_scores.append(score)
                break

    def update_score(self, timestamp: float, score: float) -> None:
        """Backward-compatible alias for record_post_score."""
        self.record_post_score(timestamp=timestamp, score=score)

    def evaluate_last_nudge(self) -> Optional[EffectivenessResult]:
        """Evaluate the most recent nudge and return its effectiveness result."""
        if not self._pending:
            return None

        pending = self._pending[-1]
        result = self._evaluate_pending_nudge(pending)
        self._results.append(result)

        if pending.persisted_nudge is not None and self.db is not None:
            pending.persisted_nudge.effectiveness_delta = result.delta
            self.db.flush()
            result.persisted = True

        return result

    def get_history(self) -> List[Dict[str, Any]]:
        """Return structured history suitable for nudge decision consumers."""
        return [self._result_to_history(entry) for entry in self._results]

    def get_stats(self) -> Dict[str, NudgeStats]:
        """Return effectiveness statistics grouped by nudge type."""
        stats: Dict[str, NudgeStats] = {}
        for entry in self._results:
            if entry.nudge_type not in stats:
                stats[entry.nudge_type] = NudgeStats()
            stats[entry.nudge_type].update(entry.effective)
        return stats

    @staticmethod
    def load_history(db: Session, user_id: int) -> List[Dict[str, Any]]:
        """Load persisted effectiveness history for a user from the database."""
        query = (
            db.query(Nudge)
            .filter(Nudge.user_id == user_id)
            .filter(Nudge.effectiveness_delta.isnot(None))
            .order_by(Nudge.created_at)
        )

        history: List[Dict[str, Any]] = []
        for row in query.all():
            result_type = (
                "notification"
                if row.nudge_type == NudgeType.POPUP
                else row.nudge_type.value
            )
            delta = row.effectiveness_delta or 0.0
            history.append(
                {
                    "nudge_type": result_type,
                    "effective": delta >= 10.0,
                    "was_effective": delta >= 10.0,
                    "delta": delta,
                    "pre_score": None,
                    "average_post_score": None,
                    "post_scores": [],
                    "timestamp": row.created_at.timestamp(),
                }
            )
        return history

    def _evaluate_pending_nudge(self, pending: _PendingNudge) -> EffectivenessResult:
        """Calculate the effectiveness result for a pending nudge."""
        if pending.post_scores:
            average_post_score = mean(pending.post_scores)
            delta = average_post_score - pending.pre_score
        else:
            average_post_score = None
            delta = 0.0

        return EffectivenessResult(
            nudge_type=pending.nudge_type,
            timestamp=pending.timestamp,
            pre_score=pending.pre_score,
            post_scores=list(pending.post_scores),
            average_post_score=average_post_score,
            delta=delta,
            effective=average_post_score is not None and delta >= 10.0,
            measurement_window=self.measurement_window,
            trigger_state=pending.trigger_state,
            user_id=pending.user_id,
            session_id=pending.session_id,
        )

    def _result_to_history(self, entry: EffectivenessResult) -> Dict[str, Any]:
        """Convert an effectiveness result into a decision-friendly history entry."""
        return {
            "nudge_type": entry.nudge_type,
            "effective": entry.effective,
            "was_effective": entry.effective,
            "delta": entry.delta,
            "pre_score": entry.pre_score,
            "average_post_score": entry.average_post_score,
            "post_scores": list(entry.post_scores),
            "timestamp": entry.timestamp,
        }

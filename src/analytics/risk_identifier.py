"""Identify at-risk students with consistently low engagement."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RiskEvaluation:
    """The result of a risk evaluation for a single student."""

    at_risk: bool
    reason: Optional[str]
    declining_trend: bool


class RiskIdentifier:
    """Identifies at-risk students based on sustained low engagement patterns."""

    def __init__(self, threshold: float = 50.0, consecutive_sessions: int = 3):
        self.threshold = threshold
        self.consecutive_sessions = consecutive_sessions

    def evaluate(self, student_id: str, session_scores: List[float]) -> RiskEvaluation:
        """Evaluate a student's engagement history for risk factors.

        Args:
            student_id: The ID of the student being evaluated.
            session_scores: A chronological list of the student's engagement scores.

        Returns:
            A RiskEvaluation object containing the results.
        """
        if not session_scores:
            return RiskEvaluation(at_risk=False, reason=None, declining_trend=False)

        at_risk = False
        reason = None
        declining_trend = False

        # 1. Consecutive Drop Detection
        if len(session_scores) >= self.consecutive_sessions:
            recent_scores = session_scores[-self.consecutive_sessions :]
            if all(score < self.threshold for score in recent_scores):
                at_risk = True
                reason = f"below_threshold_{self.consecutive_sessions}_sessions"

        # 2. Declining Trend Detection
        # Compare the average of the older half of sessions to the newer half.
        if len(session_scores) >= 4:
            mid = len(session_scores) // 2
            older_half = session_scores[:mid]
            newer_half = session_scores[mid:]

            older_avg = sum(older_half) / len(older_half)
            newer_avg = sum(newer_half) / len(newer_half)

            if older_avg > 0:
                drop_pct = (older_avg - newer_avg) / older_avg
                if drop_pct > 0.10:
                    declining_trend = True

        return RiskEvaluation(
            at_risk=at_risk, reason=reason, declining_trend=declining_trend
        )

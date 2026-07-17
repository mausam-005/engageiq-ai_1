"""Multi-signal engagement scorer."""

from typing import Optional

from src.config.scoring_weights import COURSE_WEIGHTS, CourseType


class EngagementScorer:
    """Engine that computes a unified engagement score from multiple signals."""

    def __init__(self, course_type: CourseType = CourseType.THEORY):
        """Initialize scorer with base weights for a specific course type.

        Args:
            course_type: The type of course (THEORY, LAB, SEMINAR) to dictate base weights.
        """
        self.course_type = course_type
        self.base_weights = COURSE_WEIGHTS[course_type]

    def compute_score(
        self,
        gaze: Optional[float] = None,
        pose: Optional[float] = None,
        expression: Optional[float] = None,
        alertness: Optional[float] = None,
    ) -> float:
        """Compute weighted engagement score with dynamic weight redistribution.

        If any signal is missing (None), its base weight is redistributed proportionally
        among the available signals so the total weight remains 1.0 (100%).

        Args:
            gaze: Gaze score (0-100), None if unavailable.
            pose: Head pose score (0-100), None if unavailable.
            expression: Expression score (0-100), None if unavailable.
            alertness: Alertness score (0-100), None if unavailable.

        Returns:
            Computed engagement score (0-100). Returns 0.0 if all signals are missing.
        """
        signals = {
            "gaze": gaze,
            "pose": pose,
            "expression": expression,
            "alertness": alertness,
        }

        # Filter out missing signals
        available_signals = {k: v for k, v in signals.items() if v is not None}

        if not available_signals:
            return 0.0

        # Calculate total base weight of available signals
        total_available_weight = sum(
            self.base_weights[k] for k in available_signals.keys()
        )

        # Handle edge case where all available base weights sum to 0 (shouldn't happen with valid config)
        if total_available_weight <= 0:
            return 0.0

        final_score = 0.0
        for name, value in available_signals.items():
            # Redistribute weight proportionally
            adjusted_weight = self.base_weights[name] / total_available_weight
            final_score += value * adjusted_weight

        return round(max(0.0, min(100.0, final_score)), 2)

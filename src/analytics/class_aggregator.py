"""Aggregate student engagement into class-level metrics.

Computes class-wide statistics (mean, median, std deviation, engaged
percentage) from individual student scores and detects engagement dips
where the class average drops significantly below the session average.

Privacy
-------
All student identifiers are anonymised before they leave this module.
Teachers see sequential labels ("Student A", "Student B") or hashed IDs,
never real database IDs or names.
"""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ClassStats:
    """Aggregated engagement statistics for a class at a point in time."""

    average: float
    median: float
    std_dev: float
    min_score: float
    max_score: float
    engaged_pct: float  # fraction of students scoring > 70
    student_count: int


@dataclass
class EngagementDip:
    """Represents a detected engagement dip in the class timeline."""

    minute: int
    score: float
    session_avg: float
    drop_pct: float  # how far below session avg (as a positive fraction)


class ClassAggregator:
    """Aggregates per-student engagement scores into class-level metrics.

    Usage::

        agg = ClassAggregator()

        # Compute stats from a list of scores
        stats = agg.aggregate([80, 70, 90, 60, 85])

        # Detect dips in a minute-by-minute timeline
        timeline = {10: 78, 11: 80, 12: 52, 13: 70}
        dips = agg.detect_dips(timeline, threshold=0.15)
    """

    ENGAGED_THRESHOLD: float = 70.0

    def aggregate(self, scores: List[float]) -> ClassStats:
        """Compute class-level engagement statistics from a list of scores.

        Args:
            scores: List of individual student engagement scores (0-100).
                    Missing or disconnected students should be excluded
                    before calling this method.

        Returns:
            ClassStats with mean, median, std_dev, min, max, engaged_pct.

        Raises:
            ValueError: if scores is empty.
        """
        if not scores:
            raise ValueError(
                "Cannot aggregate an empty list of scores. "
                "At least one student score is required."
            )

        avg = statistics.mean(scores)
        med = statistics.median(scores)
        std = statistics.pstdev(scores)  # population std dev
        min_s = min(scores)
        max_s = max(scores)
        engaged_count = sum(1 for s in scores if s > self.ENGAGED_THRESHOLD)
        engaged_pct = engaged_count / len(scores)

        return ClassStats(
            average=round(avg, 2),
            median=round(med, 2),
            std_dev=round(std, 2),
            min_score=round(min_s, 2),
            max_score=round(max_s, 2),
            engaged_pct=round(engaged_pct, 4),
            student_count=len(scores),
        )

    def detect_dips(
        self,
        timeline: Dict[int, float],
        threshold: float = 0.15,
    ) -> List[EngagementDip]:
        """Detect minutes where class engagement dipped significantly.

        A dip is defined as a minute where the class average score drops
        more than ``threshold`` (as a fraction) below the overall session
        average.

        Args:
            timeline: Mapping of minute → class average engagement score.
            threshold: Minimum fractional drop to flag as a dip (default 0.15
                       means 15% below session average).

        Returns:
            List of EngagementDip objects, sorted by minute.
        """
        if not timeline:
            return []

        session_avg = statistics.mean(timeline.values())
        if session_avg == 0:
            return []

        dips: List[EngagementDip] = []
        for minute in sorted(timeline.keys()):
            score = timeline[minute]
            drop = (session_avg - score) / session_avg
            if drop > threshold:
                dips.append(
                    EngagementDip(
                        minute=minute,
                        score=round(score, 2),
                        session_avg=round(session_avg, 2),
                        drop_pct=round(drop, 4),
                    )
                )

        return dips

    @staticmethod
    def anonymize_student_ids(
        student_ids: List[int],
    ) -> Dict[int, str]:
        """Replace real student IDs with anonymous labels.

        Uses a SHA-256 hash truncated to 8 characters so the mapping is
        deterministic (same input → same output) but not reversible.

        Args:
            student_ids: List of real database student IDs.

        Returns:
            Dict mapping real ID → anonymised label like "anon_a1b2c3d4".
        """
        mapping: Dict[int, str] = {}
        for sid in student_ids:
            hash_val = hashlib.sha256(str(sid).encode()).hexdigest()[:8]
            mapping[sid] = f"anon_{hash_val}"
        return mapping

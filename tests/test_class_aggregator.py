"""Tests for class-level engagement aggregator."""

import pytest

from src.analytics.class_aggregator import ClassAggregator, ClassStats, EngagementDip


@pytest.fixture
def aggregator():
    return ClassAggregator()


def test_normal_class_stats(aggregator):
    """Verify basic stats for a typical class of 10 students."""
    scores = [80, 70, 90, 60, 85, 75, 65, 80, 70, 90]
    stats = aggregator.aggregate(scores)

    assert isinstance(stats, ClassStats)
    assert stats.average == 76.5
    assert stats.median == 77.5
    assert stats.student_count == 10
    # 7 students score > 70: 80, 90, 85, 75, 80, 90 = 6? Let's check:
    # 80>70 yes, 70>70 no, 90>70 yes, 60>70 no, 85>70 yes,
    # 75>70 yes, 65>70 no, 80>70 yes, 70>70 no, 90>70 yes → 6/10
    assert stats.engaged_pct == 0.6
    assert stats.min_score == 60
    assert stats.max_score == 90
    assert stats.std_dev > 0


def test_class_with_engagement_dip(aggregator):
    """Verify dip detection when class average drops sharply at a minute."""
    timeline = {
        10: 78,
        11: 80,
        12: 75,
        13: 72,
        14: 52,  # big drop
        15: 48,  # big drop
        16: 55,
        17: 60,
        18: 70,
    }
    dips = aggregator.detect_dips(timeline, threshold=0.15)

    assert len(dips) >= 2
    dip_minutes = [d.minute for d in dips]
    assert 14 in dip_minutes
    assert 15 in dip_minutes
    for dip in dips:
        assert isinstance(dip, EngagementDip)
        assert dip.drop_pct > 0.15


def test_class_with_outlier(aggregator):
    """Verify stats handle an extreme outlier gracefully."""
    scores = [80, 80, 80, 80, 80, 80, 80, 80, 80, 5]
    stats = aggregator.aggregate(scores)

    assert stats.min_score == 5
    assert stats.max_score == 80
    assert stats.average == 72.5
    assert stats.student_count == 10


def test_class_with_single_student(aggregator):
    """Verify aggregation works with just one student."""
    stats = aggregator.aggregate([85.0])

    assert stats.average == 85.0
    assert stats.median == 85.0
    assert stats.std_dev == 0.0
    assert stats.student_count == 1
    assert stats.engaged_pct == 1.0


def test_empty_scores_raises(aggregator):
    """Verify that empty input raises ValueError."""
    with pytest.raises(ValueError, match="Cannot aggregate an empty list"):
        aggregator.aggregate([])


def test_no_dips_in_stable_timeline(aggregator):
    """No dips should be detected when engagement is stable."""
    timeline = {1: 75, 2: 76, 3: 74, 4: 77, 5: 75}
    dips = aggregator.detect_dips(timeline, threshold=0.15)
    assert dips == []


def test_anonymize_student_ids(aggregator):
    """Verify student IDs are anonymised deterministically."""
    ids = [101, 202, 303]
    mapping = ClassAggregator.anonymize_student_ids(ids)

    assert len(mapping) == 3
    for real_id in ids:
        label = mapping[real_id]
        assert label.startswith("anon_")
        assert len(label) == 13  # "anon_" + 8 hex chars

    # Deterministic: same input → same output
    mapping2 = ClassAggregator.anonymize_student_ids(ids)
    assert mapping == mapping2

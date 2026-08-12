"""Tests for nudge effectiveness tracking."""

from statistics import mean

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models import Base, Course, Session, User
from src.nudge.effectiveness_tracker import EffectivenessResult, EffectivenessTracker


def test_effective_nudge_calculation() -> None:
    tracker = EffectivenessTracker(measurement_window=60.0)
    tracker.record_nudge(nudge_type="notification", timestamp=0.0, pre_score=35.0)
    tracker.record_post_score(timestamp=30.0, score=55.0)
    tracker.record_post_score(timestamp=60.0, score=65.0)

    result = tracker.evaluate_last_nudge()

    assert isinstance(result, EffectivenessResult)
    assert result.effective is True
    assert result.delta == 25.0
    assert result.average_post_score == 60.0
    assert result.post_scores == [55.0, 65.0]


def test_ineffective_nudge() -> None:
    tracker = EffectivenessTracker(measurement_window=60.0)
    tracker.record_nudge(nudge_type="audio", timestamp=0.0, pre_score=30.0)
    tracker.record_post_score(timestamp=30.0, score=32.0)

    result = tracker.evaluate_last_nudge()

    assert result.effective is False
    assert result.delta == 2.0
    assert result.average_post_score == 32.0


def test_observation_window_ignores_late_scores() -> None:
    tracker = EffectivenessTracker(measurement_window=60.0)
    tracker.record_nudge(nudge_type="notification", timestamp=100.0, pre_score=40.0)
    tracker.record_post_score(timestamp=110.0, score=50.0)
    tracker.record_post_score(timestamp=161.0, score=80.0)

    result = tracker.evaluate_last_nudge()

    assert result.post_scores == [50.0]
    assert result.delta == 10.0
    assert result.effective is True


def test_multiple_post_scores_average_used() -> None:
    tracker = EffectivenessTracker(measurement_window=60.0)
    tracker.record_nudge(nudge_type="audio", timestamp=0.0, pre_score=20.0)
    tracker.record_post_score(timestamp=10.0, score=25.0)
    tracker.record_post_score(timestamp=20.0, score=35.0)
    tracker.record_post_score(timestamp=30.0, score=30.0)

    result = tracker.evaluate_last_nudge()

    assert result.average_post_score == mean([25.0, 35.0, 30.0])
    assert result.delta == result.average_post_score - 20.0
    assert result.effective is True


def test_stats_by_nudge_type() -> None:
    tracker = EffectivenessTracker(measurement_window=60.0)
    tracker.record_nudge(nudge_type="notification", timestamp=0.0, pre_score=30.0)
    tracker.record_post_score(timestamp=10.0, score=40.0)
    tracker.evaluate_last_nudge()

    tracker.record_nudge(nudge_type="audio", timestamp=100.0, pre_score=30.0)
    tracker.record_post_score(timestamp=110.0, score=32.0)
    tracker.evaluate_last_nudge()

    stats = tracker.get_stats()

    assert stats["notification"].total_count == 1
    assert stats["notification"].effective_count == 1
    assert stats["notification"].success_rate == 1.0
    assert stats["audio"].total_count == 1
    assert stats["audio"].effective_count == 0
    assert stats["audio"].success_rate == 0.0


def test_db_persistence_and_history_format() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    db_session = SessionLocal()
    try:
        teacher = User(name="Dr. Smith", email="smith@nst.edu", role="teacher")
        db_session.add(teacher)
        db_session.commit()

        course = Course(name="Chemistry", code="CH101", teacher_id=teacher.id)
        db_session.add(course)
        db_session.commit()

        lecture_session = Session(course_id=course.id, status="active")
        db_session.add(lecture_session)
        db_session.commit()

        student = User(name="Jane Doe", email="jane@nst.edu", role="student")
        db_session.add(student)
        db_session.commit()

        tracker = EffectivenessTracker(
            measurement_window=60.0,
            db=db_session,
            user_id=student.id,
            session_id=lecture_session.id,
        )
        tracker.record_nudge(
            nudge_type="notification",
            timestamp=0.0,
            pre_score=35.0,
            trigger_state="distracted",
        )
        tracker.record_post_score(timestamp=30.0, score=55.0)
        result = tracker.evaluate_last_nudge()

        assert result.effective is True
        assert result.delta == 20.0

        persisted_history = EffectivenessTracker.load_history(db_session, student.id)
        assert persisted_history[-1]["nudge_type"] == "notification"
        assert persisted_history[-1]["effective"] is True
    finally:
        db_session.close()

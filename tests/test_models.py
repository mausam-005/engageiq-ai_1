"""Unit tests for database models."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from src.config.settings import (
    EngagementState,
    NudgeType,
    PrivacyMode,
    UserRole,
)
from src.models import Base, Course, EngagementLog, Nudge, Report, Session, User


@pytest.fixture(name="db_session")
def fixture_db_session():
    """Fixture to provide a clean in-memory SQLite database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


def test_model_creation(db_session):
    """Test instantiating and inserting all 6 models."""
    # 1. User (Teacher)
    teacher = User(
        name="Dr. Smith",
        email="smith@nst.edu",
        role=UserRole.TEACHER,
        privacy_mode=PrivacyMode.SHARE_WITH_TEACHER,
    )
    db_session.add(teacher)
    db_session.commit()
    assert teacher.id is not None
    assert teacher.role == UserRole.TEACHER

    # 2. User (Student)
    student = User(
        name="John Doe",
        email="john@nst.edu",
        role=UserRole.STUDENT,
        privacy_mode=PrivacyMode.LOCAL_ONLY,
    )
    db_session.add(student)
    db_session.commit()
    assert student.id is not None
    assert student.role == UserRole.STUDENT

    # 3. Course
    course = Course(
        name="Data Structures",
        code="CS201",
        teacher_id=teacher.id,
    )
    db_session.add(course)
    db_session.commit()
    assert course.id is not None

    # 4. Session
    lecture_session = Session(
        course_id=course.id,
        status="active",
    )
    db_session.add(lecture_session)
    db_session.commit()
    assert lecture_session.id is not None
    assert lecture_session.status == "active"

    # 5. EngagementLog
    log = EngagementLog(
        session_id=lecture_session.id,
        user_id=student.id,
        score=85.5,
        state=EngagementState.ENGAGED,
        gaze="focused",
        drowsiness=0.1,
        expression="happy",
    )
    db_session.add(log)
    db_session.commit()
    assert log.id is not None
    assert log.score == 85.5

    # 6. Nudge
    nudge = Nudge(
        session_id=lecture_session.id,
        user_id=student.id,
        nudge_type=NudgeType.POPUP,
        trigger_state="distracted",
        effectiveness_delta=15.0,
    )
    db_session.add(nudge)
    db_session.commit()
    assert nudge.id is not None
    assert nudge.nudge_type == NudgeType.POPUP

    # 7. Report
    report = Report(
        session_id=lecture_session.id,
        report_type="lecture_summary",
        content_json={"average_engagement": 85.5},
    )
    db_session.add(report)
    db_session.commit()
    assert report.id is not None
    assert report.content_json["average_engagement"] == 85.5


def test_model_relationships(db_session):
    """Test ORM relationships and back_populates attributes."""
    teacher = User(name="Dr. Smith", email="smith@nst.edu", role=UserRole.TEACHER)
    student = User(name="John Doe", email="john@nst.edu", role=UserRole.STUDENT)
    db_session.add_all([teacher, student])
    db_session.commit()

    course = Course(name="Algorithms", code="CS301", teacher_id=teacher.id)
    db_session.add(course)
    db_session.commit()

    # Relationship: User -> Course (teacher to courses)
    assert len(teacher.courses) == 1
    assert teacher.courses[0].code == "CS301"

    lecture_session = Session(course_id=course.id, status="completed")
    db_session.add(lecture_session)
    db_session.commit()

    # Relationship: Course -> Session
    assert len(course.sessions) == 1
    assert course.sessions[0].status == "completed"

    log = EngagementLog(
        session_id=lecture_session.id,
        user_id=student.id,
        score=70.0,
        state=EngagementState.PASSIVE,
    )
    nudge = Nudge(
        session_id=lecture_session.id,
        user_id=student.id,
        nudge_type=NudgeType.AUDIO,
        trigger_state="drowsy",
    )
    report = Report(
        session_id=lecture_session.id,
        report_type="lecture_summary",
        content_json={"average_engagement": 70.0},
    )
    db_session.add_all([log, nudge, report])
    db_session.commit()

    # Relationship: Session -> logs, nudges, reports
    assert len(lecture_session.engagement_logs) == 1
    assert lecture_session.engagement_logs[0].score == 70.0

    assert len(lecture_session.nudges) == 1
    assert lecture_session.nudges[0].nudge_type == NudgeType.AUDIO

    assert len(lecture_session.reports) == 1
    assert lecture_session.reports[0].report_type == "lecture_summary"

    # Relationship: User -> logs, nudges (student backreferences)
    assert len(student.engagement_logs) == 1
    assert student.engagement_logs[0].score == 70.0

    assert len(student.nudges) == 1
    assert student.nudges[0].nudge_type == NudgeType.AUDIO


def test_model_constraints(db_session):
    """Test model constraints such as unique emails, unique course codes, and nullability."""
    # Test Unique User Email
    u1 = User(name="User 1", email="duplicate@nst.edu", role=UserRole.STUDENT)
    u2 = User(name="User 2", email="duplicate@nst.edu", role=UserRole.STUDENT)
    db_session.add(u1)
    db_session.commit()

    db_session.add(u2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # Test Unique Course Code
    teacher = User(name="Dr. Smith", email="smith@nst.edu", role=UserRole.TEACHER)
    db_session.add(teacher)
    db_session.commit()

    c1 = Course(name="Maths", code="MATH101", teacher_id=teacher.id)
    c2 = Course(name="Algebra", code="MATH101", teacher_id=teacher.id)
    db_session.add(c1)
    db_session.commit()

    db_session.add(c2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # Test Not Nullable Constraints (e.g. User name cannot be null)
    bad_user = User(name=None, email="noname@nst.edu", role=UserRole.STUDENT)
    db_session.add(bad_user)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

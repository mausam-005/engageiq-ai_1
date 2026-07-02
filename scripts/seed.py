"""Database seed script."""

import os
import sys
from datetime import datetime, timedelta

# Add root folder to sys.path to enable imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config.settings import (
    EngagementState,
    NudgeType,
    PrivacyMode,
    UserRole,
    settings,
)
from src.models import Course, EngagementLog, Nudge, Report, Session, User


def seed_db() -> None:
    """Seed the database with sample records."""
    # Build engine
    engine = create_engine(settings.database_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    print("Seeding database...")

    # Count variables
    users_created = 0
    courses_created = 0
    sessions_created = 0
    logs_created = 0
    nudges_created = 0
    reports_created = 0

    try:
        # 1. Teachers
        teachers_data = [
            {"name": "Dr. Angela Smith", "email": "asmith@nst.edu"},
            {"name": "Prof. Robert Jones", "email": "rjones@nst.edu"},
        ]
        teachers = []
        for t_info in teachers_data:
            teacher = db.query(User).filter_by(email=t_info["email"]).first()
            if not teacher:
                teacher = User(
                    name=t_info["name"],
                    email=t_info["email"],
                    role=UserRole.TEACHER,
                    privacy_mode=PrivacyMode.SHARE_WITH_TEACHER,
                )
                db.add(teacher)
                db.flush()
                users_created += 1
            teachers.append(teacher)

        # 2. Students
        students = []
        for i in range(1, 11):
            email = f"student{i}@nst.edu"
            student = db.query(User).filter_by(email=email).first()
            if not student:
                student = User(
                    name=f"Student Name {i}",
                    email=email,
                    role=UserRole.STUDENT,
                    privacy_mode=PrivacyMode.LOCAL_ONLY,
                )
                db.add(student)
                db.flush()
                users_created += 1
            students.append(student)

        # 3. Courses
        courses_data = [
            {
                "name": "Data Structures and Algorithms",
                "code": "CS201",
                "teacher": teachers[0],
            },
            {
                "name": "Machine Learning",
                "code": "CS302",
                "teacher": teachers[0],
            },
            {
                "name": "Introduction to Databases",
                "code": "CS103",
                "teacher": teachers[1],
            },
        ]
        courses = []
        for c_info in courses_data:
            course = db.query(Course).filter_by(code=c_info["code"]).first()
            if not course:
                course = Course(
                    name=c_info["name"],
                    code=c_info["code"],
                    teacher_id=c_info["teacher"].id,
                )
                db.add(course)
                db.flush()
                courses_created += 1
            courses.append(course)

        # 4. Sessions
        # Use a fixed baseline datetime to ensure strict idempotency across multiple runs
        now = datetime(2026, 6, 30, 12, 0, 0)
        sessions_data = [
            # CS201 Sessions
            {
                "course": courses[0],
                "start_time": now - timedelta(days=2),
                "end_time": now - timedelta(days=2, hours=-2),
                "status": "completed",
            },
            {
                "course": courses[0],
                "start_time": now - timedelta(days=1),
                "end_time": now - timedelta(days=1, hours=-2),
                "status": "completed",
            },
            # CS302 Sessions
            {
                "course": courses[1],
                "start_time": now - timedelta(hours=3),
                "end_time": None,
                "status": "active",
            },
            # CS103 Sessions
            {
                "course": courses[2],
                "start_time": now - timedelta(days=5),
                "end_time": now - timedelta(days=5, hours=-3),
                "status": "completed",
            },
            {
                "course": courses[2],
                "start_time": now - timedelta(days=4),
                "end_time": now - timedelta(days=4, hours=-3),
                "status": "completed",
            },
        ]
        sessions = []
        for s_info in sessions_data:
            session = (
                db.query(Session)
                .filter_by(
                    course_id=s_info["course"].id,
                    start_time=s_info["start_time"],
                )
                .first()
            )
            if not session:
                session = Session(
                    course_id=s_info["course"].id,
                    start_time=s_info["start_time"],
                    end_time=s_info["end_time"],
                    status=s_info["status"],
                )
                db.add(session)
                db.flush()
                sessions_created += 1
            sessions.append(session)

        # Commit so far so we can reference foreign keys
        db.commit()

        # 5. Engagement Logs (Seeded for active/completed sessions)
        # We only seed if no logs exist for the first session to ensure idempotency
        first_session = sessions[0]
        existing_logs = (
            db.query(EngagementLog).filter_by(session_id=first_session.id).first()
        )
        if not existing_logs:
            log_entries = [
                {
                    "student": students[0],
                    "score": 92.5,
                    "state": EngagementState.ENGAGED,
                    "gaze": "focused",
                    "drowsiness": 0.1,
                    "expression": "neutral",
                },
                {
                    "student": students[0],
                    "score": 34.0,
                    "state": EngagementState.DISTRACTED,
                    "gaze": "looking_away",
                    "drowsiness": 0.2,
                    "expression": "neutral",
                },
                {
                    "student": students[1],
                    "score": 88.0,
                    "state": EngagementState.ENGAGED,
                    "gaze": "focused",
                    "drowsiness": 0.05,
                    "expression": "happy",
                },
                {
                    "student": students[1],
                    "score": 15.0,
                    "state": EngagementState.DROWSY,
                    "gaze": "eyes_closed",
                    "drowsiness": 0.85,
                    "expression": "neutral",
                },
                {
                    "student": students[2],
                    "score": 55.0,
                    "state": EngagementState.PASSIVE,
                    "gaze": "focused",
                    "drowsiness": 0.15,
                    "expression": "bored",
                },
            ]
            for entry in log_entries:
                log = EngagementLog(
                    session_id=first_session.id,
                    user_id=entry["student"].id,
                    score=entry["score"],
                    state=entry["state"],
                    gaze=entry["gaze"],
                    drowsiness=entry["drowsiness"],
                    expression=entry["expression"],
                )
                db.add(log)
                logs_created += 1

        # 6. Nudges (Seeded for student1/student2 on first session)
        existing_nudges = db.query(Nudge).filter_by(session_id=first_session.id).first()
        if not existing_nudges:
            nudge_entries = [
                {
                    "student": students[0],
                    "type": NudgeType.POPUP,
                    "state": "distracted",
                    "delta": 25.5,
                },
                {
                    "student": students[1],
                    "type": NudgeType.AUDIO,
                    "state": "drowsy",
                    "delta": 45.0,
                },
            ]
            for entry in nudge_entries:
                nudge = Nudge(
                    session_id=first_session.id,
                    user_id=entry["student"].id,
                    nudge_type=entry["type"],
                    trigger_state=entry["state"],
                    effectiveness_delta=entry["delta"],
                )
                db.add(nudge)
                nudges_created += 1

        # 7. Reports
        existing_reports = (
            db.query(Report).filter_by(session_id=first_session.id).first()
        )
        if not existing_reports:
            report = Report(
                session_id=first_session.id,
                report_type="lecture_summary",
                content_json={
                    "average_engagement_score": 56.9,
                    "peak_engagement_time": "10:15 AM",
                    "distraction_alerts_sent": 2,
                    "drowsiness_alerts_sent": 1,
                    "summary_notes": "The lecture started with high engagement. A noticeable dip occurred at the 45-minute mark.",
                },
            )
            db.add(report)
            reports_created += 1

        # Final commit
        db.commit()

        print("Database seeding completed successfully.")
        print(
            f"Created/Verified: {users_created} new users, {courses_created} courses, "
            f"{sessions_created} sessions, {logs_created} engagement logs, "
            f"{nudges_created} nudges, {reports_created} reports."
        )

    except Exception as e:
        db.rollback()
        print(f"Error during seeding: {e}")
        raise e
    finally:
        db.close()


if __name__ == "__main__":
    seed_db()

"""Enrollments association table."""

from sqlalchemy import Column, ForeignKey, Integer, Table

from src.models.base import Base

enrollments = Table(
    "enrollments",
    Base.metadata,
    Column(
        "student_id",
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "course_id",
        Integer,
        ForeignKey("courses.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)

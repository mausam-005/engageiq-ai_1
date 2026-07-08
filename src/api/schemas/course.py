"""Course Pydantic schemas."""

from typing import List

from pydantic import BaseModel, ConfigDict

from src.api.schemas.user import UserResponse


class CourseBase(BaseModel):
    """Base course schema."""

    code: str
    name: str


class CourseCreate(CourseBase):
    """Schema for creating a course."""

    pass


class CourseResponse(CourseBase):
    """Schema for returning course data."""

    id: int
    teacher_id: int

    model_config = ConfigDict(from_attributes=True)


class CourseWithStudentsResponse(CourseResponse):
    """Schema for course including enrolled students."""

    students: List[UserResponse] = []

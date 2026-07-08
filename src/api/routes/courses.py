"""Course management routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.middleware.auth import get_current_user, require_role
from src.api.schemas.course import (
    CourseCreate,
    CourseResponse,
)
from src.database import get_db
from src.models.course import Course
from src.models.user import User

router = APIRouter(prefix="/courses", tags=["courses"])


@router.post("/", response_model=CourseResponse)
def create_course(
    course_in: CourseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(["teacher"])),
):
    """Create a new course (Teacher only)."""
    course = Course(
        code=course_in.code,
        name=course_in.name,
        teacher_id=current_user.id,
    )
    db.add(course)
    db.commit()
    db.refresh(course)
    return course


@router.get("/", response_model=list[CourseResponse])
def list_courses(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    """List courses relevant to the user."""
    if current_user.role == "teacher":
        return current_user.courses
    else:
        return current_user.enrolled_courses


@router.post("/{course_id}/enroll")
def enroll_student(
    course_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(["student"])),
):
    """Enroll current student in a course."""
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    if course in current_user.enrolled_courses:
        raise HTTPException(status_code=400, detail="Already enrolled")

    current_user.enrolled_courses.append(course)
    db.commit()
    return {"message": "Successfully enrolled"}

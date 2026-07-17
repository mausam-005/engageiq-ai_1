from enum import Enum


class CourseType(Enum):
    """Types of courses that influence engagement scoring weights."""

    THEORY = "THEORY"
    LAB = "LAB"
    SEMINAR = "SEMINAR"


# Default base weights mapped to course types
# Total weight for each type should sum to 1.0
COURSE_WEIGHTS: dict[CourseType, dict[str, float]] = {
    CourseType.THEORY: {
        "gaze": 0.4,
        "alertness": 0.3,
        "pose": 0.2,
        "expression": 0.1,
    },
    CourseType.LAB: {
        # In a lab, students often look away from the screen to work on equipment.
        "alertness": 0.4,
        "expression": 0.2,
        "gaze": 0.2,
        "pose": 0.2,
    },
    CourseType.SEMINAR: {
        # Balanced weights for a discussion-focused seminar.
        "gaze": 0.25,
        "alertness": 0.25,
        "pose": 0.25,
        "expression": 0.25,
    },
}

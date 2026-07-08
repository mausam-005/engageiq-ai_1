"""Tests for user authentication and authorization."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.database import get_db
from src.models.base import Base

# Import all models to ensure they are registered in the Base metadata
from src.models.course import Course  # noqa: F401
from src.models.enrollment import enrollments  # noqa: F401
from src.models.user import User

# Use an in-memory SQLite database, configured for concurrent thread access via FastAPI TestClient
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db():
    """Create a fresh database for each test."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db):
    """Test client with overridden dependencies."""

    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_mock_login_creates_student(client):
    """Test that logging in with mock credentials provisions a new student."""
    response = client.post(
        "/api/v1/auth/token",
        json={"credential": "mock_new_student@nst.edu"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

    # Fetch user profile
    token = data["access_token"]
    response = client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    user_data = response.json()
    assert user_data["email"] == "mock_new_student@nst.edu"
    assert user_data["role"] == "student"


def test_role_based_access_course_creation(client, db):
    """Test that only teachers can create courses."""
    # Create Teacher
    teacher_res = client.post(
        "/api/v1/auth/token",
        json={"credential": "mock_teacher@nst.edu"},
    )
    teacher_token = teacher_res.json()["access_token"]

    # Manually update role to teacher
    teacher_user = db.query(User).filter_by(email="mock_teacher@nst.edu").first()
    teacher_user.role = "teacher"
    db.commit()

    # Create Student
    student_res = client.post(
        "/api/v1/auth/token",
        json={"credential": "mock_student@nst.edu"},
    )
    student_token = student_res.json()["access_token"]

    # Student tries to create course (Should Fail)
    res_fail = client.post(
        "/api/v1/courses/",
        json={"code": "CS101", "name": "Intro to CS"},
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert res_fail.status_code == 403

    # Teacher tries to create course (Should Succeed)
    res_success = client.post(
        "/api/v1/courses/",
        json={"code": "CS102", "name": "Advanced CS"},
        headers={"Authorization": f"Bearer {teacher_token}"},
    )
    assert res_success.status_code == 200
    assert res_success.json()["code"] == "CS102"

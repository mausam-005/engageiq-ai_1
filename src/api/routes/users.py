"""User management routes."""

from fastapi import APIRouter, Depends

from src.api.middleware.auth import get_current_user
from src.api.schemas.user import UserResponse
from src.models.user import User

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current logged-in user profile."""
    return current_user

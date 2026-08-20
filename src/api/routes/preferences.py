"""Routes for managing per-user nudge preferences."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.middleware.auth import get_current_user
from src.api.schemas.preferences import PreferencesResponse, PreferencesUpdate
from src.database import get_db
from src.models.preferences import Preferences
from src.models.user import User

router = APIRouter(prefix="/preferences", tags=["preferences"])


@router.get("/{user_id}", response_model=PreferencesResponse)
def get_preferences(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PreferencesResponse:
    """Return the persisted preferences for a user or defaults if none exist.

    Only the user themself or a teacher may read another user's preferences.
    """
    # Authorization: allow self or teacher
    if current_user.id != user_id and current_user.role != "teacher":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed")

    prefs = db.query(Preferences).filter(Preferences.user_id == user_id).first()
    if prefs is None:
        # Return defaults
        return PreferencesResponse()
    return PreferencesResponse.from_orm(prefs)


@router.put("/{user_id}", response_model=PreferencesResponse)
def update_preferences(
    user_id: int,
    payload: PreferencesUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PreferencesResponse:
    """Update and persist the preferences for a user.

    Only the user themself or a teacher may modify preferences.
    """
    if current_user.id != user_id and current_user.role != "teacher":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed")

    prefs = db.query(Preferences).filter(Preferences.user_id == user_id).first()
    if prefs is None:
        prefs = Preferences(user_id=user_id)
        db.add(prefs)

    data = payload.model_dump(exclude_none=True)
    for key, val in data.items():
        setattr(prefs, key, val)

    db.commit()
    db.refresh(prefs)
    return PreferencesResponse.from_orm(prefs)

"""Authentication routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.middleware.auth import create_access_token
from src.database import get_db
from src.models.user import User

router = APIRouter(prefix="/auth", tags=["auth"])


class GoogleAuthRequest(BaseModel):
    credential: str


@router.post("/token")
def login_for_access_token(request: GoogleAuthRequest, db: Session = Depends(get_db)):
    """
    Exchange Google OAuth credential for JWT token.
    For development, if credential starts with 'mock_', bypass Google verification.
    """
    credential = request.credential

    if credential.startswith("mock_"):
        email = credential
        name = "Mock User"
    else:
        # In a real app, verify the Google JWT token here using google-auth library
        # email = verify_google_token(credential)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only mock credentials supported in dev mode",
        )

    user = db.query(User).filter(User.email == email).first()
    if not user:
        # Auto-provision user on first login
        user = User(email=email, name=name, role="student")
        db.add(user)
        db.commit()
        db.refresh(user)

    access_token = create_access_token(data={"sub": user.email, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer"}

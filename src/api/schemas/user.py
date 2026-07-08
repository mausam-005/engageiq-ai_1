"""User Pydantic schemas."""

from pydantic import BaseModel, ConfigDict, EmailStr


class UserBase(BaseModel):
    """Base user schema."""

    email: EmailStr
    name: str | None = None
    role: str = "student"


class UserCreate(UserBase):
    """Schema for user creation."""

    pass


class UserUpdate(BaseModel):
    """Schema for updating user data."""

    name: str | None = None
    role: str | None = None


class UserResponse(UserBase):
    """Schema for returning user data."""

    id: int

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    """Token schema."""

    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token payload schema."""

    email: str | None = None
    role: str | None = None

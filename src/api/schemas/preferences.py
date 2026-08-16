"""Pydantic schemas for user nudge preferences."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, root_validator, validator

TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


class PreferencesBase(BaseModel):
    """Base schema describing nudge preferences."""

    notification_enabled: bool = True
    overlay_enabled: bool = True
    audio_enabled: bool = True
    quiet_hours_start: str | None = None
    quiet_hours_end: str | None = None
    sensitivity: Literal["less", "normal", "more"] = "normal"

    @validator("quiet_hours_start", "quiet_hours_end")
    def validate_time_format(cls, v):
        """Validate quiet hours use HH:MM 24-hour format or None."""
        if v is None:
            return v
        if not TIME_RE.match(v):
            raise ValueError("time must be in HH:MM 24-hour format")
        return v

    @root_validator(skip_on_failure=True)
    def validate_quiet_hours_pair(cls, values):
        start = values.get("quiet_hours_start")
        end = values.get("quiet_hours_end")
        # If one is set, the other may also be set. Both None is allowed.
        if (start is None) ^ (end is None):
            # allow one missing? enforce both or none
            raise ValueError(
                "Both quiet_hours_start and quiet_hours_end must be set together or omitted"
            )
        return values


class PreferencesResponse(PreferencesBase):
    """Schema returned to clients when reading preferences."""

    model_config = ConfigDict(from_attributes=True)


class PreferencesUpdate(BaseModel):
    """Schema accepted for updating preferences. All fields optional."""

    notification_enabled: bool | None = None
    overlay_enabled: bool | None = None
    audio_enabled: bool | None = None
    quiet_hours_start: str | None = None
    quiet_hours_end: str | None = None
    sensitivity: Literal["less", "normal", "more"] | None = None

    @validator("quiet_hours_start", "quiet_hours_end")
    def validate_time_format(cls, v):
        if v is None:
            return v
        if not TIME_RE.match(v):
            raise ValueError("time must be in HH:MM 24-hour format")
        return v

    @root_validator(skip_on_failure=True)
    def validate_quiet_hours_pair(cls, values):
        start = values.get("quiet_hours_start")
        end = values.get("quiet_hours_end")
        if (start is None) ^ (end is None):
            raise ValueError(
                "Both quiet_hours_start and quiet_hours_end must be set together or omitted"
            )
        return values

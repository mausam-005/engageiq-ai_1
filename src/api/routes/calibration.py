"""Calibration API endpoints.

POST /api/v1/calibrate/{user_id}
    Submit a calibration payload (collected frame measurements) and receive
    back a CalibrationData dict with personalised thresholds.

GET /api/v1/calibrate/{user_id}
    Return the stored calibration data for a student.
    Falls back to population-level defaults when no record exists.

DELETE /api/v1/calibrate/{user_id}
    Remove stored calibration so the student starts fresh.
"""

from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.models.calibration import Calibration
from src.scoring.calibration import CalibrationData, CalibrationManager

router = APIRouter(prefix="/calibrate", tags=["calibration"])


# ---------------------------------------------------------------------------
# Database session helper
# ---------------------------------------------------------------------------


def _get_db_url() -> str:
    """Return the database URL from environment or a sensible default."""
    import os

    return os.getenv("DATABASE_URL", "sqlite:///./engageiq.db")


def get_db():
    """Yield a SQLAlchemy database session."""
    engine = create_engine(_get_db_url())
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class FrameMeasurement(BaseModel):
    """Single frame's worth of raw sensor measurements."""

    ear: float = Field(..., ge=0.0, le=1.0, description="Eye Aspect Ratio (0-1)")
    pitch: float = Field(..., description="Head pitch in degrees")
    yaw: float = Field(..., description="Head yaw in degrees")
    roll: float = Field(0.0, description="Head roll in degrees")
    expression: str = Field("neutral", description="Predicted expression label")


class CalibrationRequest(BaseModel):
    """Payload sent by the frontend after a 30-second calibration session."""

    frames: List[FrameMeasurement] = Field(
        ...,
        min_length=1,
        description="Frame measurements collected during calibration",
    )


class CalibrationResponse(BaseModel):
    """Calibration result returned to the client."""

    user_id: int
    is_default: bool
    resting_ear: float
    ear_threshold: float
    baseline_pitch: float
    baseline_yaw: float
    baseline_roll: float
    gaze_yaw_min: float
    gaze_yaw_max: float
    gaze_pitch_min: float
    gaze_pitch_max: float
    expression_distribution: Dict[str, float]
    frames_collected: int
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_response(data: CalibrationData, message: str) -> CalibrationResponse:
    d = data.to_dict()
    return CalibrationResponse(**d, message=message)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/{user_id}", response_model=CalibrationResponse, status_code=201)
def submit_calibration(
    user_id: int,
    payload: CalibrationRequest,
    db: Session = Depends(get_db),
):
    """Process calibration frames and persist personalised thresholds.

    The client collects ~450 frames (30 s x 15 FPS) from the CV pipeline
    and sends them in one batch. This endpoint computes baselines and stores
    the result in the database.

    Args:
        user_id: Student's database ID.
        payload: Frame measurements from the 30-second calibration window.
        db: SQLAlchemy database session (injected).

    Returns:
        CalibrationResponse with personalised EAR and gaze thresholds.
    """
    manager = CalibrationManager(user_id=user_id)

    for frame in payload.frames:
        manager.add_frame(
            ear=frame.ear,
            pitch=frame.pitch,
            yaw=frame.yaw,
            roll=frame.roll,
            expression=frame.expression,
        )

    try:
        data = manager.finalise()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Upsert: update existing or create new
    existing = db.query(Calibration).filter_by(user_id=user_id).first()
    if existing:
        existing.calibration_data = data.to_dict()
    else:
        record = Calibration(user_id=user_id, calibration_data=data.to_dict())
        db.add(record)
    db.commit()

    return _to_response(
        data,
        message=(
            f"Calibration complete. "
            f"EAR threshold set to {data.ear_threshold:.3f} "
            f"(resting EAR {data.resting_ear:.3f}). "
            f"{data.frames_collected} frames processed."
        ),
    )


@router.get("/{user_id}", response_model=CalibrationResponse)
def get_calibration(user_id: int, db: Session = Depends(get_db)):
    """Retrieve stored calibration data for a student.

    Falls back to population-level defaults (is_default=True) when no
    calibration record exists yet.

    Args:
        user_id: Student's database ID.
        db: SQLAlchemy database session (injected).

    Returns:
        CalibrationResponse -- either personalised or default thresholds.
    """
    record = db.query(Calibration).filter_by(user_id=user_id).first()

    if record:
        data = CalibrationData.from_dict(record.calibration_data)
        message = "Returning stored calibration data."
    else:
        data = CalibrationManager.default(user_id=user_id)
        message = (
            "No calibration found for this student. "
            "Using default thresholds -- may be less accurate for you. "
            "Complete a 30-second calibration for better results."
        )

    return _to_response(data, message=message)


@router.delete("/{user_id}", status_code=204)
def delete_calibration(user_id: int, db: Session = Depends(get_db)):
    """Remove stored calibration data so the student can recalibrate.

    Args:
        user_id: Student's database ID.
        db: SQLAlchemy database session (injected).

    Raises:
        404 if no calibration record exists for this user.
    """
    record = db.query(Calibration).filter_by(user_id=user_id).first()
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"No calibration record found for user {user_id}.",
        )
    db.delete(record)
    db.commit()

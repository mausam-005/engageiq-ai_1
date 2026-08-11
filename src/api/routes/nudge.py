"""Nudge delivery routes for the API."""

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.nudge.nudge_delivery import NudgeDelivery

router = APIRouter(prefix="/nudge", tags=["nudge"])


class NudgeRequest(BaseModel):
    """Payload used to trigger a nudge via the API."""

    type: str = Field(
        ..., description="Delivery channel: notification, overlay, or audio"
    )
    message: Optional[str] = Field(default=None, description="Message to deliver")
    user_id: Optional[int] = Field(default=None, description="Target user identifier")


@router.post("/test")
async def test_nudge(request: NudgeRequest) -> dict[str, object]:
    """Send a test nudge through the configured delivery channel."""
    delivery = NudgeDelivery()
    message = request.message or "Test nudge"
    delivered = await delivery.deliver(request.type, message)
    return {
        "ok": delivered,
        "channel": request.type,
        "message": message,
        "user_id": request.user_id,
    }

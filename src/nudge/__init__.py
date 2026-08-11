"""Nudge package - decision logic and delivery for student nudges."""

from src.nudge.effectiveness_tracker import EffectivenessTracker  # noqa: F401
from src.nudge.nudge_decision import NudgeDecision  # noqa: F401

__all__ = [
    "NudgeDecision",
    "NudgeDelivery",
    "EffectivenessTracker",
]


def __getattr__(name: str):
    """Lazily import delivery support to avoid import-time side effects."""
    if name == "NudgeDelivery":
        from src.nudge.nudge_delivery import NudgeDelivery

        return NudgeDelivery
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Tests for the nudge delivery system."""

import asyncio
import time

from src.nudge.nudge_delivery import NudgeDelivery


def test_notification_delivery() -> None:
    """Notification nudges should be logged and reported as delivered."""
    delivery = NudgeDelivery()

    result = asyncio.run(delivery.deliver("notification", "Time to refocus!"))

    assert result is True
    assert delivery.delivery_log[-1]["nudge_type"] == "notification"
    assert delivery.delivery_log[-1]["message"] == "Time to refocus!"


def test_overlay_delivery() -> None:
    """Overlay nudges should be logged with the overlay channel."""
    delivery = NudgeDelivery()

    result = asyncio.run(delivery.deliver("overlay", "Stay engaged"))

    assert result is True
    assert delivery.delivery_log[-1]["nudge_type"] == "overlay"
    assert delivery.delivery_log[-1]["message"] == "Stay engaged"


def test_audio_delivery() -> None:
    """Audio nudges should be logged with the audio channel."""
    delivery = NudgeDelivery()

    result = asyncio.run(delivery.deliver("audio", "Listen carefully"))

    assert result is True
    assert delivery.delivery_log[-1]["nudge_type"] == "audio"
    assert delivery.delivery_log[-1]["message"] == "Listen carefully"


def test_async_delivery_does_not_block() -> None:
    """Delivery should be asynchronous and yield control rather than block."""
    delivery = NudgeDelivery()

    start = time.perf_counter()
    asyncio.run(delivery.deliver("notification", "Keep going"))
    elapsed = time.perf_counter() - start

    assert elapsed < 0.25

"""Multi-channel nudge delivery for notification, overlay, and audio nudges."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class NudgeDelivery:
    """Asynchronously deliver nudges through notification, overlay, or audio."""

    def __init__(self, channels: Optional[List[str]] = None) -> None:
        """Initialize the delivery service with supported delivery channels.

        Args:
            channels: Optional channel names to enable. Defaults to all channels.
        """
        self.channels: List[str] = channels or ["notification", "overlay", "audio"]
        self.delivery_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    async def deliver(self, nudge_type: str, message: str) -> bool:
        """Asynchronously deliver a nudge through the requested channel.

        Args:
            nudge_type: Delivery channel identifier: notification, overlay, or audio.
            message: Human-readable nudge content.

        Returns:
            True when the nudge is accepted for delivery.

        Raises:
            ValueError: If the requested channel is unsupported.
        """
        channel = self._normalize_channel(nudge_type)
        await asyncio.sleep(0)

        if channel == "notification":
            await self.deliver_notification(message)
        elif channel == "overlay":
            await self.deliver_overlay(message)
        elif channel == "audio":
            await self.deliver_audio(message)
        else:
            raise ValueError(f"Unsupported nudge type: {nudge_type}")

        return True

    async def deliver_notification(self, message: str) -> bool:
        """Deliver a message through the notification channel.

        Args:
            message: The message to surface in the notification channel.

        Returns:
            True after the notification is logged and dispatched.
        """
        return await self._dispatch_channel("notification", message)

    async def deliver_overlay(self, message: str) -> bool:
        """Deliver a message through the overlay channel.

        Args:
            message: The message to surface as a non-blocking overlay.

        Returns:
            True after the overlay is logged and dispatched.
        """
        return await self._dispatch_channel("overlay", message)

    async def deliver_audio(self, message: str) -> bool:
        """Deliver a message through the audio channel.

        Args:
            message: The message to speak or play through audio.

        Returns:
            True after the audio nudge is logged and dispatched.
        """
        return await self._dispatch_channel("audio", message)

    async def _dispatch_channel(self, channel: str, message: str) -> bool:
        """Log and emit a nudge through a specific channel.

        Args:
            channel: The target delivery channel.
            message: The message to send to the channel.

        Returns:
            True after the channel has been processed.
        """
        await asyncio.sleep(0)
        self._log_delivery(channel, message)
        self.logger.info("Delivered %s nudge: %s", channel, message)
        return True

    def _normalize_channel(self, nudge_type: str) -> str:
        """Normalize a requested channel name.

        Args:
            nudge_type: The requested nudge type.

        Returns:
            A supported lowercase channel name.

        Raises:
            ValueError: If the requested channel is unsupported.
        """
        normalized = nudge_type.strip().lower()
        if normalized in self.channels:
            return normalized

        raise ValueError(f"Unsupported nudge type: {nudge_type}")

    def _log_delivery(self, nudge_type: str, message: str) -> None:
        """Record a delivery attempt in the in-memory log.

        Args:
            nudge_type: The channel used to deliver the message.
            message: The content delivered.
        """
        self.delivery_log.append(
            {
                "nudge_type": nudge_type,
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for the nudge delivery module."""
    parser = argparse.ArgumentParser(description="Deliver a nudge message")
    parser.add_argument("--type", required=True, help="Delivery channel")
    parser.add_argument(
        "--message",
        default="Please stay engaged",
        help="Nudge message",
    )
    return parser


def main() -> int:
    """Run the nudge delivery module as a CLI entry point."""
    parser = _build_argument_parser()
    args = parser.parse_args()

    async def _run_delivery() -> None:
        delivery = NudgeDelivery()
        await delivery.deliver(args.type, args.message)

    asyncio.run(_run_delivery())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

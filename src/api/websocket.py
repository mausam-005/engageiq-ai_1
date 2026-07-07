"""WebSocket endpoint for real-time video frame streaming."""

import base64
import json
import logging
import time
from typing import Dict

import cv2
import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from src.pipeline.preprocessor import FramePreprocessor

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections keyed by session ID."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(
        self, websocket: WebSocket, session_id: str, token: str | None
    ) -> bool:
        """Validate token, accept connection, and track it."""
        if not token:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return False

        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info("WebSocket connected: session=%s", session_id)
        return True

    def disconnect(self, session_id: str):
        """Remove a session's connection."""
        self.active_connections.pop(session_id, None)
        logger.info("WebSocket disconnected: session=%s", session_id)

    async def send_json(self, message: dict, session_id: str):
        """Send JSON to the client for a given session."""
        ws = self.active_connections.get(session_id)
        if ws is not None:
            await ws.send_json(message)


manager = ConnectionManager()
preprocessor = FramePreprocessor()


def _decode_frame(
    frame_b64: str, width: int = 640, height: int = 480
) -> np.ndarray | None:
    """Decode a base64-encoded frame. Supports JPEG/PNG or raw bytes."""
    try:
        frame_bytes = base64.b64decode(frame_b64)
    except Exception:
        return None

    frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)

    # Try image-encoded first (JPEG/PNG)
    frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
    if frame is not None:
        return frame

    # Fall back to raw pixel bytes
    expected = height * width * 3
    if frame_arr.size == expected:
        return frame_arr.reshape((height, width, 3))

    return None


@router.websocket("/ws/session/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket, session_id: str, token: str = Query(None)
):
    """Receive base64-encoded frames, preprocess, and return engagement scores."""
    connected = await manager.connect(websocket, session_id, token)
    if not connected:
        return

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_json({"error": "Invalid JSON"}, session_id)
                continue

            frame_b64 = payload.get("frame")
            timestamp = payload.get("timestamp")

            if not frame_b64:
                await manager.send_json({"error": "Missing 'frame' field"}, session_id)
                continue

            frame = _decode_frame(
                frame_b64,
                width=payload.get("width", 640),
                height=payload.get("height", 480),
            )
            if frame is None:
                await manager.send_json({"error": "Could not decode frame"}, session_id)
                continue

            # Preprocess and time it
            t0 = time.monotonic()
            processed_frame = preprocessor.process(frame)
            processing_ms = round((time.monotonic() - t0) * 1000, 2)

            # TODO: Replace with actual ML pipeline scoring
            engagement_score = round(float(np.mean(processed_frame)), 4)

            await manager.send_json(
                {
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "engagement_score": engagement_score,
                    "processing_time_ms": processing_ms,
                },
                session_id,
            )

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception:
        logger.exception("Unexpected error in session %s", session_id)
        manager.disconnect(session_id)

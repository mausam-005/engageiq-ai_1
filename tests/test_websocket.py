"""Tests for WebSocket frame streaming endpoint."""

import base64
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

from src.api.main import app
from src.api.websocket import manager

client = TestClient(app)


def _make_frame(height=480, width=640):
    """Create a base64-encoded raw frame."""
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return base64.b64encode(frame.tobytes()).decode()


def test_rejects_connection_without_token():
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/session/123"):
            pass
    assert exc_info.value.code == 1008


def test_accepts_connection_with_token():
    with client.websocket_connect("/ws/session/123?token=valid"):
        pass


def test_frame_processing():
    with client.websocket_connect("/ws/session/s1?token=t") as ws:
        ws.send_text(json.dumps({"frame": _make_frame(), "timestamp": 1000.5}))
        resp = ws.receive_json()

        assert resp["session_id"] == "s1"
        assert resp["timestamp"] == 1000.5
        assert 0.0 <= resp["engagement_score"] <= 1.0
        assert "processing_time_ms" in resp


def test_error_on_missing_frame():
    with client.websocket_connect("/ws/session/s2?token=t") as ws:
        ws.send_text(json.dumps({"timestamp": 1.0}))
        resp = ws.receive_json()
        assert "error" in resp


def test_error_on_invalid_json():
    with client.websocket_connect("/ws/session/s3?token=t") as ws:
        ws.send_text("NOT JSON {{{")
        resp = ws.receive_json()
        assert "error" in resp


def test_disconnect_cleanup():
    sid = "cleanup-test"
    with client.websocket_connect(f"/ws/session/{sid}?token=t"):
        assert sid in manager.active_connections
    assert sid not in manager.active_connections


def test_multiple_concurrent_sessions():
    with client.websocket_connect("/ws/session/a?token=t") as ws1:
        with client.websocket_connect("/ws/session/b?token=t") as ws2:
            payload = json.dumps({"frame": _make_frame(), "timestamp": 0.0})
            ws1.send_text(payload)
            ws2.send_text(payload)

            assert ws1.receive_json()["session_id"] == "a"
            assert ws2.receive_json()["session_id"] == "b"

        assert "b" not in manager.active_connections
        assert "a" in manager.active_connections
    assert "a" not in manager.active_connections

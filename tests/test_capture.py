"""Unit tests for the WebcamCapture class."""

import time
from unittest.mock import MagicMock, patch

import numpy as np

from src.pipeline.capture import WebcamCapture


def test_capture_initialization():
    """Test WebcamCapture initialisation and configurations."""
    cap = WebcamCapture(source=0, fps=15, target_size=(640, 480))
    assert cap.source == 0
    assert cap.fps == 15
    assert cap.target_size == (640, 480)
    assert cap.cap is None


@patch("cv2.VideoCapture")
def test_capture_start_success(mock_video_capture):
    """Test capture source starts successfully and configures properties."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    cap = WebcamCapture(source=0, fps=15, target_size=(640, 480))
    assert cap.start() is True
    assert cap.cap == mock_cap

    # Check resolution set calls
    mock_cap.set.assert_any_call(3, 640)  # CAP_PROP_FRAME_WIDTH
    mock_cap.set.assert_any_call(4, 480)  # CAP_PROP_FRAME_HEIGHT


@patch("cv2.VideoCapture")
def test_capture_start_failure(mock_video_capture):
    """Test capture handles closed sources gracefully."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_video_capture.return_value = mock_cap

    cap = WebcamCapture(source=999, fps=15)
    assert cap.start() is False
    assert cap.cap is None


@patch("cv2.VideoCapture")
def test_capture_read_throttling(mock_video_capture):
    """Test capture read method returns frame, timestamp, and applies throttling."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # Return a dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_frame)
    mock_video_capture.return_value = mock_cap

    # Target 30 FPS -> interval 33.3ms
    cap = WebcamCapture(source=0, fps=30)
    assert cap.start() is True

    # Read 3 frames consecutively and time the elapsed interval
    start_time = time.time()
    for _ in range(3):
        frame, ts = cap.read()
        assert frame is not None
        assert ts > 0

    elapsed = time.time() - start_time
    # 3 frames at 30 FPS should take at least (2 * 0.0333) = ~0.066 seconds
    # (since the first read completes immediately and subsequent 2 are throttled)
    assert elapsed >= 0.05


@patch("cv2.VideoCapture")
def test_capture_graceful_release(mock_video_capture):
    """Test that capture release releases OpenCV VideoCapture."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    cap = WebcamCapture(source=0, fps=15)
    assert cap.start() is True
    cap.release()
    assert cap.cap is None
    mock_cap.release.assert_called_once()

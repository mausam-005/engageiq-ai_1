"""Tests for the MediaPipe face mesh detector."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.detection.face_mesh import FaceLandmarks, FaceMeshDetector, FaceMeshResult


class _FakeFaceMesh:
    """Simple fake mesh backend for unit tests."""

    def __init__(self, results):
        self._results = results

    def process(self, rgb_frame):
        """Return predefined processing results."""
        return self._results


def _make_landmarks(count: int):
    """Create a fake list of landmarks with the expected shape."""
    landmarks = []
    for _ in range(count):
        landmarks.append(
            [
                SimpleNamespace(x=0.0, y=0.0, z=0.0),
                SimpleNamespace(x=0.1, y=0.2, z=0.3),
            ]
        )
    return landmarks


def test_lazy_initialization():
    """The detector should not initialize the backend until detection runs."""
    detector = FaceMeshDetector()

    assert detector._mesh is None

    with patch.object(
        detector, "_init_mesh", side_effect=lambda: setattr(detector, "_mesh", object())
    ) as mock_init:
        detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    mock_init.assert_called_once()


def test_no_face_returns_empty():
    """Frames with no detected faces should return an empty result."""
    detector = FaceMeshDetector()
    detector._mesh = _FakeFaceMesh(SimpleNamespace(multi_face_landmarks=None))

    result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert isinstance(result, FaceMeshResult)
    assert result.faces == []
    assert result.frame_shape == (640, 480)
    assert isinstance(result.timestamp, float)


def test_landmarks_shape():
    """Each detected face should produce 468 landmarks in a (468, 3) array."""
    detector = FaceMeshDetector()
    fake_face = [SimpleNamespace(x=0.2, y=0.3, z=0.0) for _ in range(468)]
    fake_results = SimpleNamespace(multi_face_landmarks=[fake_face])
    detector._mesh = _FakeFaceMesh(fake_results)

    result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(result.faces) == 1
    assert isinstance(result.faces[0], FaceLandmarks)
    assert result.faces[0].landmarks.shape == (468, 3)
    assert len(result.faces[0].bbox) == 4
    assert result.faces[0].bbox[2] >= 0
    assert result.faces[0].bbox[3] >= 0


def test_confidence_range():
    """Detections should expose a confidence value within [0.0, 1.0]."""
    detector = FaceMeshDetector()
    fake_face = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(468)]
    fake_results = SimpleNamespace(multi_face_landmarks=[fake_face])
    detector._mesh = _FakeFaceMesh(fake_results)

    result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert 0.0 <= result.faces[0].confidence <= 1.0


def test_multiple_faces():
    """The detector should honor the configured maximum number of faces."""
    detector = FaceMeshDetector(max_faces=3)
    fake_faces = [
        [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(468)] for _ in range(3)
    ]
    fake_results = SimpleNamespace(multi_face_landmarks=fake_faces)
    detector._mesh = _FakeFaceMesh(fake_results)

    result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(result.faces) == 3

"""Tests for the head pose estimation module."""

from unittest.mock import patch

import cv2
import numpy as np

from src.detection.head_pose import estimate_head_pose

MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)

LANDMARK_INDICES = [1, 199, 33, 263, 61, 291]


def _make_landmarks_from_pose(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Project the canonical face model into image coordinates for a pose."""
    frame_width = 640
    frame_height = 480
    camera_matrix = np.array(
        [
            [frame_width, 0.0, frame_width / 2.0],
            [0.0, frame_width, frame_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    projected_points, _ = cv2.projectPoints(
        MODEL_POINTS,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    landmarks = np.zeros((468, 3), dtype=np.float32)
    for idx, point in zip(LANDMARK_INDICES, projected_points.reshape(-1, 2)):
        landmarks[idx, 0] = point[0] / frame_width
        landmarks[idx, 1] = point[1] / frame_height
        landmarks[idx, 2] = 0.0

    return landmarks


def test_frontal_face_returns_floats():
    """A frontal face should yield a valid pose estimate with float angles."""
    landmarks = _make_landmarks_from_pose(
        np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    )

    pose = estimate_head_pose(landmarks, (480, 640, 3))

    assert pose is not None
    assert all(isinstance(value, float) for value in pose)


def test_head_turned_left_has_negative_yaw():
    """A leftward head turn should produce a negative yaw relative to frontal."""
    frontal = estimate_head_pose(
        _make_landmarks_from_pose(
            np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        ),
        (480, 640, 3),
    )
    left = estimate_head_pose(
        _make_landmarks_from_pose(
            np.array([0.0, 0.2, 0.0], dtype=np.float64), np.zeros(3, dtype=np.float64)
        ),
        (480, 640, 3),
    )

    assert frontal is not None
    assert left is not None
    assert left[1] < frontal[1]


def test_head_turned_right_has_positive_yaw():
    """A rightward head turn should produce a positive yaw relative to frontal."""
    frontal = estimate_head_pose(
        _make_landmarks_from_pose(
            np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        ),
        (480, 640, 3),
    )
    right = estimate_head_pose(
        _make_landmarks_from_pose(
            np.array([0.0, -0.2, 0.0], dtype=np.float64), np.zeros(3, dtype=np.float64)
        ),
        (480, 640, 3),
    )

    assert frontal is not None
    assert right is not None
    assert right[1] > frontal[1]


def test_looking_down_has_positive_pitch():
    """A downward head tilt should increase pitch relative to the frontal pose."""
    frontal = estimate_head_pose(
        _make_landmarks_from_pose(
            np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        ),
        (480, 640, 3),
    )
    down = estimate_head_pose(
        _make_landmarks_from_pose(
            np.array([0.2, 0.0, 0.0], dtype=np.float64), np.zeros(3, dtype=np.float64)
        ),
        (480, 640, 3),
    )

    assert frontal is not None
    assert down is not None
    assert down[0] > frontal[0]


def test_missing_landmarks_returns_none():
    """Insufficient landmark information should return None."""
    landmarks = np.zeros((5, 3), dtype=np.float32)

    assert estimate_head_pose(landmarks, (480, 640, 3)) is None


def test_solvepnp_failure_returns_none():
    """A failed solvePnP call should be handled gracefully."""
    landmarks = np.zeros((468, 3), dtype=np.float32)

    with patch(
        "src.detection.head_pose.cv2.solvePnP", return_value=(False, None, None)
    ):
        assert estimate_head_pose(landmarks, (480, 640, 3)) is None

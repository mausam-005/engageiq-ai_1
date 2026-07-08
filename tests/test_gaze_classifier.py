import numpy as np

from src.detection.gaze_classifier import (
    GazeState,
    calculate_ear,
    calculate_iris_ratio,
    classify_gaze,
)


def test_gaze_frontal():
    """Verify that AT_SCREEN is returned for straight-on pose and centered iris."""
    state = classify_gaze(pitch=0, yaw=0, iris_ratio=0.5, ear=1.0)
    assert state == GazeState.AT_SCREEN


def test_gaze_away_left():
    """Verify AWAY_LEFT is returned when looking left."""
    # Test head pose dominant
    state1 = classify_gaze(pitch=0, yaw=-25, iris_ratio=0.5, ear=1.0)
    assert state1 == GazeState.AWAY_LEFT

    # Test iris position dominant
    state2 = classify_gaze(pitch=0, yaw=0, iris_ratio=0.2, ear=1.0)
    assert state2 == GazeState.AWAY_LEFT


def test_gaze_away_right():
    """Verify AWAY_RIGHT is returned when looking right."""
    # Test head pose dominant
    state1 = classify_gaze(pitch=0, yaw=25, iris_ratio=0.5, ear=1.0)
    assert state1 == GazeState.AWAY_RIGHT

    # Test iris position dominant
    state2 = classify_gaze(pitch=0, yaw=0, iris_ratio=0.8, ear=1.0)
    assert state2 == GazeState.AWAY_RIGHT


def test_gaze_looking_down():
    """Verify LOOKING_DOWN is returned when head is tilted down."""
    state = classify_gaze(pitch=-30, yaw=0, iris_ratio=0.5, ear=1.0)
    assert state == GazeState.LOOKING_DOWN


def test_gaze_eyes_closed():
    """Verify EYES_CLOSED is returned when EAR is below threshold."""
    # Even if head is straight and iris is centered
    state = classify_gaze(pitch=0, yaw=0, iris_ratio=0.5, ear=0.1)
    assert state == GazeState.EYES_CLOSED


def test_calculate_ear():
    """Test EAR calculation logic with mock coordinates."""
    # A perfectly square eye would have EAR = (1 + 1) / (2 * 2) = 0.5
    landmarks = np.array(
        [
            [0, 1],  # 0: left inner
            [1, 2],  # 1: top left
            [3, 2],  # 2: top right
            [4, 1],  # 3: right outer
            [3, 0],  # 4: bottom right
            [1, 0],  # 5: bottom left
        ],
        dtype=float,
    )

    ear = calculate_ear(landmarks)

    # V1 (dist between [1,2] and [1,0]) = 2.0
    # V2 (dist between [3,2] and [3,0]) = 2.0
    # H (dist between [0,1] and [4,1]) = 4.0
    # EAR = (2 + 2) / (2 * 4) = 0.5
    assert np.isclose(ear, 0.5)


def test_calculate_iris_ratio():
    """Test Iris Position Ratio calculation logic."""
    landmarks = np.array(
        [
            [0, 0],  # inner corner
            [10, 0],  # random
            [10, 0],  # random
            [20, 0],  # outer corner (width = 20)
        ],
        dtype=float,
    )

    # Iris perfectly centered (X=10)
    ratio = calculate_iris_ratio(landmarks, np.array([10, 0]))
    assert np.isclose(ratio, 0.5)

    # Iris looking left (X=4)
    ratio_left = calculate_iris_ratio(landmarks, np.array([4, 0]))
    assert np.isclose(ratio_left, 0.2)

    # Iris looking right (X=16)
    ratio_right = calculate_iris_ratio(landmarks, np.array([16, 0]))
    assert np.isclose(ratio_right, 0.8)

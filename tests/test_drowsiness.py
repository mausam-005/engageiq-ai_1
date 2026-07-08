"""Tests for the EAR-based drowsiness detector.

Covers:
  1. Open-eye EAR value
  2. Closed-eye EAR value
  3. Blink (short closure) does NOT trigger drowsiness
  4. Sustained closure DOES trigger drowsiness
  5. Threshold edge cases and detector reset
  6. compute_ear validation errors
  7. extract_eye_landmarks helper
"""

import numpy as np
import pytest

from src.detection.drowsiness import (
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    DrowsinessDetector,
    DrowsinessState,
    compute_ear,
    extract_eye_landmarks,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _open_eye() -> list[tuple[float, float]]:
    """Return 6 landmark points representing a clearly open eye.

    Layout:

        p2(0.2,0.4)  p3(0.4,0.4)
     p1(0.0,0.3)              p4(0.6,0.3)
        p6(0.2,0.2)  p5(0.4,0.2)

    vertical_a = |p2-p6| = 0.2
    vertical_b = |p3-p5| = 0.2
    horizontal = |p1-p4| = 0.6
    EAR = (0.2 + 0.2) / (2 * 0.6) = 0.333...
    """
    return [
        (0.0, 0.3),  # p1 - left corner
        (0.2, 0.4),  # p2 - upper-left lid
        (0.4, 0.4),  # p3 - upper-right lid
        (0.6, 0.3),  # p4 - right corner
        (0.4, 0.2),  # p5 - lower-right lid
        (0.2, 0.2),  # p6 - lower-left lid
    ]


def _closed_eye() -> list[tuple[float, float]]:
    """Return 6 landmark points representing a fully closed eye.

    Layout (upper and lower lids almost touching):

        p2(0.2,0.31)  p3(0.4,0.31)
     p1(0.0,0.3)                p4(0.6,0.3)
        p6(0.2,0.29)  p5(0.4,0.29)

    vertical_a = |p2-p6| = 0.02
    vertical_b = |p3-p5| = 0.02
    horizontal = |p1-p4| = 0.6
    EAR = (0.02 + 0.02) / (2 * 0.6) = 0.0333...
    """
    return [
        (0.0, 0.3),  # p1
        (0.2, 0.31),  # p2
        (0.4, 0.31),  # p3
        (0.6, 0.3),  # p4
        (0.4, 0.29),  # p5
        (0.2, 0.29),  # p6
    ]


# ---------------------------------------------------------------------------
# Tests: compute_ear
# ---------------------------------------------------------------------------


class TestComputeEar:
    """Tests for the compute_ear function."""

    def test_open_eye_ear_above_threshold(self) -> None:
        """Open eye landmarks should produce an EAR well above 0.25."""
        ear = compute_ear(_open_eye())
        assert ear > 0.25, f"Open-eye EAR should be > 0.25, got {ear:.4f}"
        assert abs(ear - 1 / 3) < 1e-6  # exact expected value

    def test_closed_eye_ear_below_threshold(self) -> None:
        """Closed eye landmarks should produce an EAR well below 0.10."""
        ear = compute_ear(_closed_eye())
        assert ear < 0.10, f"Closed-eye EAR should be < 0.10, got {ear:.4f}"

    def test_ear_non_negative(self) -> None:
        """EAR must never be negative for any valid landmark set."""
        for pts in [_open_eye(), _closed_eye()]:
            assert compute_ear(pts) >= 0.0

    def test_wrong_number_of_landmarks_raises(self) -> None:
        """Passing other than 6 landmarks must raise ValueError."""
        with pytest.raises(ValueError, match="Expected 6 eye landmarks"):
            compute_ear([(0, 0)] * 5)
        with pytest.raises(ValueError, match="Expected 6 eye landmarks"):
            compute_ear([(0, 0)] * 7)

    def test_zero_horizontal_returns_zero(self) -> None:
        """When both eye corners are the same point, EAR should be 0."""
        same_point = [(0.5, 0.5)] * 6
        assert compute_ear(same_point) == 0.0


# ---------------------------------------------------------------------------
# Tests: DrowsinessDetector
# ---------------------------------------------------------------------------


class TestDrowsinessDetector:
    """Tests for temporal drowsiness detection."""

    def test_blink_does_not_trigger_drowsy(self) -> None:
        """A short blink (< 0.3s) must NOT be flagged as drowsy."""
        detector = DrowsinessDetector(ear_threshold=0.25, drowsy_sec=1.5)

        # Simulate ~0.2s of low EAR at 15 FPS (3 frames)
        for i in range(3):
            result = detector.update(ear=0.10, timestamp=i / 15.0)
        assert result is False, "Blink should not trigger drowsiness"

        # Eyes open again
        result = detector.update(ear=0.30, timestamp=0.25)
        assert result is False

    def test_sustained_closure_triggers_drowsy(self) -> None:
        """Sustained eye closure (> 1.5s) MUST be flagged as drowsy."""
        detector = DrowsinessDetector(ear_threshold=0.25, drowsy_sec=1.5)

        # Feed low EAR for 2 seconds at 15 FPS (30 frames)
        result = False
        for i in range(30):
            result = detector.update(ear=0.10, timestamp=i / 15.0)
        assert result is True, "Sustained 2s closure should trigger drowsiness"

    def test_exact_threshold_boundary(self) -> None:
        """EAR exactly equal to threshold should NOT count as closed."""
        detector = DrowsinessDetector(ear_threshold=0.25, drowsy_sec=1.5)

        for i in range(30):
            result = detector.update(ear=0.25, timestamp=i / 15.0)
        assert result is False, "EAR at threshold should be treated as open"

    def test_just_below_threshold_triggers(self) -> None:
        """EAR just below threshold should count as closed."""
        detector = DrowsinessDetector(ear_threshold=0.25, drowsy_sec=1.5)

        for i in range(30):
            result = detector.update(ear=0.249, timestamp=i / 15.0)
        assert result is True

    def test_recovery_resets_timer(self) -> None:
        """Opening eyes should reset the closure timer."""
        detector = DrowsinessDetector(ear_threshold=0.25, drowsy_sec=1.5)

        # Close for 1 second (not enough)
        for i in range(15):
            detector.update(ear=0.10, timestamp=i / 15.0)

        # Open briefly
        detector.update(ear=0.30, timestamp=1.1)

        # Close again for 1 second (still not enough if timer reset)
        result = False
        for i in range(15):
            result = detector.update(ear=0.10, timestamp=1.2 + i / 15.0)
        assert result is False, "Timer should have reset when eyes opened"

    def test_reset_clears_state(self) -> None:
        """Calling reset() must clear the internal closure timer."""
        detector = DrowsinessDetector(ear_threshold=0.25, drowsy_sec=1.5)

        # Start closing
        detector.update(ear=0.10, timestamp=0.0)
        detector.update(ear=0.10, timestamp=1.0)

        # Reset
        detector.reset()

        # Close again — timer should restart from now
        result = detector.update(ear=0.10, timestamp=1.1)
        assert result is False, "After reset, should need full duration again"

    def test_configurable_thresholds(self) -> None:
        """Custom thresholds should be respected."""
        # Use a very short drowsy duration
        detector = DrowsinessDetector(ear_threshold=0.20, drowsy_sec=0.5)

        # 0.22 is above the 0.20 threshold -> should not trigger
        for i in range(15):
            result = detector.update(ear=0.22, timestamp=i / 15.0)
        assert result is False

        # 0.18 is below 0.20 threshold -> trigger after 0.5s
        for i in range(15):
            result = detector.update(ear=0.18, timestamp=1.0 + i / 15.0)
        assert result is True

    def test_update_detailed_returns_state(self) -> None:
        """update_detailed should return a DrowsinessState dataclass."""
        detector = DrowsinessDetector(ear_threshold=0.25, drowsy_sec=1.5)

        state = detector.update_detailed(ear=0.30, timestamp=0.0)
        assert isinstance(state, DrowsinessState)
        assert state.is_drowsy is False
        assert state.status == "open"
        assert state.ear == 0.30


# ---------------------------------------------------------------------------
# Tests: extract_eye_landmarks
# ---------------------------------------------------------------------------


class TestExtractEyeLandmarks:
    """Tests for the extract_eye_landmarks helper."""

    def test_extracts_correct_indices(self) -> None:
        """Should pull the correct (x, y) from a 468-landmark array."""
        landmarks = np.random.rand(468, 3).astype(np.float32)

        left = extract_eye_landmarks(landmarks, LEFT_EYE_IDX)
        assert len(left) == 6
        for i, idx in enumerate(LEFT_EYE_IDX):
            assert abs(left[i][0] - float(landmarks[idx][0])) < 1e-6
            assert abs(left[i][1] - float(landmarks[idx][1])) < 1e-6

        right = extract_eye_landmarks(landmarks, RIGHT_EYE_IDX)
        assert len(right) == 6

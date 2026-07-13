"""Tests for the facial expression classifier.

Covers:
  1. _map_fer_emotions: every FER label maps to the correct Expression class
  2. _map_fer_emotions: class with highest summed probability wins
  3. classify: valid BGR crop returns an ExpressionResult
  4. classify: returns ENGAGED for happy-dominant emotions
  5. classify: returns CONFUSED for fear-dominant emotions
  6. classify: returns BORED for sad-dominant emotions
  7. classify: returns NEUTRAL for neutral-dominant emotions
  8. classify: raises ValueError on None input
  9. classify: raises ValueError on wrong-shape (grayscale) input
  10. classify: returns NEUTRAL with 0.0 confidence when no face detected
  11. classify: inference time well under 50ms with mocked detector
"""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.detection.expression import (
    _FER_TO_EXPRESSION,
    Expression,
    ExpressionClassifier,
    ExpressionResult,
    _map_fer_emotions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bgr_crop(h: int = 96, w: int = 96) -> np.ndarray:
    """Return a synthetic BGR face crop filled with a uniform colour."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _classifier_with_emotions(emotions: dict[str, float]) -> ExpressionClassifier:
    """Return an ExpressionClassifier whose FER detector is mocked."""
    clf = ExpressionClassifier()
    mock = MagicMock()
    mock.detect_emotions.return_value = [{"emotions": emotions}]
    clf._detector = mock
    return clf


# ---------------------------------------------------------------------------
# _map_fer_emotions unit tests
# ---------------------------------------------------------------------------


class TestMapFerEmotions:
    """Pure-function tests for the FER -> 4-class mapping."""

    @pytest.mark.parametrize(
        "fer_label, expected",
        [
            ("happy", Expression.ENGAGED),
            ("surprise", Expression.ENGAGED),
            ("fear", Expression.CONFUSED),
            ("disgust", Expression.CONFUSED),
            ("angry", Expression.BORED),
            ("sad", Expression.BORED),
            ("neutral", Expression.NEUTRAL),
        ],
    )
    def test_single_fer_label_maps_correctly(
        self, fer_label: str, expected: Expression
    ) -> None:
        """Each individual FER label must map to the expected Expression."""
        emotions = {label: 0.0 for label in _FER_TO_EXPRESSION}
        emotions[fer_label] = 1.0
        result, confidence = _map_fer_emotions(emotions)
        assert result == expected
        assert confidence == pytest.approx(1.0, abs=1e-6)

    def test_highest_summed_class_wins(self) -> None:
        """The class whose FER labels sum to the highest probability wins."""
        emotions = {
            "angry": 0.40,  # bored total = 0.70
            "sad": 0.30,
            "happy": 0.20,  # engaged total = 0.30
            "surprise": 0.10,
            "neutral": 0.00,
            "fear": 0.00,
            "disgust": 0.00,
        }
        result, confidence = _map_fer_emotions(emotions)
        assert result == Expression.BORED
        assert confidence == pytest.approx(0.70, abs=1e-6)


# ---------------------------------------------------------------------------
# ExpressionClassifier tests
# ---------------------------------------------------------------------------


class TestExpressionClassifier:
    """Integration tests using a mocked FER detector."""

    def test_classify_returns_expression_result(self) -> None:
        """classify() must return an ExpressionResult instance."""
        emotions = {k: 1 / 7 for k in _FER_TO_EXPRESSION}
        clf = _classifier_with_emotions(emotions)
        result = clf.classify(_bgr_crop())
        assert isinstance(result, ExpressionResult)
        assert isinstance(result.expression, Expression)
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_engaged(self) -> None:
        """Happy + surprise dominant → ENGAGED."""
        emotions = {
            "happy": 0.60,
            "surprise": 0.15,
            "neutral": 0.10,
            "sad": 0.05,
            "angry": 0.05,
            "fear": 0.03,
            "disgust": 0.02,
        }
        clf = _classifier_with_emotions(emotions)
        assert clf.classify(_bgr_crop()).expression == Expression.ENGAGED

    def test_classify_confused(self) -> None:
        """Fear + disgust dominant → CONFUSED."""
        emotions = {
            "fear": 0.45,
            "disgust": 0.30,
            "neutral": 0.10,
            "happy": 0.05,
            "sad": 0.05,
            "angry": 0.03,
            "surprise": 0.02,
        }
        clf = _classifier_with_emotions(emotions)
        assert clf.classify(_bgr_crop()).expression == Expression.CONFUSED

    def test_classify_bored(self) -> None:
        """Sad + angry dominant → BORED."""
        emotions = {
            "sad": 0.50,
            "angry": 0.25,
            "neutral": 0.10,
            "happy": 0.05,
            "surprise": 0.05,
            "fear": 0.03,
            "disgust": 0.02,
        }
        clf = _classifier_with_emotions(emotions)
        assert clf.classify(_bgr_crop()).expression == Expression.BORED

    def test_classify_neutral(self) -> None:
        """Neutral dominant → NEUTRAL."""
        emotions = {
            "neutral": 0.80,
            "happy": 0.05,
            "sad": 0.05,
            "angry": 0.04,
            "fear": 0.03,
            "disgust": 0.02,
            "surprise": 0.01,
        }
        clf = _classifier_with_emotions(emotions)
        assert clf.classify(_bgr_crop()).expression == Expression.NEUTRAL

    def test_classify_raises_on_none(self) -> None:
        """classify() must raise ValueError when face_crop is None."""
        clf = ExpressionClassifier()
        clf._detector = MagicMock()
        with pytest.raises(ValueError, match="None"):
            clf.classify(None)  # type: ignore[arg-type]

    def test_classify_raises_on_wrong_shape(self) -> None:
        """classify() must raise ValueError for a non-3-channel array."""
        clf = ExpressionClassifier()
        clf._detector = MagicMock()
        gray = np.zeros((48, 48), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            clf.classify(gray)  # type: ignore[arg-type]

    def test_classify_neutral_when_no_face_detected(self) -> None:
        """When FER returns no faces, classify() must return NEUTRAL with 0.0."""
        clf = ExpressionClassifier()
        mock = MagicMock()
        mock.detect_emotions.return_value = []
        clf._detector = mock
        result = clf.classify(_bgr_crop())
        assert result.expression == Expression.NEUTRAL
        assert result.confidence == pytest.approx(0.0)

    def test_inference_time_under_50ms(self) -> None:
        """Inference (with mocked detector) must complete under 50ms per frame."""
        emotions = {
            "happy": 0.50,
            "surprise": 0.20,
            "neutral": 0.10,
            "sad": 0.08,
            "angry": 0.06,
            "fear": 0.04,
            "disgust": 0.02,
        }
        clf = _classifier_with_emotions(emotions)
        crop = _bgr_crop(96, 96)

        # Warm up
        clf.classify(crop)

        t0 = time.perf_counter()
        for _ in range(20):
            clf.classify(crop)
        elapsed_ms = (time.perf_counter() - t0) * 1000 / 20

        assert (
            elapsed_ms < 50.0
        ), f"Mean inference {elapsed_ms:.1f}ms exceeds 50ms target"

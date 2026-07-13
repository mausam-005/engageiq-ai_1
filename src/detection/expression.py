"""Facial expression classifier for EngageIQ.

Classifies a face crop into one of four pedagogically meaningful classes:
  - engaged  (FER: happy, surprise)
  - confused (FER: fear, disgust)
  - bored    (FER: sad, angry)
  - neutral  (FER: neutral)

Uses the FER library (pre-trained on FER2013) as the inference backend.
Inference target: < 50ms per frame on CPU.

Usage::

    from src.detection.expression import ExpressionClassifier, Expression
    import cv2

    classifier = ExpressionClassifier()
    face_crop = cv2.imread('tests/fixtures/sample_face.jpg')
    result = classifier.classify(face_crop)
    print(f'Expression: {result.expression}, Confidence: {result.confidence:.2f}')
"""

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class Expression(str, Enum):
    """Four engagement-relevant expression classes."""

    ENGAGED = "engaged"
    CONFUSED = "confused"
    BORED = "bored"
    NEUTRAL = "neutral"


@dataclass
class ExpressionResult:
    """Result of an expression classification."""

    expression: Expression
    confidence: float  # summed class probability in [0, 1]


# Mapping from FER's 7-class output to EngageIQ's 4-class output.
# FER labels: angry, disgust, fear, happy, neutral, sad, surprise
_FER_TO_EXPRESSION: dict[str, Expression] = {
    "happy": Expression.ENGAGED,
    "surprise": Expression.ENGAGED,
    "fear": Expression.CONFUSED,
    "disgust": Expression.CONFUSED,
    "angry": Expression.BORED,
    "sad": Expression.BORED,
    "neutral": Expression.NEUTRAL,
}


def _map_fer_emotions(emotions: dict[str, float]) -> tuple[Expression, float]:
    """Aggregate raw FER probabilities into our 4-class output.

    For each of the 4 output classes, we sum the probabilities of the FER
    emotions that map to it. The class with the highest summed probability wins.

    Args:
        emotions: Dict mapping FER emotion name -> probability (sums to ~1.0).

    Returns:
        Tuple of (Expression, confidence) where confidence is the summed
        probability for the winning class.
    """
    class_scores: dict[Expression, float] = {e: 0.0 for e in Expression}
    for fer_label, prob in emotions.items():
        target = _FER_TO_EXPRESSION.get(fer_label)
        if target is not None:
            class_scores[target] += prob

    best = max(class_scores, key=lambda e: class_scores[e])
    return best, float(class_scores[best])


class ExpressionClassifier:
    """Classifies facial expressions from BGR face images.

    Uses the FER library (TFLite backend, pre-trained on FER2013) for
    inference and maps the 7-class output to EngageIQ's 4 classes.

    The FER detector is lazily initialised on first call to ``classify``
    to avoid loading TensorFlow at import time.

    Example::

        classifier = ExpressionClassifier()
        result = classifier.classify(face_crop_bgr)
        print(result.expression, result.confidence)
    """

    def __init__(self) -> None:
        self._detector = None

    def _load(self) -> None:
        """Lazily load the FER detector (TFLite, no MTCNN)."""
        if self._detector is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from fer.fer import FER  # type: ignore[import]

            # mtcnn=False: use OpenCV Haar cascade (faster, good enough for
            # pre-cropped faces from MediaPipe).
            self._detector = FER(mtcnn=False)

    def classify(self, face_crop: np.ndarray) -> ExpressionResult:
        """Classify the expression from a BGR face crop.

        Args:
            face_crop: BGR face image as a numpy array with shape (H, W, 3).

        Returns:
            ExpressionResult with .expression and .confidence in [0, 1].
            Returns ExpressionResult(NEUTRAL, 0.0) when no face is detected
            in the crop (e.g., too small or poorly lit).

        Raises:
            ValueError: If face_crop is None or has invalid shape.
        """
        if face_crop is None:
            raise ValueError("face_crop must not be None")
        if face_crop.ndim != 3 or face_crop.shape[2] != 3:
            raise ValueError(
                f"face_crop must be a BGR image with shape (H, W, 3), "
                f"got {face_crop.shape}"
            )

        self._load()

        results = self._detector.detect_emotions(face_crop)  # type: ignore[union-attr]

        if not results:
            # No face detected — return safe neutral default with low confidence.
            return ExpressionResult(expression=Expression.NEUTRAL, confidence=0.0)

        emotions: dict[str, float] = results[0]["emotions"]
        expression, confidence = _map_fer_emotions(emotions)
        return ExpressionResult(expression=expression, confidence=confidence)


# ---------------------------------------------------------------------------
# Module-level __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expression Classifier Demo — press 'q' to quit."
    )
    parser.add_argument("--demo", action="store_true", help="Run live webcam demo")
    args = parser.parse_args()

    if not args.demo:
        print("Run with --demo to open the webcam and see expression labels.")
        sys.exit(0)

    try:
        from src.detection.face_mesh import FaceMeshDetector
    except ImportError:
        print("FaceMeshDetector not available. Make sure M2 code is present.")
        sys.exit(1)

    classifier = ExpressionClassifier()
    face_detector = FaceMeshDetector()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    _COLORS = {
        Expression.ENGAGED: (0, 220, 0),
        Expression.CONFUSED: (0, 180, 255),
        Expression.BORED: (0, 80, 200),
        Expression.NEUTRAL: (200, 200, 200),
    }

    print("Webcam demo started. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_detector.process(frame)
        if faces:
            face = faces[0]
            bbox = face["bbox"]
            x_min, y_min, x_max, y_max = (int(v) for v in bbox)

            h, w = frame.shape[:2]
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            face_crop = frame[y_min:y_max, x_min:x_max]

            if face_crop.size > 0:
                t0 = time.perf_counter()
                result = classifier.classify(face_crop)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                color = _COLORS[result.expression]
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(
                    frame,
                    f"{result.expression.value} ({result.confidence:.0%})  "
                    f"{elapsed_ms:.0f}ms",
                    (x_min, max(20, y_min - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

        cv2.imshow("Expression Classifier Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

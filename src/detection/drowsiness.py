"""EAR-based drowsiness detection with temporal smoothing.

This module provides the Eye Aspect Ratio (EAR) computation and a
temporal drowsiness detector that distinguishes normal blinks from
sustained eye closure indicating drowsiness.

MediaPipe Face Mesh eye landmark indices used:
    Right eye: [33, 160, 158, 133, 153, 144]
    Left eye:  [362, 385, 387, 263, 373, 380]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass

import numpy as np

# MediaPipe Face Mesh landmark indices for each eye.
# Layout:
#     p2  p3
#  p1        p4   (p1, p4 = eye corners)
#     p6  p5
RIGHT_EYE_IDX: list[int] = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX: list[int] = [362, 385, 387, 263, 373, 380]

# Default thresholds
EAR_THRESHOLD: float = 0.25
DROWSY_SECONDS: float = 1.5
BLINK_SECONDS: float = 0.3


def _euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Compute 2-D Euclidean distance between two points.

    Args:
        p1: First point as (x, y).
        p2: Second point as (x, y).

    Returns:
        Euclidean distance as a float.
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_ear(eye_landmarks: list[tuple[float, float]]) -> float:
    """Compute the Eye Aspect Ratio from 6 eye landmarks.

    The landmarks must follow this layout:

        p2  p3
     p1        p4
        p6  p5

    Where p1 and p4 are the eye corners, p2/p3 are the upper lid,
    and p5/p6 are the lower lid.

    Formula:

        EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

    Args:
        eye_landmarks: Exactly 6 ``(x, y)`` tuples ordered as
            ``[p1, p2, p3, p4, p5, p6]``.

    Returns:
        EAR value as a float.  Typically 0.25-0.35 when open,
        approaching 0.0 when fully closed.

    Raises:
        ValueError: If *eye_landmarks* does not contain exactly 6 points.
    """
    if len(eye_landmarks) != 6:
        raise ValueError(f"Expected 6 eye landmarks, got {len(eye_landmarks)}")

    p1, p2, p3, p4, p5, p6 = eye_landmarks

    # Vertical distances (upper-lid to lower-lid)
    vertical_a = _euclidean(p2, p6)
    vertical_b = _euclidean(p3, p5)

    # Horizontal distance (corner to corner)
    horizontal = _euclidean(p1, p4)

    if horizontal == 0.0:
        return 0.0

    ear = (vertical_a + vertical_b) / (2.0 * horizontal)
    return ear


def extract_eye_landmarks(
    landmarks: np.ndarray,
    eye_indices: list[int],
) -> list[tuple[float, float]]:
    """Extract (x, y) tuples for one eye from the full 468-landmark array.

    Args:
        landmarks: NumPy array of shape ``(468, 3)`` with normalized
            coordinates from MediaPipe Face Mesh.
        eye_indices: Six landmark indices for the eye.

    Returns:
        A list of 6 ``(x, y)`` tuples.
    """
    return [(float(landmarks[i][0]), float(landmarks[i][1])) for i in eye_indices]


@dataclass
class DrowsinessState:
    """Internal state emitted by :class:`DrowsinessDetector`."""

    is_drowsy: bool = False
    is_blink: bool = False
    ear: float = 0.0
    closure_duration: float = 0.0
    status: str = "open"  # open | closing | blink | DROWSY


class DrowsinessDetector:
    """Detects drowsiness from sustained eye closure over time.

    Normal blinks (< ``blink_sec`` seconds) are filtered out.
    Only closures lasting longer than ``drowsy_sec`` seconds trigger
    a drowsy signal.
    """

    def __init__(
        self,
        ear_threshold: float = EAR_THRESHOLD,
        drowsy_sec: float = DROWSY_SECONDS,
        blink_sec: float = BLINK_SECONDS,
        drowsy_frames: int | None = None,
        drowsy_duration: float | None = None,
    ) -> None:
        """Initialize the drowsiness detector.

        Args:
            ear_threshold: EAR value below which the eye is considered
                closed.  Default 0.25.
            drowsy_sec: Seconds of sustained closure required to flag
                drowsiness.  Default 1.5.
            blink_sec: Maximum closure duration (in seconds) that counts
                as a blink rather than drowsiness.  Default 0.3.
            drowsy_frames: Deprecated convenience alias.  If provided,
                it is ignored in favour of *drowsy_sec*.
            drowsy_duration: Deprecated alias for *drowsy_sec*.  If provided,
                it overrides *drowsy_sec*.
        """
        self.ear_threshold = ear_threshold
        self.drowsy_sec = drowsy_duration if drowsy_duration is not None else drowsy_sec
        self.blink_sec = blink_sec

        self._closed_since: float | None = None
        self._last_state: str = "open"

    def update(self, ear: float, timestamp: float) -> bool:
        """Feed a new EAR reading and return whether drowsiness is detected.

        Args:
            ear: Current Eye Aspect Ratio (average of both eyes recommended).
            timestamp: Current epoch time in seconds.

        Returns:
            ``True`` if the eye has been closed longer than *drowsy_sec*
            seconds (drowsy), ``False`` otherwise.
        """
        if ear < self.ear_threshold:
            # Eye is below threshold (closing / closed).
            if self._closed_since is None:
                self._closed_since = timestamp

            duration = timestamp - self._closed_since
            if duration > self.drowsy_sec:
                self._last_state = "DROWSY"
                return True

            self._last_state = "closing"
            return False

        # Eye is open again — evaluate how long it was closed.
        if self._closed_since is not None:
            duration = timestamp - self._closed_since
            self._closed_since = None
            if duration < self.blink_sec:
                self._last_state = "blink"
            else:
                self._last_state = "open"
        else:
            self._last_state = "open"

        return False

    def update_detailed(self, ear: float, timestamp: float) -> DrowsinessState:
        """Like :meth:`update` but returns a rich :class:`DrowsinessState`.

        Args:
            ear: Current Eye Aspect Ratio.
            timestamp: Current epoch time in seconds.

        Returns:
            A :class:`DrowsinessState` with fields for drowsy flag,
            blink detection, EAR, closure duration, and status string.
        """
        is_drowsy = self.update(ear, timestamp)
        closure_duration = 0.0
        if self._closed_since is not None:
            closure_duration = timestamp - self._closed_since

        return DrowsinessState(
            is_drowsy=is_drowsy,
            is_blink=(self._last_state == "blink"),
            ear=ear,
            closure_duration=closure_duration,
            status=self._last_state,
        )

    def reset(self) -> None:
        """Reset internal temporal state."""
        self._closed_since = None
        self._last_state = "open"


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------


def run_demo(camera_index: int = 0) -> None:
    """Open the webcam and display real-time EAR and drowsiness status.

    Press **q** to quit.

    Args:
        camera_index: Index of the webcam device.
    """
    import cv2

    from src.detection.face_mesh import FaceMeshDetector

    detector = FaceMeshDetector(max_faces=1)
    drowsiness = DrowsinessDetector()
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: cannot open webcam", file=sys.stderr)
        sys.exit(1)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            result = detector.detect(frame)
            now = time.time()

            if result.faces:
                lm = result.faces[0].landmarks
                left_eye = extract_eye_landmarks(lm, LEFT_EYE_IDX)
                right_eye = extract_eye_landmarks(lm, RIGHT_EYE_IDX)
                ear = (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0

                state = drowsiness.update_detailed(ear, now)

                # Draw eye landmarks
                h, w = frame.shape[:2]
                for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                    x = int(lm[idx][0] * w)
                    y = int(lm[idx][1] * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Overlay text
                color = (0, 0, 255) if state.is_drowsy else (0, 255, 0)
                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}  |  {state.status}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )
                if state.is_drowsy:
                    cv2.putText(
                        frame,
                        "*** DROWSY ***",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3,
                    )
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Drowsiness Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


def main() -> None:
    """CLI entry point for the drowsiness module."""
    parser = argparse.ArgumentParser(description="EAR-based drowsiness detection demo")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the live webcam demo",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        print("Run with --demo to start the webcam drowsiness demo")


if __name__ == "__main__":
    main()

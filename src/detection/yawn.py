import argparse
import sys
import time
from typing import List

import cv2
import numpy as np

from src.config.settings import settings


def compute_mar(mouth_landmarks: np.ndarray) -> float:
    """
    Calculate the Mouth Aspect Ratio (MAR).
    If a 4-point array is passed, it assumes the order: [left, right, top, bottom].
    If a full 468-point face mesh array is passed, it extracts standard inner lip landmarks.
    """
    if mouth_landmarks is None:
        return 0.0

    if len(mouth_landmarks) == 4:
        # Expected manual format for testing: [left, right, top, bottom]
        left = mouth_landmarks[0]
        right = mouth_landmarks[1]
        top = mouth_landmarks[2]
        bottom = mouth_landmarks[3]
    elif len(mouth_landmarks) >= 468:
        # Standard MediaPipe inner lip indices
        left = mouth_landmarks[78]
        right = mouth_landmarks[308]
        top = mouth_landmarks[13]
        bottom = mouth_landmarks[14]
    else:
        return 0.0

    horizontal_dist = np.linalg.norm(left - right)
    vertical_dist = np.linalg.norm(top - bottom)

    if horizontal_dist == 0:
        return 0.0

    return float(vertical_dist / horizontal_dist)


class YawnDetector:
    """Detects yawns based on the Mouth Aspect Ratio (MAR) over time."""

    def __init__(self, mar_threshold: float = None, yawn_duration: float = None):
        self.mar_threshold = (
            mar_threshold if mar_threshold is not None else settings.yawn_mar_threshold
        )
        self.yawn_duration = (
            yawn_duration
            if yawn_duration is not None
            else settings.yawn_duration_seconds
        )
        self.yawn_start_time = None
        self.is_yawning = False
        self.yawn_timestamps: List[float] = []

    def update(self, mar: float, current_time: float) -> bool:
        """
        Updates the internal state with a new MAR reading.
        Returns True if a sustained yawn is currently happening.
        """
        if mar > self.mar_threshold:
            if self.yawn_start_time is None:
                self.yawn_start_time = current_time
            elif (current_time - self.yawn_start_time) >= self.yawn_duration:
                if not self.is_yawning:
                    self.is_yawning = True
                    self.record_yawn(current_time)
        else:
            self.yawn_start_time = None
            self.is_yawning = False

        return self.is_yawning

    def record_yawn(self, timestamp: float):
        """Records a yawn at the specified timestamp."""
        self.yawn_timestamps.append(timestamp)

    def is_fatigued(self, current_time: float, ear_data=None) -> bool:
        """
        Assesses fatigue by checking if there were 3 or more yawns in the last 10 minutes.
        (Optionally combined with EAR data upstream).
        """
        window_start = current_time - (settings.yawn_fatigue_window_minutes * 60)
        recent_yawns = [t for t in self.yawn_timestamps if t >= window_start]
        return len(recent_yawns) >= settings.yawn_fatigue_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yawn Detector Demo")
    parser.add_argument("--demo", action="store_true", help="Run webcam demo")
    args = parser.parse_args()

    if not args.demo:
        print("Run with --demo to see the live webcam feed.")
        sys.exit(0)

    try:
        from src.detection.face_mesh import FaceMeshDetector
    except ImportError:
        print("Missing FaceMeshDetector module. Demo cannot run.")
        sys.exit(1)

    cap = cv2.VideoCapture(settings.webcam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = FaceMeshDetector()
    yawn_detector = YawnDetector()

    print("Starting webcam demo. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        faces = detector.process(frame)
        if faces:
            face = faces[0]
            landmarks = face["landmarks"]

            mar = compute_mar(landmarks)
            current_time = time.time()
            is_yawning = yawn_detector.update(mar, current_time)

            bbox = face["bbox"]
            x_min, y_min, x_max, y_max = bbox

            color = (0, 0, 255) if is_yawning else (0, 255, 0)

            cv2.rectangle(
                frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2
            )
            cv2.putText(
                frame,
                f"MAR: {mar:.2f}",
                (int(x_min), int(max(20, y_min - 10))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            if is_yawning:
                cv2.putText(
                    frame,
                    "YAWN DETECTED",
                    (int(x_min), int(max(45, y_min + 15))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            if yawn_detector.is_fatigued(current_time):
                cv2.putText(
                    frame,
                    "FATIGUE DETECTED",
                    (int(x_min), int(max(70, y_min + 40))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        cv2.imshow("Yawn Detector Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

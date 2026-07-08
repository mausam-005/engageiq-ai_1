"""Gaze direction classifier based on head pose and eye landmarks."""

import argparse
import sys
from enum import Enum

import cv2
import numpy as np

from src.config.settings import settings


class GazeState(str, Enum):
    AT_SCREEN = "at_screen"
    AWAY_LEFT = "away_left"
    AWAY_RIGHT = "away_right"
    LOOKING_DOWN = "looking_down"
    EYES_CLOSED = "eyes_closed"


def calculate_ear(eye_landmarks: np.ndarray) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR).
    Uses the 6 standard eye landmarks.
    """
    if eye_landmarks is None or len(eye_landmarks) < 6:
        return 1.0

    # Vertical eye landmarks
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # Horizontal eye landmarks
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    if h == 0:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return float(ear)


def calculate_iris_ratio(eye_landmarks: np.ndarray, iris_center: np.ndarray) -> float:
    """
    Calculate the Iris Position Ratio.
    Returns the ratio of the distance from the inner corner to the iris center,
    over the total width of the eye.
    Assumes eye_landmarks[0] is the leftmost point and eye_landmarks[3] is rightmost point.
    """
    if eye_landmarks is None or len(eye_landmarks) < 4 or iris_center is None:
        return 0.5

    # Find the leftmost and rightmost points of the eye contour
    x_coords = eye_landmarks[:, 0]
    leftmost = np.min(x_coords)
    rightmost = np.max(x_coords)

    eye_width = rightmost - leftmost
    if eye_width == 0:
        return 0.5

    # Ratio = distance from left corner / total width
    ratio = (iris_center[0] - leftmost) / eye_width

    # Clip to [0, 1]
    return float(np.clip(ratio, 0.0, 1.0))


def classify_gaze(
    pitch: float, yaw: float, iris_ratio: float, ear: float = 1.0
) -> GazeState:
    """Classify gaze direction from head pose and eye state.

    Args:
        pitch: Head pitch in degrees (negative = looking down).
        yaw: Head yaw in degrees (positive = looking right).
        iris_ratio: Iris position ratio (distance from inner corner to iris center / total width).
        ear: Eye Aspect Ratio (below threshold = eyes closed).

    Returns:
        GazeState
    """
    if ear < settings.gaze_ear_closed_threshold:
        return GazeState.EYES_CLOSED

    if pitch < settings.gaze_pitch_down_threshold_deg:
        return GazeState.LOOKING_DOWN

    if (
        yaw < -settings.gaze_yaw_threshold_deg
        or iris_ratio < settings.gaze_iris_left_threshold
    ):
        return GazeState.AWAY_LEFT

    if (
        yaw > settings.gaze_yaw_threshold_deg
        or iris_ratio > settings.gaze_iris_right_threshold
    ):
        return GazeState.AWAY_RIGHT

    return GazeState.AT_SCREEN


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaze Classifier Demo")
    parser.add_argument("--demo", action="store_true", help="Run webcam demo")
    args = parser.parse_args()

    if not args.demo:
        print("Run with --demo to see the live webcam feed.")
        sys.exit(0)

    try:
        from src.detection.face_mesh import FaceMeshDetector
        from src.detection.head_pose import estimate_head_pose
    except ImportError:
        print("Missing FaceMeshDetector or head_pose modules. Demo cannot run.")
        sys.exit(1)

    cap = cv2.VideoCapture(settings.webcam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = FaceMeshDetector()

    colors = {
        GazeState.AT_SCREEN: (0, 255, 0),  # Green
        GazeState.AWAY_LEFT: (0, 0, 255),  # Red
        GazeState.AWAY_RIGHT: (0, 0, 255),  # Red
        GazeState.LOOKING_DOWN: (0, 255, 255),  # Yellow
        GazeState.EYES_CLOSED: (128, 128, 128),  # Gray
    }

    print("Starting webcam demo. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        faces = detector.process(frame)
        if faces:
            # Just grab the first face for the demo
            face = faces[0]
            landmarks = face["landmarks"]

            pitch, yaw, roll = estimate_head_pose(landmarks)

            # Simple mock calculations for the demo if real indices aren't known
            # In a real app we'd extract the actual left/right eye indices.
            # Using 0.5 for EAR and iris ratio to let head pose drive the demo.
            ear = 0.5
            iris_ratio = 0.5

            state = classify_gaze(pitch, yaw, iris_ratio, ear)
            color = colors.get(state, (255, 255, 255))

            bbox = face["bbox"]
            x_min, y_min, x_max, y_max = bbox

            # Draw color-coded bounding box
            cv2.rectangle(
                frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2
            )

            # Put state text
            cv2.putText(
                frame,
                f"State: {state.value}",
                (int(x_min), int(max(20, y_min - 10))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"P:{pitch:.1f} Y:{yaw:.1f}",
                (int(x_min), int(max(45, y_min + 15))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        cv2.imshow("Gaze Classifier Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

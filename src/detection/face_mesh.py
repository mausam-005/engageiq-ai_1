"""MediaPipe Face Mesh integration for 468-landmark face detection."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class FaceLandmarks:
    """A single detected face with 468 landmarks and a confidence score."""

    landmarks: np.ndarray
    confidence: float
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass
class FaceMeshResult:
    """The complete face-mesh detection result for a frame."""

    faces: list[FaceLandmarks]
    frame_shape: tuple[int, int] = (0, 0)
    timestamp: float = 0.0


class FaceMeshDetector:
    """Detects 468 facial landmarks using MediaPipe Face Mesh."""

    def __init__(self, max_faces: int = 1, min_detection_confidence: float = 0.5):
        """Create a detector with lazy MediaPipe initialization.

        Args:
            max_faces: Maximum number of faces to detect per frame.
            min_detection_confidence: Minimum confidence threshold for face
                detection.
        """
        self.max_faces = max_faces
        self.min_detection_confidence = min_detection_confidence
        self._mesh: Any | None = None
        self._use_tasks_api: bool = False

    def _init_mesh(self) -> None:
        """Initialize MediaPipe Face Mesh when first needed.

        Supports both legacy mp.solutions API (mediapipe <= 0.10.18) and
        the newer mediapipe versions (>= 0.10.20+) that dropped solutions.
        """
        if self._mesh is not None:
            return

        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            # Legacy API: mediapipe <= 0.10.18
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=self.max_faces,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=0.5,
                refine_landmarks=True,
            )
        else:
            # New Tasks API: mediapipe >= 0.10.20 (mp.solutions removed)
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            base_options = mp_python.BaseOptions(
                model_asset_path=None,  # uses bundled model
            )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=self.max_faces,
                min_face_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=0.5,
            )
            self._mesh = vision.FaceLandmarker.create_from_options(options)
            self._use_tasks_api = True
            return

        self._use_tasks_api = False

    def _coerce_landmarks(self, face: Any) -> list[Any]:
        """Extract a landmark sequence from either a MediaPipe face object or a test stub.

        Args:
            face: Face object or iterable containing landmark-like objects.

        Returns:
            A list of landmark objects.
        """
        if face is None:
            return []
        if hasattr(face, "landmark"):
            return list(face.landmark)
        if isinstance(face, (list, tuple)):
            return list(face)
        return []

    def _landmarks_to_array(self, landmarks: list[Any]) -> np.ndarray:
        """Convert landmark objects into a fixed-size NumPy array.

        Args:
            landmarks: Landmark objects from MediaPipe or a stub.

        Returns:
            A NumPy array with shape (468, 3).
        """
        if not landmarks:
            return np.zeros((468, 3), dtype=np.float32)

        coords = np.array(
            [
                [
                    getattr(point, "x", 0.0),
                    getattr(point, "y", 0.0),
                    getattr(point, "z", 0.0),
                ]
                for point in landmarks
            ],
            dtype=np.float32,
        )
        if coords.shape[0] >= 468:
            return coords[:468]

        padding = np.zeros((468 - coords.shape[0], 3), dtype=np.float32)
        return np.vstack([coords, padding])

    def _compute_bbox(
        self, landmarks: np.ndarray, frame_shape: tuple[int, int]
    ) -> tuple[float, float, float, float]:
        """Compute a pixel-space bounding box from normalized landmarks.

        Args:
            landmarks: Landmark coordinates with shape (468, 3) or similar.
            frame_shape: Frame shape as (width, height).

        Returns:
            A tuple of (x, y, width, height) pixel coordinates.
        """
        if landmarks.size == 0:
            return (0.0, 0.0, 0.0, 0.0)

        width, height = frame_shape
        if width <= 0 or height <= 0:
            return (0.0, 0.0, 0.0, 0.0)

        xs = landmarks[:, 0]
        ys = landmarks[:, 1]
        x_min = float(np.min(xs)) * width
        y_min = float(np.min(ys)) * height
        x_max = float(np.max(xs)) * width
        y_max = float(np.max(ys)) * height
        return (
            float(x_min),
            float(y_min),
            float(max(0.0, x_max - x_min)),
            float(max(0.0, y_max - y_min)),
        )

    def detect(self, frame: np.ndarray) -> FaceMeshResult:
        """Detect face landmarks in a BGR frame.

        Args:
            frame: BGR image as a NumPy array.

        Returns:
            A FaceMeshResult containing detected faces. Empty results are
            returned for no-face or invalid frames.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return FaceMeshResult(faces=[], frame_shape=(0, 0), timestamp=time.time())

        if frame.ndim != 3 or frame.shape[2] != 3:
            return FaceMeshResult(
                faces=[],
                frame_shape=(frame.shape[1], frame.shape[0]),
                timestamp=time.time(),
            )

        frame_shape = (frame.shape[1], frame.shape[0])
        self._init_mesh()
        if self._mesh is None or not hasattr(self._mesh, "process"):
            return FaceMeshResult(
                faces=[], frame_shape=frame_shape, timestamp=time.time()
            )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb_frame)

        if (
            not hasattr(results, "multi_face_landmarks")
            or not results.multi_face_landmarks
        ):
            return FaceMeshResult(
                faces=[], frame_shape=frame_shape, timestamp=time.time()
            )

        faces: list[FaceLandmarks] = []
        for face in results.multi_face_landmarks[: self.max_faces]:
            landmark_sequence = self._coerce_landmarks(face)
            landmarks_array = self._landmarks_to_array(landmark_sequence)
            confidence = 1.0 if landmarks_array.shape[0] > 0 else 0.0
            bbox = self._compute_bbox(landmarks_array, frame_shape)
            faces.append(
                FaceLandmarks(
                    landmarks=landmarks_array,
                    confidence=confidence,
                    bbox=bbox,
                )
            )

        return FaceMeshResult(
            faces=faces,
            frame_shape=frame_shape,
            timestamp=time.time(),
        )

    def close(self) -> None:
        """Release MediaPipe resources if they have been initialized."""
        if self._mesh is not None:
            self._mesh.close()
            self._mesh = None


def run_demo(camera_index: int = 0, window_name: str = "Face Mesh Demo") -> None:
    """Open the webcam and overlay detected landmarks in a simple demo window.

    Args:
        camera_index: Index of the webcam device to open.
        window_name: Title of the OpenCV window.
    """
    detector = FaceMeshDetector(max_faces=1)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam for face mesh demo")

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                continue

            result = detector.detect(frame)
            if result.faces:
                for landmark in result.faces[0].landmarks:
                    x = int(landmark[0] * frame.shape[1])
                    y = int(landmark[1] * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


def main() -> None:
    """CLI entry point for the face mesh module."""
    parser = argparse.ArgumentParser(description="MediaPipe face mesh demo")
    parser.add_argument("--demo", action="store_true", help="Run the webcam demo")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        print("Run with --demo to start the webcam demo")


if __name__ == "__main__":
    main()

"""Head pose estimation using MediaPipe landmarks + OpenCV solvePnP."""

from __future__ import annotations

import argparse
from typing import Any

import cv2
import numpy as np

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

LANDMARK_INDICES = (1, 199, 33, 263, 61, 291)


def _resolve_frame_size(frame_shape: tuple[int, ...]) -> tuple[int, int]:
    """Normalize frame shape into a width/height tuple.

    Args:
        frame_shape: Either a 2-tuple of (width, height) or a 3-tuple of
            (height, width, channels).

    Returns:
        A tuple of (width, height).
    """
    if len(frame_shape) == 3:
        height, width = frame_shape[0], frame_shape[1]
    else:
        width, height = frame_shape[0], frame_shape[1]
    return int(width), int(height)


def _extract_landmarks(landmarks: Any) -> np.ndarray | None:
    """Convert landmarks into a NumPy array with shape (N, 3).

    Args:
        landmarks: Landmark values from MediaPipe or a simple list.

    Returns:
        A NumPy array containing the landmark coordinates, or None if the
        input is invalid.
    """
    if landmarks is None:
        return None

    if isinstance(landmarks, np.ndarray):
        array = np.asarray(landmarks, dtype=np.float64)
        if array.ndim == 1 and array.size >= 18:
            array = array.reshape(-1, 3)
        if array.ndim == 2 and array.shape[1] >= 3:
            return array[:, :3]
        return None

    if isinstance(landmarks, (list, tuple)):
        points: list[np.ndarray] = []
        for point in landmarks:
            if isinstance(point, np.ndarray):
                points.append(np.asarray(point[:3], dtype=np.float64))
            elif isinstance(point, (list, tuple)):
                points.append(np.asarray(point[:3], dtype=np.float64))
            elif hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
                points.append(
                    np.array(
                        [float(point.x), float(point.y), float(point.z)],
                        dtype=np.float64,
                    )
                )
            else:
                return None
        if not points:
            return None
        return np.asarray(points, dtype=np.float64)

    return None


def _build_camera_matrix(width: int, height: int) -> np.ndarray:
    """Create a standard intrinsic camera matrix from frame dimensions."""
    focal_length = float(max(width, height))
    center_x = width / 2.0
    center_y = height / 2.0
    return np.array(
        [
            [focal_length, 0.0, center_x],
            [0.0, focal_length, center_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_head_pose(
    landmarks: Any, frame_shape: tuple[int, ...]
) -> tuple[float, float, float] | None:
    """Estimate head pose from facial landmarks using solvePnP.

    Args:
        landmarks: Landmark coordinates from FaceMeshDetector or a compatible
            sequence.
        frame_shape: Frame shape as (width, height) or (height, width, channels).

    Returns:
        A tuple of (pitch, yaw, roll) in degrees, or None if pose estimation
        is not possible.
    """
    landmark_array = _extract_landmarks(landmarks)
    if landmark_array is None or landmark_array.shape[0] < max(LANDMARK_INDICES) + 1:
        return None

    width, height = _resolve_frame_size(frame_shape)
    if width <= 0 or height <= 0:
        return None

    image_points: list[list[float]] = []
    for index in LANDMARK_INDICES:
        if index >= landmark_array.shape[0]:
            return None

        point = landmark_array[index]
        if point.size < 3:
            return None

        image_points.append(
            [
                float(point[0]) * width,
                float(point[1]) * height,
            ]
        )

    if len(image_points) != len(LANDMARK_INDICES):
        return None

    camera_matrix = _build_camera_matrix(width, height)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    image_points_array = np.asarray(image_points, dtype=np.float64)

    try:
        success, rotation_vector, _ = cv2.solvePnP(
            MODEL_POINTS,
            image_points_array,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None

    if not success or rotation_vector is None:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    try:
        angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    except cv2.error:
        return None

    return float(angles[0]), float(-angles[1]), float(angles[2])


def _draw_pose_axes(
    frame: np.ndarray, landmarks: np.ndarray, pose: tuple[float, float, float]
) -> None:
    """Draw simple 3D axes from the nose tip for the demo view."""
    if landmarks.shape[0] <= 1:
        return

    nose = landmarks[1]
    nose_x = int(float(nose[0]) * frame.shape[1])
    nose_y = int(float(nose[1]) * frame.shape[0])

    pitch, yaw, roll = pose
    rotation_vector = np.deg2rad(np.array([pitch, yaw, roll], dtype=np.float64))
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    axes = np.array(
        [
            [80.0, 0.0, 0.0],
            [0.0, 80.0, 0.0],
            [0.0, 0.0, 80.0],
        ],
        dtype=np.float64,
    )
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    for axis, color in zip(axes, colors):
        direction = rotation_matrix @ axis
        end_point = (
            int(nose_x + direction[0]),
            int(nose_y - direction[1]),
        )
        cv2.line(frame, (nose_x, nose_y), end_point, color, 2)


def run_demo(camera_index: int = 0, window_name: str = "Head Pose Demo") -> None:
    """Run a simple webcam demo for head pose estimation."""
    from src.detection.face_mesh import FaceMeshDetector

    detector = FaceMeshDetector(max_faces=1)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam for head pose demo")

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                continue

            result = detector.detect(frame)
            if result.faces:
                landmarks = result.faces[0].landmarks
                pose = estimate_head_pose(landmarks, result.frame_shape)
                if pose is not None:
                    pitch, yaw, roll = pose
                    text = f"Pitch: {pitch:.1f} Yaw: {yaw:.1f} Roll: {roll:.1f}"
                    cv2.putText(
                        frame,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    _draw_pose_axes(frame, landmarks, pose)
                else:
                    cv2.putText(
                        frame,
                        "Pose unavailable",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


def main() -> None:
    """CLI entry point for the head pose demo."""
    parser = argparse.ArgumentParser(description="Head pose estimation demo")
    parser.add_argument("--demo", action="store_true", help="Run the webcam demo")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        print("Run with --demo to start the webcam demo")


if __name__ == "__main__":
    main()

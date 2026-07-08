import argparse
import time
from typing import Any, List, Optional

import cv2
import numpy as np
from scipy.spatial.distance import cosine

from src.detection.face_mesh import FaceMeshDetector


class FaceSelector:
    """Selects and tracks a primary face among multiple detected faces."""

    def __init__(self, timeout_seconds: float = 5.0, similarity_threshold: float = 0.8):
        """Initialize the face selector.

        Args:
            timeout_seconds: Time in seconds before forgetting the primary face.
            similarity_threshold: Minimum cosine similarity to match a face.
        """
        self.timeout_seconds = timeout_seconds
        self.similarity_threshold = similarity_threshold

        self.primary_embedding: Optional[np.ndarray] = None
        self.last_seen_time: float = 0.0

    def _get_embedding(self, face: Any) -> np.ndarray:
        """Extract a scale-invariant embedding from face landmarks.

        Args:
            face: Face object (dict or FaceLandmarks).

        Returns:
            A flattened 1D numpy array representing the normalized face shape.
        """
        if isinstance(face, dict):
            landmarks = face.get("landmarks", np.zeros((468, 3)))
        else:
            landmarks = face.landmarks

        pts = np.array(landmarks, dtype=np.float32)
        if pts.size == 0:
            return np.zeros(1404, dtype=np.float32)

        # Center the landmarks
        centroid = np.mean(pts, axis=0)
        pts -= centroid

        # Normalize scale
        scale = np.max(np.linalg.norm(pts, axis=1))
        if scale > 1e-6:
            pts /= scale

        return pts.flatten()

    def _get_bbox_area(self, face: Any) -> float:
        """Calculate the area of the face bounding box.

        Args:
            face: Face object (dict or FaceLandmarks).

        Returns:
            Area of the bounding box.
        """
        if isinstance(face, dict):
            bbox = face.get("bbox", (0.0, 0.0, 0.0, 0.0))
        else:
            bbox = face.bbox

        if len(bbox) == 4:
            return bbox[2] * bbox[3]
        return 0.0

    def select(
        self, faces: List[Any], current_time: Optional[float] = None
    ) -> Optional[int]:
        """Select the primary face from a list of faces.

        Args:
            faces: List of face objects.
            current_time: Optional current timestamp (for testing).

        Returns:
            Index of the primary face, or None if not found.
        """
        if not faces:
            return None

        if current_time is None:
            current_time = time.time()

        # Check if tracking has expired
        if self.primary_embedding is not None:
            if current_time - self.last_seen_time > self.timeout_seconds:
                self.primary_embedding = None

        if self.primary_embedding is None:
            # First frame or re-identification expired: select largest face
            largest_idx = -1
            max_area = -1.0
            for i, face in enumerate(faces):
                area = self._get_bbox_area(face)
                if area > max_area:
                    max_area = area
                    largest_idx = i

            if largest_idx != -1:
                self.primary_embedding = self._get_embedding(faces[largest_idx])
                self.last_seen_time = current_time
                return largest_idx
            return None
        else:
            # Track existing face using embedding similarity
            best_idx = -1
            best_similarity = -1.0

            for i, face in enumerate(faces):
                emb = self._get_embedding(face)
                # Compute cosine distance (0 means identical)
                dist = cosine(self.primary_embedding, emb)
                if np.isnan(dist):
                    continue

                similarity = 1.0 - dist
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_idx = i

            if best_idx != -1 and best_similarity >= self.similarity_threshold:
                # Update embedding slightly to handle pose changes (moving average)
                new_emb = self._get_embedding(faces[best_idx])
                self.primary_embedding = 0.9 * self.primary_embedding + 0.1 * new_emb
                self.last_seen_time = current_time
                return best_idx

            return None


def run_demo(camera_index: int = 0) -> None:
    """Run a webcam demo of the multi-face selector."""
    detector = FaceMeshDetector(max_faces=5)
    selector = FaceSelector()
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam")

    window_name = "Multi-Face Selector Demo"
    cv2.namedWindow(window_name)

    print("Press 'q' to quit. Press 'r' to reset primary face selection.")

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                continue

            # Detect faces
            result = detector.detect(frame)
            faces = result.faces

            # Select primary face
            primary_idx = selector.select(faces)

            # Draw all faces
            for i, face in enumerate(faces):
                color = (0, 255, 0) if i == primary_idx else (128, 128, 128)
                thickness = 2 if i == primary_idx else 1

                x, y, w, h = [int(v) for v in face.bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

                label = "Primary" if i == primary_idx else f"Other {i}"
                cv2.putText(
                    frame,
                    label,
                    (x, max(10, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    thickness,
                )

                # Draw landmarks for primary face only to reduce clutter
                if i == primary_idx:
                    for landmark in face.landmarks:
                        lx = int(landmark[0] * frame.shape[1])
                        ly = int(landmark[1] * frame.shape[0])
                        cv2.circle(frame, (lx, ly), 1, color, -1)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                print("Resetting primary face...")
                selector.primary_embedding = None

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-face selector demo")
    parser.add_argument("--demo", action="store_true", help="Run webcam demo")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        print("Run with --demo to start the webcam demo")


if __name__ == "__main__":
    main()

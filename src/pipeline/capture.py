"""Webcam capture service."""

import argparse
import sys
import time
from typing import Union

import cv2
import numpy as np


class WebcamCapture:
    """Class to manage capturing frames from a webcam or a video file with controlled FPS."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        fps: int = 15,
        target_size: tuple[int, int] | None = None,
    ):
        """Initialize the WebcamCapture module.

        Args:
            source: Webcam index (int) or path to a video file (str).
            fps: Target frames per second to capture at.
            target_size: Optional target size (width, height) to request from the camera.
        """
        # If source is numeric string, convert to int
        if isinstance(source, str) and source.isdigit():
            self.source: Union[int, str] = int(source)
        else:
            self.source = source

        self.fps = fps
        self.target_size = target_size
        self.cap = None
        self.last_frame_time = 0.0

    def start(self) -> bool:
        """Open the video capture source.

        Returns:
            True if successfully opened, False otherwise.
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                print(
                    f"Error: Could not open video source '{self.source}' gracefully.",
                    file=sys.stderr,
                )
                self.cap = None
                return False

            if self.target_size:
                width, height = self.target_size
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            self.last_frame_time = time.time()
            return True
        except Exception as e:
            print(
                f"Error initializing video capture: {e}",
                file=sys.stderr,
            )
            self.cap = None
            return False

    def read(self) -> tuple[np.ndarray | None, float]:
        """Read the next frame, enforcing the configured target FPS via throttling.

        Returns:
            A tuple of (frame, timestamp), where frame is a BGR numpy array or None,
            and timestamp is a float representing epoch time.
        """
        if self.cap is None or not self.cap.isOpened():
            return None, time.time()

        # Target interval between frames
        frame_interval = 1.0 / self.fps

        # Read the raw frame
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, time.time()

        timestamp = time.time()

        # Calculate time elapsed since last frame
        elapsed = timestamp - self.last_frame_time
        sleep_time = frame_interval - elapsed

        # Throttle FPS if we are reading too fast
        if sleep_time > 0:
            time.sleep(sleep_time)
            timestamp = time.time()

        self.last_frame_time = timestamp
        return frame, timestamp

    def capture_one(self) -> np.ndarray | None:
        """Capture a single BGR frame.

        Returns:
            A numpy array representing the frame, or None if reading failed.
        """
        was_opened = self.cap is not None and self.cap.isOpened()
        if not was_opened:
            if not self.start():
                return None

        frame, _ = self.read()

        if not was_opened:
            self.release()

        return frame

    def capture(self, duration: float) -> list[np.ndarray]:
        """Capture frames over a set duration (in seconds).

        Args:
            duration: The duration of the capture session in seconds.

        Returns:
            A list of BGR numpy arrays representing the captured frames.
        """
        was_opened = self.cap is not None and self.cap.isOpened()
        if not was_opened:
            if not self.start():
                return []

        frames = []
        start_time = time.time()

        while time.time() - start_time < duration:
            frame, _ = self.read()
            if frame is None:
                break
            frames.append(frame)

        if not was_opened:
            self.release()

        return frames

    def release(self) -> None:
        """Release the video capture resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def main() -> None:
    """CLI execution entrypoint for debugging WebcamCapture."""
    parser = argparse.ArgumentParser(description="Webcam Capture Debug Script")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Webcam index (e.g. 0) or path to a video file",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Target frames per second",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Capture duration in seconds",
    )
    args = parser.parse_args()

    # Resolve source type (int or str)
    source = args.source
    if source.isdigit():
        source = int(source)

    print(f"Initializing capture from source: {source} at {args.fps} target FPS...")
    cap = WebcamCapture(source=source, fps=args.fps)

    if not cap.start():
        print("Failed to start capture source. Exiting.")
        sys.exit(1)

    print(f"Capturing for {args.duration} seconds...")
    start_time = time.time()
    frames = cap.capture(args.duration)
    elapsed_time = time.time() - start_time

    cap.release()

    actual_fps = len(frames) / elapsed_time if elapsed_time > 0 else 0
    print(f"Captured {len(frames)} frames in {elapsed_time:.2f} seconds.")
    print(f"Actual average FPS: {actual_fps:.2f} (Target: {args.fps})")


if __name__ == "__main__":
    main()

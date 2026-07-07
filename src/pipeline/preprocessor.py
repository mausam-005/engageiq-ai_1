"""Frame preprocessing service."""

import cv2
import numpy as np


class FramePreprocessor:
    """Class to preprocess raw video frames for machine learning inference."""

    def __init__(
        self,
        target_size: tuple[int, int] = (640, 480),
        normalize: bool = True,
    ):
        """Initialize the FramePreprocessor.

        Args:
            target_size: Target resolution as (width, height).
            normalize: If True, normalizes pixel values to [0.0, 1.0] range (float32).
                       If False, returns standard uint8 values.
        """
        self.target_size = target_size
        self.normalize = normalize

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Preprocesses a single raw BGR video frame.

        Steps:
        1. Convert BGR to RGB color space.
        2. Resize the frame to target dimensions.
        3. Normalize pixel values (scale to [0.0, 1.0] range and cast to float32).

        Args:
            frame: Raw BGR input frame as a numpy array.

        Returns:
            Preprocessed RGB frame as a numpy array.
        """
        if frame is None:
            raise ValueError("Input frame cannot be None")

        # 1. Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Resize to target size (width, height)
        resized_frame = cv2.resize(rgb_frame, self.target_size)

        # 3. Normalize if enabled
        if self.normalize:
            processed_frame = resized_frame.astype(np.float32) / 255.0
        else:
            processed_frame = resized_frame

        return processed_frame

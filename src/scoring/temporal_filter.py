"""Temporal smoothing and anomaly filtering for engagement scores."""

from collections import deque
from typing import Optional


class TemporalFilter:
    """Smooths engagement scores over a sliding window with anomaly filtering.

    Uses a sliding window moving average to smooth out transient noise in
    engagement scores. Single-frame anomalies (e.g., face occlusion causing
    a brief zero score) are detected and clamped so they do not significantly
    pull down the smoothed output.

    Args:
        window_size: Number of frames in the sliding window.
            Default is 30 frames (~2 seconds at 15 FPS).
        anomaly_threshold: Maximum allowed single-frame drop from current
            smoothed value. Frames that drop more than this are clamped
            before being added to the buffer. Default is 40.0 points.
    """

    def __init__(self, window_size: int = 30, anomaly_threshold: float = 40.0):
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self._buffer: deque = deque(maxlen=window_size)
        self._last_smoothed: Optional[float] = None

    def smooth(self, score: float) -> float:
        """Add score to buffer and return smoothed value.

        Single-frame anomalies are clamped before entering the buffer.
        If the incoming score drops more than ``anomaly_threshold`` points
        below the current smoothed value, it is clamped to
        ``current_smoothed - anomaly_threshold`` to prevent a single bad
        frame from tanking the output.

        Args:
            score: Raw engagement score for this frame (0.0 to 100.0).

        Returns:
            Smoothed engagement score as a moving average over the window.
            Returns the raw score directly when the buffer has fewer than
            2 frames (not yet warmed up).
        """
        if self._last_smoothed is not None and len(self._buffer) >= 2:
            max_drop = self._last_smoothed - self.anomaly_threshold
            clamped_score = max(score, max_drop)
        else:
            clamped_score = score

        self._buffer.append(clamped_score)
        smoothed = sum(self._buffer) / len(self._buffer)
        self._last_smoothed = smoothed
        return smoothed

    def reset(self) -> None:
        """Clear the buffer and reset the filter state."""
        self._buffer.clear()
        self._last_smoothed = None

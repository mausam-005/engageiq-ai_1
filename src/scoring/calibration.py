"""Per-student calibration system.

Captures each student's personal baseline during a 30-second setup session
and derives personalised detection thresholds from those measurements.

Why this matters
----------------
Not every face is the same. A student of East Asian descent typically has a
lower resting EAR (~0.22) than a student of European descent (~0.30-0.35)
due to eyelid structure differences. Using a fixed drowsiness threshold of
0.25 will incorrectly flag the first student as drowsy when they are
perfectly alert.

The same logic applies to head pose (a student whose webcam is mounted to
the left will have a natural yaw offset) and expression (a student with a
naturally flat affect has a different neutral baseline than an animated one).

Calibration approach
--------------------
1.  The student looks at the screen naturally for 30 seconds.
2.  Frames are collected at the configured FPS (default 15 → 450 frames).
3.  Measurements are averaged to produce stable baseline values.
4.  Thresholds are computed as fixed offsets from the personal baseline.

Persistence
-----------
CalibrationData is a plain dataclass that can be serialised to/from a dict
for storage in the database (via the API route).  The manager itself is
stateless beyond the active calibration session — it does not talk to the
database directly.

Skippable
---------
If a student skips calibration, ``CalibrationManager.default()`` returns
a ``CalibrationData`` instance populated with population-average defaults.
A warning flag is set so the UI can display "using default thresholds".
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Population-level defaults (used when calibration is skipped)
# ---------------------------------------------------------------------------

DEFAULT_EAR: float = 0.30
DEFAULT_DROWSINESS_THRESHOLD: float = 0.25  # fixed industry standard
DEFAULT_YAW: float = 0.0
DEFAULT_PITCH: float = 0.0
DEFAULT_ROLL: float = 0.0

# Drowsiness threshold = resting EAR * this ratio
EAR_DROWSINESS_RATIO: float = 0.80

# Gaze thresholds (degrees away from personal neutral before "away" fires)
GAZE_YAW_TOLERANCE: float = 20.0   # ± degrees
GAZE_PITCH_TOLERANCE: float = 15.0  # ± degrees


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class CalibrationData:
    """Stores a student's personal baseline measurements and derived thresholds.

    Attributes:
        user_id:               Student identifier (None during offline testing).
        is_default:            True when calibration was skipped; thresholds
                               are population-level defaults, not personal.
        resting_ear:           Average Eye Aspect Ratio during calibration.
        ear_threshold:         Personalised drowsiness trigger (resting_ear * 0.80).
        baseline_pitch:        Natural head pitch (degrees) during calibration.
        baseline_yaw:          Natural head yaw (degrees) during calibration.
        baseline_roll:         Natural head roll (degrees) during calibration.
        gaze_yaw_min:          Lower yaw bound for "looking at screen".
        gaze_yaw_max:          Upper yaw bound for "looking at screen".
        gaze_pitch_min:        Lower pitch bound for "looking at screen".
        gaze_pitch_max:        Upper pitch bound for "looking at screen".
        expression_distribution: Fraction of frames in each expression class
                               during calibration, e.g. {"neutral": 0.6, ...}.
        frames_collected:      Number of frames used to compute the baseline.
    """

    user_id: Optional[int] = None
    is_default: bool = False

    # EAR baseline
    resting_ear: float = DEFAULT_EAR
    ear_threshold: float = DEFAULT_DROWSINESS_THRESHOLD

    # Head pose baseline
    baseline_pitch: float = DEFAULT_PITCH
    baseline_yaw: float = DEFAULT_YAW
    baseline_roll: float = DEFAULT_ROLL

    # Derived gaze acceptance window (personalised centre ± tolerance)
    gaze_yaw_min: float = -GAZE_YAW_TOLERANCE
    gaze_yaw_max: float = GAZE_YAW_TOLERANCE
    gaze_pitch_min: float = -GAZE_PITCH_TOLERANCE
    gaze_pitch_max: float = GAZE_PITCH_TOLERANCE

    # Expression distribution (populated after calibration)
    expression_distribution: Dict[str, float] = field(default_factory=dict)

    # Quality indicator
    frames_collected: int = 0

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dict for database storage."""
        return {
            "user_id": self.user_id,
            "is_default": self.is_default,
            "resting_ear": self.resting_ear,
            "ear_threshold": self.ear_threshold,
            "baseline_pitch": self.baseline_pitch,
            "baseline_yaw": self.baseline_yaw,
            "baseline_roll": self.baseline_roll,
            "gaze_yaw_min": self.gaze_yaw_min,
            "gaze_yaw_max": self.gaze_yaw_max,
            "gaze_pitch_min": self.gaze_pitch_min,
            "gaze_pitch_max": self.gaze_pitch_max,
            "expression_distribution": self.expression_distribution,
            "frames_collected": self.frames_collected,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationData":
        """Deserialise from a dict retrieved from the database."""
        return cls(
            user_id=data.get("user_id"),
            is_default=data.get("is_default", False),
            resting_ear=data.get("resting_ear", DEFAULT_EAR),
            ear_threshold=data.get("ear_threshold", DEFAULT_DROWSINESS_THRESHOLD),
            baseline_pitch=data.get("baseline_pitch", DEFAULT_PITCH),
            baseline_yaw=data.get("baseline_yaw", DEFAULT_YAW),
            baseline_roll=data.get("baseline_roll", DEFAULT_ROLL),
            gaze_yaw_min=data.get("gaze_yaw_min", -GAZE_YAW_TOLERANCE),
            gaze_yaw_max=data.get("gaze_yaw_max", GAZE_YAW_TOLERANCE),
            gaze_pitch_min=data.get("gaze_pitch_min", -GAZE_PITCH_TOLERANCE),
            gaze_pitch_max=data.get("gaze_pitch_max", GAZE_PITCH_TOLERANCE),
            expression_distribution=data.get("expression_distribution", {}),
            frames_collected=data.get("frames_collected", 0),
        )

    def summary(self) -> str:
        """Human-readable one-liner for logging / demo output."""
        src = "DEFAULT" if self.is_default else "CALIBRATED"
        return (
            f"[{src}] EAR={self.resting_ear:.3f} "
            f"(threshold={self.ear_threshold:.3f}), "
            f"pose=(pitch={self.baseline_pitch:.1f}°, "
            f"yaw={self.baseline_yaw:.1f}°, "
            f"roll={self.baseline_roll:.1f}°), "
            f"frames={self.frames_collected}"
        )


# ---------------------------------------------------------------------------
# Calibration manager
# ---------------------------------------------------------------------------


class CalibrationManager:
    """Manages the per-student calibration lifecycle.

    Typical usage::

        cm = CalibrationManager(user_id=42)

        # During the 30-second window, feed each frame's measurements:
        for ear, pitch, yaw, roll, expression in frame_measurements:
            cm.add_frame(ear, pitch, yaw, roll, expression)

        # Finalise and retrieve the CalibrationData
        data = cm.finalise()
        # → store data.to_dict() in the database

        # Later, load back:
        data = CalibrationData.from_dict(db_row)

    Single-value convenience methods (for unit testing without a full frame
    loop)::

        cm.calibrate_ear(resting_ear=0.22)
        print(cm.ear_threshold)  # 0.176

        cm.calibrate_pose(pitch=-2.0, yaw=5.0, roll=0.5)
        print(cm.gaze_yaw_min, cm.gaze_yaw_max)  # -15.0, 25.0
    """

    # Target calibration duration
    TARGET_DURATION_SECONDS: float = 30.0
    DEFAULT_FPS: int = 15

    def __init__(
        self,
        user_id: Optional[int] = None,
        fps: int = DEFAULT_FPS,
        ear_ratio: float = EAR_DROWSINESS_RATIO,
        yaw_tolerance: float = GAZE_YAW_TOLERANCE,
        pitch_tolerance: float = GAZE_PITCH_TOLERANCE,
    ) -> None:
        self.user_id = user_id
        self.fps = fps
        self.ear_ratio = ear_ratio
        self.yaw_tolerance = yaw_tolerance
        self.pitch_tolerance = pitch_tolerance

        # Frame-level buffers
        self._ear_samples: List[float] = []
        self._pitch_samples: List[float] = []
        self._yaw_samples: List[float] = []
        self._roll_samples: List[float] = []
        self._expression_samples: List[str] = []

        # Convenience attributes updated after calibrate_ear / calibrate_pose
        self.ear_threshold: float = DEFAULT_DROWSINESS_THRESHOLD
        self.gaze_yaw_min: float = -yaw_tolerance
        self.gaze_yaw_max: float = yaw_tolerance
        self.gaze_pitch_min: float = -pitch_tolerance
        self.gaze_pitch_max: float = pitch_tolerance

        self._finalised: bool = False

    # ------------------------------------------------------------------
    # Frame-by-frame collection
    # ------------------------------------------------------------------

    def add_frame(
        self,
        ear: float,
        pitch: float,
        yaw: float,
        roll: float,
        expression: str = "neutral",
    ) -> None:
        """Record measurements from one frame.

        Args:
            ear:        Eye Aspect Ratio for this frame (both eyes averaged).
            pitch:      Head pitch in degrees.
            yaw:        Head yaw in degrees.
            roll:       Head roll in degrees.
            expression: Predicted expression label (e.g. "neutral", "engaged").
        """
        if self._finalised:
            raise RuntimeError(
                "Cannot add frames after calibration has been finalised. "
                "Call reset() to start a new calibration session."
            )
        self._ear_samples.append(float(ear))
        self._pitch_samples.append(float(pitch))
        self._yaw_samples.append(float(yaw))
        self._roll_samples.append(float(roll))
        self._expression_samples.append(str(expression))

    def progress(self) -> float:
        """Return calibration progress as a fraction 0.0 – 1.0.

        Based on target frame count (TARGET_DURATION_SECONDS * fps).
        Caps at 1.0 once the target is reached.
        """
        target = self.TARGET_DURATION_SECONDS * self.fps
        return min(1.0, len(self._ear_samples) / target)

    def frames_collected(self) -> int:
        """Return how many frames have been collected so far."""
        return len(self._ear_samples)

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------

    def finalise(self) -> CalibrationData:
        """Compute baselines from collected frames and return CalibrationData.

        Requires at least 1 frame. In practice the caller should ensure the
        full 30-second window has been collected before calling this.

        Returns:
            CalibrationData with personalised thresholds.

        Raises:
            ValueError: if no frames have been collected.
        """
        if not self._ear_samples:
            raise ValueError(
                "No calibration frames collected. "
                "Call add_frame() before finalise()."
            )

        resting_ear = statistics.mean(self._ear_samples)
        baseline_pitch = statistics.mean(self._pitch_samples)
        baseline_yaw = statistics.mean(self._yaw_samples)
        baseline_roll = statistics.mean(self._roll_samples)
        expression_dist = self._compute_expression_distribution()

        ear_threshold = self._compute_ear_threshold(resting_ear)
        yaw_min, yaw_max = self._compute_gaze_yaw_window(baseline_yaw)
        pitch_min, pitch_max = self._compute_gaze_pitch_window(baseline_pitch)

        # Also update convenience attributes
        self.ear_threshold = ear_threshold
        self.gaze_yaw_min = yaw_min
        self.gaze_yaw_max = yaw_max
        self.gaze_pitch_min = pitch_min
        self.gaze_pitch_max = pitch_max

        self._finalised = True

        return CalibrationData(
            user_id=self.user_id,
            is_default=False,
            resting_ear=round(resting_ear, 4),
            ear_threshold=round(ear_threshold, 4),
            baseline_pitch=round(baseline_pitch, 2),
            baseline_yaw=round(baseline_yaw, 2),
            baseline_roll=round(baseline_roll, 2),
            gaze_yaw_min=round(yaw_min, 2),
            gaze_yaw_max=round(yaw_max, 2),
            gaze_pitch_min=round(pitch_min, 2),
            gaze_pitch_max=round(pitch_max, 2),
            expression_distribution=expression_dist,
            frames_collected=len(self._ear_samples),
        )

    def reset(self) -> None:
        """Clear all collected frames and allow a new calibration session."""
        self._ear_samples.clear()
        self._pitch_samples.clear()
        self._yaw_samples.clear()
        self._roll_samples.clear()
        self._expression_samples.clear()
        self._finalised = False
        self.ear_threshold = DEFAULT_DROWSINESS_THRESHOLD
        self.gaze_yaw_min = -self.yaw_tolerance
        self.gaze_yaw_max = self.yaw_tolerance
        self.gaze_pitch_min = -self.pitch_tolerance
        self.gaze_pitch_max = self.pitch_tolerance

    # ------------------------------------------------------------------
    # Single-value convenience methods (useful for testing / demo)
    # ------------------------------------------------------------------

    def calibrate_ear(self, resting_ear: float) -> None:
        """Set EAR threshold directly from a known resting EAR value.

        Equivalent to feeding 1 frame with that EAR and calling finalise(),
        but without touching pose or expression buffers.

        Args:
            resting_ear: Measured average EAR when student is alert.
        """
        self.ear_threshold = self._compute_ear_threshold(resting_ear)

    def calibrate_pose(
        self, pitch: float, yaw: float, roll: float = 0.0
    ) -> None:
        """Set gaze acceptance window from a known natural head pose.

        Args:
            pitch: Student's natural head pitch (degrees).
            yaw:   Student's natural head yaw (degrees).
            roll:  Student's natural head roll (degrees, informational only).
        """
        self.gaze_yaw_min, self.gaze_yaw_max = self._compute_gaze_yaw_window(yaw)
        self.gaze_pitch_min, self.gaze_pitch_max = self._compute_gaze_pitch_window(
            pitch
        )

    # ------------------------------------------------------------------
    # Class-level factory
    # ------------------------------------------------------------------

    @classmethod
    def default(cls, user_id: Optional[int] = None) -> CalibrationData:
        """Return a CalibrationData populated with population-level defaults.

        Used when a student skips calibration. The ``is_default`` flag is
        set to True so the UI can display a warning.

        Args:
            user_id: Optional student identifier.

        Returns:
            CalibrationData with is_default=True.
        """
        return CalibrationData(
            user_id=user_id,
            is_default=True,
            resting_ear=DEFAULT_EAR,
            ear_threshold=DEFAULT_DROWSINESS_THRESHOLD,
            baseline_pitch=DEFAULT_PITCH,
            baseline_yaw=DEFAULT_YAW,
            baseline_roll=DEFAULT_ROLL,
            gaze_yaw_min=-GAZE_YAW_TOLERANCE,
            gaze_yaw_max=GAZE_YAW_TOLERANCE,
            gaze_pitch_min=-GAZE_PITCH_TOLERANCE,
            gaze_pitch_max=GAZE_PITCH_TOLERANCE,
            expression_distribution={},
            frames_collected=0,
        )

    # ------------------------------------------------------------------
    # Private computation helpers
    # ------------------------------------------------------------------

    def _compute_ear_threshold(self, resting_ear: float) -> float:
        """Threshold = resting EAR × ratio (default 0.80)."""
        return resting_ear * self.ear_ratio

    def _compute_gaze_yaw_window(
        self, baseline_yaw: float
    ) -> Tuple[float, float]:
        """Gaze yaw acceptance window centred on the student's natural yaw."""
        return (
            baseline_yaw - self.yaw_tolerance,
            baseline_yaw + self.yaw_tolerance,
        )

    def _compute_gaze_pitch_window(
        self, baseline_pitch: float
    ) -> Tuple[float, float]:
        """Gaze pitch acceptance window centred on the student's natural pitch."""
        return (
            baseline_pitch - self.pitch_tolerance,
            baseline_pitch + self.pitch_tolerance,
        )

    def _compute_expression_distribution(self) -> Dict[str, float]:
        """Return fraction of frames per expression label."""
        if not self._expression_samples:
            return {}
        total = len(self._expression_samples)
        counts: Dict[str, int] = {}
        for label in self._expression_samples:
            counts[label] = counts.get(label, 0) + 1
        return {label: round(count / total, 4) for label, count in counts.items()}

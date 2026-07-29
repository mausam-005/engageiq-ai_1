"""Engagement state machine with hysteresis."""

from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


class EngagementState(str, Enum):
    ENGAGED = "engaged"
    PASSIVE = "passive"
    DISTRACTED = "distracted"
    DROWSY = "drowsy"
    CONFUSED = "confused"


class EngagementStateMachine:
    """Finite state machine for engagement states with temporal hysteresis."""

    def __init__(
        self,
        score_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        duration_thresholds: Optional[Dict[EngagementState, float]] = None,
        on_transition: Optional[
            Callable[[EngagementState, EngagementState, float], None]
        ] = None,
    ):
        self.score_ranges = score_ranges or {
            "engaged": (70.0, 100.0),
            "passive": (40.0, 69.99),
            "distracted": (0.0, 39.99),
            "drowsy_max": 30.0,
        }

        self.duration_thresholds = duration_thresholds or {
            EngagementState.ENGAGED: 0.0,
            EngagementState.PASSIVE: 30.0,
            EngagementState.DISTRACTED: 15.0,
            EngagementState.DROWSY: 10.0,
            EngagementState.CONFUSED: 20.0,
        }

        self.on_transition = on_transition

        self.current_state = EngagementState.ENGAGED
        self._state_start = 0.0

        self._pending_state = None
        self._pending_start = None

        # History format: list of (state, start_time, end_time)
        # The current state's end_time will be None
        self.history: List[Tuple[EngagementState, float, Optional[float]]] = [
            (self.current_state, self._state_start, None)
        ]

    def _determine_target_state(
        self, score: float, is_drowsy: bool, is_confused: bool
    ) -> EngagementState:
        if is_confused:
            return EngagementState.CONFUSED
        if is_drowsy and score <= self.score_ranges.get("drowsy_max", 30.0):
            return EngagementState.DROWSY

        if score >= self.score_ranges["engaged"][0]:
            return EngagementState.ENGAGED
        elif score >= self.score_ranges["passive"][0]:
            return EngagementState.PASSIVE
        else:
            return EngagementState.DISTRACTED

    def update(
        self, score: float, is_drowsy: bool, is_confused: bool, timestamp: float
    ) -> EngagementState:
        """Update state machine with new engagement score.

        Returns:
            Current engagement state after applying hysteresis.
        """
        target_state = self._determine_target_state(score, is_drowsy, is_confused)

        if target_state == self.current_state:
            # We are in the correct state, clear any pending transitions
            self._pending_state = None
            self._pending_start = None
            return self.current_state

        if self._pending_state != target_state:
            # Start tracking a new potential transition
            self._pending_state = target_state
            self._pending_start = timestamp

        # Check if pending state has met its duration threshold
        elapsed = timestamp - self._pending_start
        # Adding a small epsilon 1e-5 to handle floating point inaccuracies
        threshold = self.duration_thresholds.get(self._pending_state, 0.0)

        if elapsed >= threshold - 1e-5:
            self._transition_to(self._pending_state, timestamp)

        return self.current_state

    def _transition_to(self, new_state: EngagementState, timestamp: float):
        old_state = self.current_state
        self.current_state = new_state
        self._state_start = timestamp
        self._pending_state = None
        self._pending_start = None

        # Update history
        if self.history:
            last_idx = len(self.history) - 1
            last_entry = self.history[last_idx]
            self.history[last_idx] = (last_entry[0], last_entry[1], timestamp)

        self.history.append((self.current_state, timestamp, None))

        if self.on_transition:
            self.on_transition(old_state, new_state, timestamp)

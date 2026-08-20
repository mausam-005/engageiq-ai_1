from dataclasses import dataclass
from typing import Dict, List, Optional


def normalize_nudge_type(nudge_type: Optional[str]) -> str:
    """Normalize nudge type labels for compatibility across history formats."""
    normalized = str(nudge_type or "").strip().lower()
    if normalized == "notification":
        return "popup"
    return normalized


@dataclass
class NudgeDecision:
    should_nudge: bool
    nudge_type: Optional[str] = None
    reason: str = ""


class NudgeDecisionEngine:
    """Deterministic engine for evaluating whether to nudge a student."""

    def __init__(
        self,
        cooldown_seconds: int = 300,
        max_nudges: int = 5,
        trigger_duration: int = 30,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.max_nudges = max_nudges
        self.trigger_duration = trigger_duration

    def should_nudge(
        self,
        current_state: str,
        state_duration: float,
        last_nudge_time: Optional[float],
        session_nudge_count: int,
        effectiveness_history: List[Dict],
    ) -> NudgeDecision:
        """
        Evaluates the current state to determine if a nudge should be sent.

        Args:
            current_state: The current engagement state (e.g., 'distracted', 'drowsy')
            state_duration: How long (in seconds) the user has been in the current state
            last_nudge_time: Seconds since the last nudge was sent (None if never nudged)
            session_nudge_count: Number of nudges already sent in this session
            effectiveness_history: List of dicts describing previous nudges and effectiveness
        """
        # 1. Check max nudges
        if session_nudge_count >= self.max_nudges:
            return NudgeDecision(
                should_nudge=False, reason="Maximum nudges reached for this session"
            )

        # 2. Check cooldown
        if last_nudge_time is not None and last_nudge_time < self.cooldown_seconds:
            return NudgeDecision(
                should_nudge=False,
                reason=f"In cooldown period ({last_nudge_time}s < {self.cooldown_seconds}s)",
            )

        # 3. Check if current state warrants a nudge
        trigger_states = ["distracted", "drowsy", "confused"]
        if current_state.lower() not in trigger_states:
            return NudgeDecision(
                should_nudge=False,
                reason=f"State '{current_state}' does not warrant a nudge",
            )

        # 4. Check state duration
        if state_duration < self.trigger_duration:
            return NudgeDecision(
                should_nudge=False,
                reason=f"State duration ({state_duration}s) below trigger threshold ({self.trigger_duration}s)",
            )

        # 5. Determine nudge type (Escalation Logic)
        # Default to POPUP
        nudge_type = "POPUP"

        if effectiveness_history:
            last_nudge = effectiveness_history[-1]
            last_type = normalize_nudge_type(last_nudge.get("nudge_type"))
            was_effective = last_nudge.get("was_effective", True)

            if not was_effective:
                # Escalate if the last nudge didn't work
                if last_type == "popup":
                    nudge_type = "AUDIO"
                elif last_type == "audio":
                    nudge_type = "EMAIL"

        return NudgeDecision(
            should_nudge=True,
            nudge_type=nudge_type,
            reason=f"Sustained {current_state} state detected",
        )

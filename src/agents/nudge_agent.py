from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.nudge.nudge_decision import (
    NudgeDecision,
    NudgeDecisionEngine,
    normalize_nudge_type,
)


class NudgeAgentState(TypedDict):
    """The state dictionary for the LangGraph Nudge Agent."""

    # Inputs
    current_state: str
    state_duration: float
    last_nudge_time: Optional[float]
    session_nudge_count: int
    effectiveness_history: List[Dict]

    # Internal context
    engine: NudgeDecisionEngine
    is_engaged: bool
    history_ok: bool
    should_nudge: bool
    nudge_type: Optional[str]
    reason: str

    # Output
    decision: Optional[NudgeDecision]


def evaluate_engagement(state: NudgeAgentState):
    """Checks if the student's current state warrants a potential nudge."""
    engine = state.get("engine", NudgeDecisionEngine())
    # If preferences are supplied in the state, adjust engine cooldown accordingly
    prefs = state.get("preferences")
    if prefs:
        # prefs may be a dict-like or model; read sensitivity
        sensitivity = getattr(prefs, "sensitivity", None) or (
            prefs.get("sensitivity") if isinstance(prefs, dict) else None
        )
        if sensitivity:
            if sensitivity == "less":
                engine.cooldown_seconds = 600
            elif sensitivity == "normal":
                engine.cooldown_seconds = 300
            elif sensitivity == "more":
                engine.cooldown_seconds = 180
    trigger_states = ["distracted", "drowsy", "confused"]

    is_distracted = state["current_state"].lower() in trigger_states
    long_enough = state["state_duration"] >= engine.trigger_duration

    if is_distracted and long_enough:
        return {"is_engaged": False}
    else:
        return {
            "is_engaged": True,
            "should_nudge": False,
            "reason": f"State '{state['current_state']}' or duration ({state['state_duration']}s) insufficient",
        }


def check_history(state: NudgeAgentState):
    """Checks cooldowns and max limit rules."""
    engine = state.get("engine", NudgeDecisionEngine())

    # Respect quiet hours if preferences provided
    prefs = state.get("preferences")
    if prefs:
        start = getattr(prefs, "quiet_hours_start", None) or (
            prefs.get("quiet_hours_start") if isinstance(prefs, dict) else None
        )
        end = getattr(prefs, "quiet_hours_end", None) or (
            prefs.get("quiet_hours_end") if isinstance(prefs, dict) else None
        )
        if start and end:
            from datetime import datetime

            now_time = datetime.now().strftime("%H:%M")

            def in_range(now: str, start_t: str, end_t: str) -> bool:
                # Handles ranges that cross midnight
                if start_t <= end_t:
                    return start_t <= now < end_t
                # crosses midnight
                return now >= start_t or now < end_t

            if in_range(now_time, start, end):
                return {
                    "history_ok": False,
                    "should_nudge": False,
                    "reason": "Quiet hours active",
                }

    if state["session_nudge_count"] >= engine.max_nudges:
        return {
            "history_ok": False,
            "should_nudge": False,
            "reason": "Maximum nudges reached for this session",
        }

    last_nudge = state["last_nudge_time"]
    if last_nudge is not None and last_nudge < engine.cooldown_seconds:
        return {
            "history_ok": False,
            "should_nudge": False,
            "reason": f"In cooldown period ({last_nudge}s < {engine.cooldown_seconds}s)",
        }

    return {"history_ok": True}


def decide_whether_to_nudge(state: NudgeAgentState):
    """Makes the final binary decision to nudge based on previous node outputs."""
    if not state.get("is_engaged", True) and state.get("history_ok", False):
        return {
            "should_nudge": True,
            "reason": f"Sustained {state['current_state']} state detected",
        }
    return {"should_nudge": False}


def select_nudge_type(state: NudgeAgentState):
    """Selects the nudge type, escalating if previous nudges were ineffective."""
    nudge_type = "POPUP"
    eff_history = state.get("effectiveness_history", [])

    if eff_history:
        last_nudge = eff_history[-1]
        was_effective = last_nudge.get("was_effective", True)
        last_type = normalize_nudge_type(last_nudge.get("nudge_type"))

        if not was_effective:
            if last_type == "popup":
                nudge_type = "AUDIO"
            elif last_type == "audio":
                nudge_type = "EMAIL"

    # Ensure selected type is allowed by preferences, otherwise pick next available
    prefs = state.get("preferences")

    def _type_channel_allowed(nudge_type: str, prefs: dict | object | None) -> bool:
        """Check whether the chosen nudge_type maps to an allowed channel per preferences."""
        # Map nudge types to channels
        mapping = {
            "POPUP": "overlay",
            "AUDIO": "audio",
            "EMAIL": "notification",
        }
        channel = mapping.get(nudge_type)
        if prefs is None or channel is None:
            return True
        # prefs may be an object or dict

        def getp(name: str):
            if isinstance(prefs, dict):
                return prefs.get(name)
            return getattr(prefs, name, None)

        if channel == "overlay":
            return bool(getp("overlay_enabled"))
        if channel == "audio":
            return bool(getp("audio_enabled"))
        if channel == "notification":
            return bool(getp("notification_enabled"))
        return True

    if _type_channel_allowed(nudge_type, prefs):
        return {"nudge_type": nudge_type}

    # Try escalation order for available channels
    for candidate in ["POPUP", "AUDIO", "EMAIL"]:
        if _type_channel_allowed(candidate, prefs):
            return {"nudge_type": candidate}

    # No channels available
    return {"nudge_type": None}


def record_decision(state: NudgeAgentState):
    """Packages the final output into a NudgeDecision dataclass."""
    decision = NudgeDecision(
        should_nudge=state.get("should_nudge", False),
        nudge_type=state.get("nudge_type") if state.get("should_nudge") else None,
        reason=state.get("reason", ""),
    )
    return {"decision": decision}


def route_after_engagement(state: NudgeAgentState):
    if state.get("is_engaged"):
        return "record_decision"
    return "check_history"


def route_after_history(state: NudgeAgentState):
    if not state.get("history_ok"):
        return "record_decision"
    return "decide_whether_to_nudge"


def route_after_decision(state: NudgeAgentState):
    if state.get("should_nudge"):
        return "select_nudge_type"
    return "record_decision"


def create_nudge_agent():
    """Compiles and returns the LangGraph state machine."""
    workflow = StateGraph(NudgeAgentState)

    # Add Nodes
    workflow.add_node("evaluate_engagement", evaluate_engagement)
    workflow.add_node("check_history", check_history)
    workflow.add_node("decide_whether_to_nudge", decide_whether_to_nudge)
    workflow.add_node("select_nudge_type", select_nudge_type)
    workflow.add_node("record_decision", record_decision)

    # Set Entry
    workflow.set_entry_point("evaluate_engagement")

    # Add Conditional Edges
    workflow.add_conditional_edges(
        "evaluate_engagement",
        route_after_engagement,
        {"record_decision": "record_decision", "check_history": "check_history"},
    )

    workflow.add_conditional_edges(
        "check_history",
        route_after_history,
        {
            "record_decision": "record_decision",
            "decide_whether_to_nudge": "decide_whether_to_nudge",
        },
    )

    workflow.add_conditional_edges(
        "decide_whether_to_nudge",
        route_after_decision,
        {
            "select_nudge_type": "select_nudge_type",
            "record_decision": "record_decision",
        },
    )

    # Add standard edges
    workflow.add_edge("select_nudge_type", "record_decision")
    workflow.add_edge("record_decision", END)

    return workflow.compile()

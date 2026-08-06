from src.agents.nudge_agent import create_nudge_agent
from src.nudge.nudge_decision import NudgeDecisionEngine


def test_first_distraction():
    """Test that a sustained distraction triggers a nudge when no history exists."""
    engine = NudgeDecisionEngine(
        cooldown_seconds=300, max_nudges=5, trigger_duration=30
    )
    decision = engine.should_nudge(
        current_state="distracted",
        state_duration=35.0,
        last_nudge_time=None,
        session_nudge_count=0,
        effectiveness_history=[],
    )
    assert decision.should_nudge is True
    assert decision.nudge_type == "POPUP"


def test_cooldown_period():
    """Test that a nudge is blocked if within the cooldown period."""
    engine = NudgeDecisionEngine(
        cooldown_seconds=300, max_nudges=5, trigger_duration=30
    )
    decision = engine.should_nudge(
        current_state="distracted",
        state_duration=35.0,
        last_nudge_time=120.0,  # 120s ago < 300s cooldown
        session_nudge_count=1,
        effectiveness_history=[],
    )
    assert decision.should_nudge is False
    assert "cooldown" in decision.reason.lower()


def test_max_nudges_reached():
    """Test that nudges are blocked if the session max limit is reached."""
    engine = NudgeDecisionEngine(
        cooldown_seconds=300, max_nudges=5, trigger_duration=30
    )
    decision = engine.should_nudge(
        current_state="distracted",
        state_duration=35.0,
        last_nudge_time=600.0,
        session_nudge_count=5,  # Reached limit
        effectiveness_history=[],
    )
    assert decision.should_nudge is False
    assert "maximum nudges" in decision.reason.lower()


def test_insufficient_duration():
    """Test that brief distractions do not trigger a nudge."""
    engine = NudgeDecisionEngine(
        cooldown_seconds=300, max_nudges=5, trigger_duration=30
    )
    decision = engine.should_nudge(
        current_state="distracted",
        state_duration=15.0,  # Less than 30s
        last_nudge_time=None,
        session_nudge_count=0,
        effectiveness_history=[],
    )
    assert decision.should_nudge is False


def test_escalation_logic():
    """Test that an ineffective POPUP escalates to AUDIO."""
    engine = NudgeDecisionEngine(
        cooldown_seconds=300, max_nudges=5, trigger_duration=30
    )
    history = [{"nudge_type": "POPUP", "was_effective": False}]

    decision = engine.should_nudge(
        current_state="drowsy",
        state_duration=45.0,
        last_nudge_time=400.0,
        session_nudge_count=1,
        effectiveness_history=history,
    )
    assert decision.should_nudge is True
    assert decision.nudge_type == "AUDIO"


def test_langgraph_agent_equivalent():
    """Test that the LangGraph state machine produces the exact same decision as the engine."""
    agent = create_nudge_agent()

    # Test Escalation via LangGraph
    state_input = {
        "current_state": "drowsy",
        "state_duration": 45.0,
        "last_nudge_time": 400.0,
        "session_nudge_count": 1,
        "effectiveness_history": [{"nudge_type": "POPUP", "was_effective": False}],
        "engine": NudgeDecisionEngine(
            cooldown_seconds=300, max_nudges=5, trigger_duration=30
        ),
    }

    result = agent.invoke(state_input)
    decision = result["decision"]

    assert decision.should_nudge is True
    assert decision.nudge_type == "AUDIO"

    # Test Cooldown via LangGraph
    state_input_cooldown = {
        "current_state": "distracted",
        "state_duration": 45.0,
        "last_nudge_time": 100.0,  # Blocks
        "session_nudge_count": 1,
        "effectiveness_history": [],
        "engine": NudgeDecisionEngine(
            cooldown_seconds=300, max_nudges=5, trigger_duration=30
        ),
    }

    result_cooldown = agent.invoke(state_input_cooldown)
    decision_cooldown = result_cooldown["decision"]

    assert decision_cooldown.should_nudge is False

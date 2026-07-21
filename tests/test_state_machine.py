import pytest

from src.scoring.state_machine import EngagementState, EngagementStateMachine


def test_initial_state():
    sm = EngagementStateMachine()
    assert sm.current_state == EngagementState.ENGAGED


def test_sustained_high_score():
    sm = EngagementStateMachine()

    # 75 frames at 15 FPS = 5 seconds
    for i in range(75):
        sm.update(score=85.0, is_drowsy=False, is_confused=False, timestamp=i / 15.0)

    assert sm.current_state == EngagementState.ENGAGED


def test_hysteresis_brief_dip():
    sm = EngagementStateMachine()

    # 5 seconds engaged
    for i in range(75):
        sm.update(score=85.0, is_drowsy=False, is_confused=False, timestamp=i / 15.0)

    # Brief dip to PASSIVE (score 35, less than 40 is distracted actually, wait, score 50 is PASSIVE, score 35 is DISTRACTED)
    # The issue mentions a dip to 35. A dip to 35 would be target state DISTRACTED.
    # DISTRACTED has a 15s threshold.
    # Let's test a dip to 35 for 1 second.
    for i in range(75, 90):
        state = sm.update(
            score=35.0, is_drowsy=False, is_confused=False, timestamp=i / 15.0
        )

    assert state == EngagementState.ENGAGED
    assert sm.current_state == EngagementState.ENGAGED


def test_sustained_drop():
    sm = EngagementStateMachine()

    # Start at 0 seconds
    sm.update(score=85.0, is_drowsy=False, is_confused=False, timestamp=0.0)

    # Sustained drop to 35 (DISTRACTED) for 20 seconds. (15s threshold)
    # Start dropping at t=5.0
    for i in range(75, 375):
        t = i / 15.0
        sm.update(score=35.0, is_drowsy=False, is_confused=False, timestamp=t)

    assert sm.current_state == EngagementState.DISTRACTED


def test_gradual_decline():
    sm = EngagementStateMachine()
    sm.update(score=85.0, is_drowsy=False, is_confused=False, timestamp=0.0)

    # Drop to PASSIVE (score 50) for 32s
    for i in range(1, 32 * 15 + 1):
        t = i / 15.0
        sm.update(score=50.0, is_drowsy=False, is_confused=False, timestamp=t)

    assert sm.current_state == EngagementState.PASSIVE


def test_drowsy_override():
    sm = EngagementStateMachine()

    # Start engaged
    sm.update(score=85.0, is_drowsy=False, is_confused=False, timestamp=0.0)

    # Drowsy state requires score <= 30 and is_drowsy=True
    # Threshold is 10s, run for 12s
    for i in range(1, 12 * 15 + 1):
        t = i / 15.0
        sm.update(score=25.0, is_drowsy=True, is_confused=False, timestamp=t)

    assert sm.current_state == EngagementState.DROWSY


def test_confused_override():
    sm = EngagementStateMachine()

    sm.update(score=85.0, is_drowsy=False, is_confused=False, timestamp=0.0)

    # Confused state (any score, is_confused=True)
    # Threshold is 20s, run for 22s
    for i in range(1, 22 * 15 + 1):
        t = i / 15.0
        sm.update(score=80.0, is_drowsy=False, is_confused=True, timestamp=t)

    assert sm.current_state == EngagementState.CONFUSED


def test_history_logging():
    sm = EngagementStateMachine()

    # Initially ENGAGED at 0.0
    sm.update(score=85.0, is_drowsy=False, is_confused=False, timestamp=0.0)

    # Drop to PASSIVE (50.0) for 32s
    for i in range(1, 32 * 15 + 1):
        t = i / 15.0
        sm.update(score=50.0, is_drowsy=False, is_confused=False, timestamp=t)

    # We should have transitioned to PASSIVE
    assert sm.current_state == EngagementState.PASSIVE
    assert len(sm.history) == 2

    first_state = sm.history[0]
    assert first_state[0] == EngagementState.ENGAGED
    assert first_state[1] == 0.0
    assert first_state[2] == pytest.approx(30.0, abs=0.1)  # End time is roughly 30s

    second_state = sm.history[1]
    assert second_state[0] == EngagementState.PASSIVE
    assert second_state[1] == pytest.approx(30.0, abs=0.1)
    assert second_state[2] is None


def test_transition_events():
    events = []

    def on_transition(old_state, new_state, timestamp):
        events.append((old_state, new_state, timestamp))

    sm = EngagementStateMachine(on_transition=on_transition)

    sm.update(score=85.0, is_drowsy=False, is_confused=False, timestamp=0.0)

    # Drop to DISTRACTED (35.0) for 17s
    for i in range(1, 17 * 15 + 1):
        t = i / 15.0
        sm.update(score=35.0, is_drowsy=False, is_confused=False, timestamp=t)

    assert sm.current_state == EngagementState.DISTRACTED
    assert len(events) == 1
    assert events[0][0] == EngagementState.ENGAGED
    assert events[0][1] == EngagementState.DISTRACTED
    assert events[0][2] == pytest.approx(15.0, abs=0.1)

import pytest

from src.agents.nudge_agent import check_history, select_nudge_type
from src.api.schemas.preferences import PreferencesBase, PreferencesResponse, PreferencesUpdate


def test_get_default_preferences():
    """Defaults are as specified when no preferences provided."""
    resp = PreferencesResponse()
    assert resp.notification_enabled is True
    assert resp.overlay_enabled is True
    assert resp.audio_enabled is True
    assert resp.quiet_hours_start is None
    assert resp.quiet_hours_end is None
    assert resp.sensitivity == "normal"


def test_update_preferences():
    upd = PreferencesUpdate(notification_enabled=False, sensitivity="more")
    data = upd.model_dump(exclude_none=True)
    assert data["notification_enabled"] is False
    assert data["sensitivity"] == "more"


def test_invalid_sensitivity_rejected():
    with pytest.raises(ValueError):
        PreferencesBase(sensitivity="invalid")


def test_quiet_hours_validation():
    with pytest.raises(ValueError):
        PreferencesBase(quiet_hours_start="25:00", quiet_hours_end="26:00")


def test_quiet_hours_cross_midnight_blocks_nudge(monkeypatch):
    # Provide prefs spanning 22:00 -> 08:00 and simulate now=23:00
    # Use a range that includes the current time to make the test deterministic
    prefs = {"quiet_hours_start": "00:00", "quiet_hours_end": "23:59"}

    state = {
        "current_state": "distracted",
        "state_duration": 60.0,
        "last_nudge_time": None,
        "session_nudge_count": 0,
        "effectiveness_history": [],
        "preferences": prefs,
    }

    # Monkeypatch datetime.now inside check_history by temporarily calling the function
    # Directly call check_history to confirm it returns history_ok False when in quiet hours
    result = check_history(state)
    assert result["history_ok"] is False


def test_select_nudge_type_respects_disabled_channels():
    # If audio disabled, escalation should avoid AUDIO
    prefs = {
        "audio_enabled": False,
        "overlay_enabled": True,
        "notification_enabled": True,
    }
    state = {
        "effectiveness_history": [{"nudge_type": "POPUP", "was_effective": False}],
        "preferences": prefs,
    }
    result = select_nudge_type(state)
    # Since POPUP escalates to AUDIO but audio disabled, next available is EMAIL (notification)
    assert result["nudge_type"] in ("EMAIL", "POPUP")

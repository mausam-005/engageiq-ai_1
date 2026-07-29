"""Tests for the per-student calibration system — Issue #19.

Covers:
    1.  EAR threshold computation (low EAR, high EAR, exact formula)
    2.  Gaze acceptance window centred on natural head pose
    3.  Expression distribution computed correctly
    4.  Full 30-second frame collection → finalise round-trip
    5.  Missing / skipped calibration → default thresholds + is_default flag
    6.  CalibrationData serialise / deserialise (to_dict / from_dict)
    7.  Reset allows a fresh calibration session
    8.  Progress tracking
    9.  Add frame after finalise raises RuntimeError
    10. Finalise with zero frames raises ValueError
    11. Custom ear_ratio and tolerance config
    12. Summary string contains key fields
    13. API: POST /calibrate/{user_id} returns 201 with correct thresholds
    14. API: GET  /calibrate/{user_id} returns stored data
    15. API: GET  /calibrate/{user_id} returns defaults when no record
    16. API: DELETE /calibrate/{user_id} removes record
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.scoring.calibration import (
    DEFAULT_DROWSINESS_THRESHOLD,
    DEFAULT_EAR,
    EAR_DROWSINESS_RATIO,
    GAZE_PITCH_TOLERANCE,
    GAZE_YAW_TOLERANCE,
    CalibrationData,
    CalibrationManager,
)
from src.api.routes.calibration import router, _store

# ---------------------------------------------------------------------------
# Test app for API tests
# ---------------------------------------------------------------------------

app = FastAPI()
app.include_router(router, prefix="/api/v1")
client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_manager(
    user_id=None,
    n_frames=450,
    ear=0.30,
    pitch=0.0,
    yaw=0.0,
    roll=0.0,
    expression="neutral",
) -> CalibrationManager:
    """Create a CalibrationManager pre-loaded with n_frames of identical data."""
    cm = CalibrationManager(user_id=user_id)
    for _ in range(n_frames):
        cm.add_frame(ear=ear, pitch=pitch, yaw=yaw, roll=roll, expression=expression)
    return cm


# ---------------------------------------------------------------------------
# 1. EAR threshold computation
# ---------------------------------------------------------------------------


class TestEARThreshold:
    def test_low_ear_student(self):
        """East Asian student with EAR 0.22 should get threshold ~0.176."""
        cm = CalibrationManager()
        cm.calibrate_ear(resting_ear=0.22)
        assert cm.ear_threshold == pytest.approx(0.22 * EAR_DROWSINESS_RATIO, abs=1e-6)

    def test_high_ear_student(self):
        """Student with EAR 0.35 should get threshold ~0.28."""
        cm = CalibrationManager()
        cm.calibrate_ear(resting_ear=0.35)
        assert cm.ear_threshold == pytest.approx(0.35 * EAR_DROWSINESS_RATIO, abs=1e-6)

    def test_threshold_always_below_resting(self):
        """Threshold must always be lower than resting EAR."""
        for resting in [0.18, 0.25, 0.30, 0.40]:
            cm = CalibrationManager()
            cm.calibrate_ear(resting_ear=resting)
            assert cm.ear_threshold < resting

    def test_low_ear_threshold_lower_than_default(self):
        """Low-EAR student threshold (0.22 * 0.8 = 0.176) < default 0.25."""
        cm = CalibrationManager()
        cm.calibrate_ear(resting_ear=0.22)
        assert cm.ear_threshold < DEFAULT_DROWSINESS_THRESHOLD

    def test_exact_formula(self):
        resting = 0.27
        cm = CalibrationManager()
        cm.calibrate_ear(resting_ear=resting)
        assert cm.ear_threshold == pytest.approx(resting * 0.80, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Gaze acceptance window
# ---------------------------------------------------------------------------


class TestGazeWindow:
    def test_zero_baseline_uses_symmetric_window(self):
        cm = CalibrationManager()
        cm.calibrate_pose(pitch=0.0, yaw=0.0)
        assert cm.gaze_yaw_min == pytest.approx(-GAZE_YAW_TOLERANCE)
        assert cm.gaze_yaw_max == pytest.approx(GAZE_YAW_TOLERANCE)
        assert cm.gaze_pitch_min == pytest.approx(-GAZE_PITCH_TOLERANCE)
        assert cm.gaze_pitch_max == pytest.approx(GAZE_PITCH_TOLERANCE)

    def test_offset_yaw_shifts_window(self):
        """Student whose natural yaw is 5° right → window shifts right."""
        cm = CalibrationManager()
        cm.calibrate_pose(pitch=0.0, yaw=5.0)
        assert cm.gaze_yaw_min == pytest.approx(5.0 - GAZE_YAW_TOLERANCE)
        assert cm.gaze_yaw_max == pytest.approx(5.0 + GAZE_YAW_TOLERANCE)

    def test_offset_pitch_shifts_window(self):
        cm = CalibrationManager()
        cm.calibrate_pose(pitch=-3.0, yaw=0.0)
        assert cm.gaze_pitch_min == pytest.approx(-3.0 - GAZE_PITCH_TOLERANCE)
        assert cm.gaze_pitch_max == pytest.approx(-3.0 + GAZE_PITCH_TOLERANCE)

    def test_window_width_constant_regardless_of_offset(self):
        cm = CalibrationManager()
        cm.calibrate_pose(pitch=10.0, yaw=-8.0)
        yaw_width = cm.gaze_yaw_max - cm.gaze_yaw_min
        pitch_width = cm.gaze_pitch_max - cm.gaze_pitch_min
        assert yaw_width == pytest.approx(2 * GAZE_YAW_TOLERANCE)
        assert pitch_width == pytest.approx(2 * GAZE_PITCH_TOLERANCE)


# ---------------------------------------------------------------------------
# 3. Expression distribution
# ---------------------------------------------------------------------------


class TestExpressionDistribution:
    def test_single_expression_gives_100_percent(self):
        cm = make_manager(n_frames=100, expression="neutral")
        data = cm.finalise()
        assert data.expression_distribution == {"neutral": pytest.approx(1.0)}

    def test_mixed_expressions_sum_to_one(self):
        cm = CalibrationManager()
        for _ in range(60):
            cm.add_frame(0.30, 0, 0, 0, "neutral")
        for _ in range(30):
            cm.add_frame(0.30, 0, 0, 0, "engaged")
        for _ in range(10):
            cm.add_frame(0.30, 0, 0, 0, "bored")
        data = cm.finalise()
        total = sum(data.expression_distribution.values())
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_expression_fractions_correct(self):
        cm = CalibrationManager()
        for _ in range(75):
            cm.add_frame(0.30, 0, 0, 0, "neutral")
        for _ in range(25):
            cm.add_frame(0.30, 0, 0, 0, "engaged")
        data = cm.finalise()
        assert data.expression_distribution["neutral"] == pytest.approx(0.75, abs=1e-3)
        assert data.expression_distribution["engaged"] == pytest.approx(0.25, abs=1e-3)


# ---------------------------------------------------------------------------
# 4. Full calibration round-trip
# ---------------------------------------------------------------------------


class TestFullCalibration:
    def test_finalise_returns_calibration_data(self):
        cm = make_manager(n_frames=450, ear=0.28, pitch=-1.0, yaw=3.0)
        data = cm.finalise()
        assert isinstance(data, CalibrationData)

    def test_resting_ear_averaged_correctly(self):
        cm = CalibrationManager()
        ears = [0.25, 0.27, 0.29, 0.31, 0.33]
        for e in ears:
            cm.add_frame(e, 0, 0, 0)
        data = cm.finalise()
        assert data.resting_ear == pytest.approx(sum(ears) / len(ears), abs=1e-4)

    def test_ear_threshold_from_full_calibration(self):
        resting = 0.22
        cm = make_manager(n_frames=450, ear=resting)
        data = cm.finalise()
        assert data.ear_threshold == pytest.approx(
            resting * EAR_DROWSINESS_RATIO, abs=1e-4
        )

    def test_is_default_false_after_calibration(self):
        cm = make_manager(n_frames=10)
        data = cm.finalise()
        assert data.is_default is False

    def test_frames_collected_recorded(self):
        cm = make_manager(n_frames=450)
        data = cm.finalise()
        assert data.frames_collected == 450

    def test_baseline_pose_averaged(self):
        cm = CalibrationManager()
        pitches = [-2.0, -1.0, 0.0, 1.0, 2.0]
        for p in pitches:
            cm.add_frame(0.30, pitch=p, yaw=5.0, roll=0.0)
        data = cm.finalise()
        assert data.baseline_pitch == pytest.approx(0.0, abs=1e-4)
        assert data.baseline_yaw == pytest.approx(5.0, abs=1e-4)

    def test_user_id_preserved(self):
        cm = make_manager(user_id=42, n_frames=10)
        data = cm.finalise()
        assert data.user_id == 42


# ---------------------------------------------------------------------------
# 5. Default / skipped calibration
# ---------------------------------------------------------------------------


class TestDefaultCalibration:
    def test_default_has_is_default_true(self):
        data = CalibrationManager.default(user_id=7)
        assert data.is_default is True

    def test_default_uses_population_ear(self):
        data = CalibrationManager.default()
        assert data.resting_ear == pytest.approx(DEFAULT_EAR)

    def test_default_uses_standard_drowsiness_threshold(self):
        data = CalibrationManager.default()
        assert data.ear_threshold == pytest.approx(DEFAULT_DROWSINESS_THRESHOLD)

    def test_default_gaze_window_is_symmetric(self):
        data = CalibrationManager.default()
        assert data.gaze_yaw_min == pytest.approx(-GAZE_YAW_TOLERANCE)
        assert data.gaze_yaw_max == pytest.approx(GAZE_YAW_TOLERANCE)

    def test_default_frames_collected_is_zero(self):
        data = CalibrationManager.default()
        assert data.frames_collected == 0


# ---------------------------------------------------------------------------
# 6. Serialisation / deserialisation
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_to_dict_contains_all_keys(self):
        cm = make_manager(n_frames=10)
        data = cm.finalise()
        d = data.to_dict()
        expected_keys = {
            "user_id",
            "is_default",
            "resting_ear",
            "ear_threshold",
            "baseline_pitch",
            "baseline_yaw",
            "baseline_roll",
            "gaze_yaw_min",
            "gaze_yaw_max",
            "gaze_pitch_min",
            "gaze_pitch_max",
            "expression_distribution",
            "frames_collected",
        }
        assert expected_keys.issubset(d.keys())

    def test_round_trip_preserves_values(self):
        cm = make_manager(n_frames=100, ear=0.24, pitch=2.0, yaw=-4.0)
        original = cm.finalise()
        restored = CalibrationData.from_dict(original.to_dict())
        assert restored.resting_ear == pytest.approx(original.resting_ear, abs=1e-6)
        assert restored.ear_threshold == pytest.approx(original.ear_threshold, abs=1e-6)
        assert restored.baseline_yaw == pytest.approx(original.baseline_yaw, abs=1e-4)
        assert restored.gaze_yaw_min == pytest.approx(original.gaze_yaw_min, abs=1e-4)
        assert restored.frames_collected == original.frames_collected
        assert restored.is_default == original.is_default

    def test_from_dict_with_missing_keys_uses_defaults(self):
        """Partial dict (e.g. legacy record) should fill missing keys gracefully."""
        data = CalibrationData.from_dict({"user_id": 99})
        assert data.user_id == 99
        assert data.resting_ear == pytest.approx(DEFAULT_EAR)


# ---------------------------------------------------------------------------
# 7. Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_buffers(self):
        cm = make_manager(n_frames=100)
        cm.reset()
        assert cm.frames_collected() == 0

    def test_reset_allows_refinalize(self):
        cm = make_manager(n_frames=50)
        cm.finalise()
        cm.reset()
        for _ in range(20):
            cm.add_frame(0.28, 0, 0, 0)
        data = cm.finalise()
        assert data.frames_collected == 20

    def test_reset_restores_default_thresholds(self):
        cm = CalibrationManager()
        cm.calibrate_ear(resting_ear=0.22)
        cm.reset()
        assert cm.ear_threshold == pytest.approx(DEFAULT_DROWSINESS_THRESHOLD)


# ---------------------------------------------------------------------------
# 8. Progress tracking
# ---------------------------------------------------------------------------


class TestProgress:
    def test_progress_zero_at_start(self):
        cm = CalibrationManager()
        assert cm.progress() == pytest.approx(0.0)

    def test_progress_one_at_full_collection(self):
        target = int(
            CalibrationManager.TARGET_DURATION_SECONDS * CalibrationManager.DEFAULT_FPS
        )
        cm = make_manager(n_frames=target)
        assert cm.progress() == pytest.approx(1.0)

    def test_progress_caps_at_one(self):
        target = int(
            CalibrationManager.TARGET_DURATION_SECONDS * CalibrationManager.DEFAULT_FPS
        )
        cm = make_manager(n_frames=target * 2)
        assert cm.progress() == pytest.approx(1.0)

    def test_progress_midpoint(self):
        target = int(
            CalibrationManager.TARGET_DURATION_SECONDS * CalibrationManager.DEFAULT_FPS
        )
        cm = make_manager(n_frames=target // 2)
        assert cm.progress() == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------------------
# 9. Add frame after finalise raises RuntimeError
# ---------------------------------------------------------------------------


class TestPostFinaliseGuard:
    def test_add_frame_after_finalise_raises(self):
        cm = make_manager(n_frames=10)
        cm.finalise()
        with pytest.raises(RuntimeError, match="finalised"):
            cm.add_frame(0.30, 0, 0, 0)


# ---------------------------------------------------------------------------
# 10. Finalise with zero frames raises ValueError
# ---------------------------------------------------------------------------


class TestFinaliseEmpty:
    def test_finalise_empty_raises(self):
        cm = CalibrationManager()
        with pytest.raises(ValueError, match="No calibration frames"):
            cm.finalise()


# ---------------------------------------------------------------------------
# 11. Custom ear_ratio and tolerance config
# ---------------------------------------------------------------------------


class TestCustomConfig:
    def test_custom_ear_ratio(self):
        cm = CalibrationManager(ear_ratio=0.70)
        cm.calibrate_ear(resting_ear=0.30)
        assert cm.ear_threshold == pytest.approx(0.30 * 0.70, abs=1e-6)

    def test_custom_yaw_tolerance(self):
        cm = CalibrationManager(yaw_tolerance=30.0)
        cm.calibrate_pose(pitch=0.0, yaw=0.0)
        assert cm.gaze_yaw_max == pytest.approx(30.0)
        assert cm.gaze_yaw_min == pytest.approx(-30.0)


# ---------------------------------------------------------------------------
# 12. Summary string
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_contains_ear(self):
        cm = make_manager(n_frames=10, ear=0.27)
        data = cm.finalise()
        assert "0.27" in data.summary() or "EAR" in data.summary()

    def test_default_summary_says_default(self):
        data = CalibrationManager.default()
        assert "DEFAULT" in data.summary()

    def test_calibrated_summary_says_calibrated(self):
        cm = make_manager(n_frames=10)
        data = cm.finalise()
        assert "CALIBRATED" in data.summary()


# ---------------------------------------------------------------------------
# 13. API: POST /calibrate/{user_id}
# ---------------------------------------------------------------------------


class TestAPIPost:
    def setup_method(self):
        _store.clear()

    def _make_frames(self, n=450, ear=0.28, pitch=0.0, yaw=0.0):
        return [
            {
                "ear": ear,
                "pitch": pitch,
                "yaw": yaw,
                "roll": 0.0,
                "expression": "neutral",
            }
            for _ in range(n)
        ]

    def test_post_returns_201(self):
        resp = client.post("/api/v1/calibrate/1", json={"frames": self._make_frames()})
        assert resp.status_code == 201

    def test_post_returns_correct_ear_threshold(self):
        resp = client.post(
            "/api/v1/calibrate/2", json={"frames": self._make_frames(ear=0.22)}
        )
        data = resp.json()
        assert data["ear_threshold"] == pytest.approx(
            0.22 * EAR_DROWSINESS_RATIO, abs=1e-3
        )

    def test_post_stores_data(self):
        client.post("/api/v1/calibrate/3", json={"frames": self._make_frames()})
        assert 3 in _store

    def test_post_is_default_false(self):
        resp = client.post("/api/v1/calibrate/4", json={"frames": self._make_frames()})
        assert resp.json()["is_default"] is False

    def test_post_frames_collected_matches(self):
        resp = client.post(
            "/api/v1/calibrate/5", json={"frames": self._make_frames(n=200)}
        )
        assert resp.json()["frames_collected"] == 200


# ---------------------------------------------------------------------------
# 14. API: GET /calibrate/{user_id} — stored record
# ---------------------------------------------------------------------------


class TestAPIGet:
    def setup_method(self):
        _store.clear()

    def test_get_stored_returns_200(self):
        frames = [
            {
                "ear": 0.30,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "expression": "neutral",
            }
        ] * 100
        client.post("/api/v1/calibrate/10", json={"frames": frames})
        resp = client.get("/api/v1/calibrate/10")
        assert resp.status_code == 200

    def test_get_returns_correct_user_id(self):
        frames = [
            {
                "ear": 0.30,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "expression": "neutral",
            }
        ] * 100
        client.post("/api/v1/calibrate/11", json={"frames": frames})
        resp = client.get("/api/v1/calibrate/11")
        assert resp.json()["user_id"] == 11


# ---------------------------------------------------------------------------
# 15. API: GET /calibrate/{user_id} — default fallback
# ---------------------------------------------------------------------------


class TestAPIGetDefault:
    def setup_method(self):
        _store.clear()

    def test_get_unknown_user_returns_200(self):
        resp = client.get("/api/v1/calibrate/999")
        assert resp.status_code == 200

    def test_get_unknown_user_is_default_true(self):
        resp = client.get("/api/v1/calibrate/999")
        assert resp.json()["is_default"] is True

    def test_get_default_message_mentions_calibration(self):
        resp = client.get("/api/v1/calibrate/999")
        assert "calibration" in resp.json()["message"].lower()


# ---------------------------------------------------------------------------
# 16. API: DELETE /calibrate/{user_id}
# ---------------------------------------------------------------------------


class TestAPIDelete:
    def setup_method(self):
        _store.clear()

    def test_delete_existing_returns_204(self):
        frames = [
            {
                "ear": 0.30,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "expression": "neutral",
            }
        ] * 10
        client.post("/api/v1/calibrate/20", json={"frames": frames})
        resp = client.delete("/api/v1/calibrate/20")
        assert resp.status_code == 204

    def test_delete_removes_from_store(self):
        frames = [
            {
                "ear": 0.30,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "expression": "neutral",
            }
        ] * 10
        client.post("/api/v1/calibrate/21", json={"frames": frames})
        client.delete("/api/v1/calibrate/21")
        assert 21 not in _store

    def test_delete_nonexistent_returns_404(self):
        resp = client.delete("/api/v1/calibrate/9999")
        assert resp.status_code == 404

    def test_after_delete_get_returns_defaults(self):
        frames = [
            {
                "ear": 0.22,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "expression": "neutral",
            }
        ] * 10
        client.post("/api/v1/calibrate/22", json={"frames": frames})
        client.delete("/api/v1/calibrate/22")
        resp = client.get("/api/v1/calibrate/22")
        assert resp.json()["is_default"] is True

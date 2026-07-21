import pytest

from src.scoring.temporal_filter import TemporalFilter


def test_stable_input():
    """Stable scores should produce a smoothed output equal to the input."""
    tf = TemporalFilter(window_size=30)

    for _ in range(30):
        smoothed = tf.smooth(85.0)

    assert smoothed == pytest.approx(85.0, abs=0.1)


def test_single_anomaly_does_not_tank_output():
    """A single anomalous frame should not drop smoothed output by more than 5 points."""
    tf = TemporalFilter(window_size=30)

    # Warm up with stable high scores
    for _ in range(30):
        smoothed = tf.smooth(85.0)

    baseline = smoothed  # ~85.0

    # Inject a single zero-score anomaly (e.g., face occluded)
    smoothed_after = tf.smooth(0.0)

    # Must not drop by more than 5 points
    assert baseline - smoothed_after <= 5.0, (
        f"Single anomaly dropped score by {baseline - smoothed_after:.1f} points, "
        f"expected <= 5.0"
    )


def test_sustained_drop_reflected_in_output():
    """A sustained drop lasting 3+ seconds should be fully reflected in the output."""
    tf = TemporalFilter(window_size=30)

    # Warm up at 85
    for _ in range(30):
        tf.smooth(85.0)

    # Sustain 3 seconds of low scores (45 frames at 15 FPS)
    for _ in range(45):
        smoothed = tf.smooth(25.0)

    # Output should be close to 25 (the sustained value)
    assert (
        smoothed <= 35.0
    ), f"Sustained drop not reflected: smoothed={smoothed:.1f}, expected <= 35.0"


def test_ramp_up():
    """Scores ramping up from 0 to 100 should produce increasing smoothed values."""
    tf = TemporalFilter(window_size=30)
    smoothed_values = []

    for i in range(60):
        score = (i / 59) * 100.0
        smoothed_values.append(tf.smooth(score))

    # Later values should be higher than earlier values
    early_avg = sum(smoothed_values[:10]) / 10
    late_avg = sum(smoothed_values[-10:]) / 10
    assert late_avg > early_avg


def test_ramp_down():
    """Scores ramping down from 100 to 0 should produce decreasing smoothed values."""
    tf = TemporalFilter(window_size=30)
    smoothed_values = []

    for i in range(60):
        score = 100.0 - (i / 59) * 100.0
        smoothed_values.append(tf.smooth(score))

    early_avg = sum(smoothed_values[:10]) / 10
    late_avg = sum(smoothed_values[-10:]) / 10
    assert late_avg < early_avg


def test_empty_buffer_returns_raw_score():
    """Before the buffer is warmed up, the first score should be returned directly."""
    tf = TemporalFilter(window_size=30)
    result = tf.smooth(72.0)
    assert result == pytest.approx(72.0, abs=0.1)


def test_memory_bounded_by_window_size():
    """Buffer must never exceed window_size frames."""
    tf = TemporalFilter(window_size=10)

    for i in range(100):
        tf.smooth(float(i))

    assert len(tf._buffer) == 10


def test_reset_clears_state():
    """After reset, the filter should behave as if freshly instantiated."""
    tf = TemporalFilter(window_size=30)

    for _ in range(30):
        tf.smooth(85.0)

    tf.reset()

    # After reset, buffer is empty and first score is returned raw
    result = tf.smooth(50.0)
    assert result == pytest.approx(50.0, abs=0.1)
    assert len(tf._buffer) == 1

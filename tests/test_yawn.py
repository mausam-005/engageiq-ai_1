import numpy as np

from src.detection.yawn import YawnDetector, compute_mar


def test_compute_mar_yawn():
    """Verify that a wide mouth generates a high MAR > 0.6"""
    # [left, right, top, bottom]
    mouth_open = np.array(
        [
            [0, 5],  # left corner
            [10, 5],  # right corner (horizontal dist = 10)
            [5, 10],  # top lip
            [5, 0],  # bottom lip (vertical dist = 10)
        ]
    )
    mar = compute_mar(mouth_open)
    # Ratio = 10 / 10 = 1.0
    assert mar == 1.0


def test_compute_mar_closed():
    """Verify that a closed mouth generates a low MAR < 0.3"""
    mouth_closed = np.array(
        [
            [0, 5],  # left corner
            [10, 5],  # right corner (horizontal dist = 10)
            [5, 6],  # top lip
            [5, 4],  # bottom lip (vertical dist = 2)
        ]
    )
    mar = compute_mar(mouth_closed)
    # Ratio = 2 / 10 = 0.2
    assert mar == 0.2


def test_speech_filtering():
    """Verify that a high MAR for a brief duration (speech) does not trigger a yawn."""
    detector = YawnDetector(mar_threshold=0.6, yawn_duration=2.0)

    # Time 0.0: Mouth opens wide
    is_yawning = detector.update(mar=0.8, current_time=0.0)
    assert not is_yawning

    # Time 1.0: Mouth still open, but hasn't reached 2.0s threshold
    is_yawning = detector.update(mar=0.8, current_time=1.0)
    assert not is_yawning

    # Time 1.5: Mouth closes (speech finished)
    is_yawning = detector.update(mar=0.2, current_time=1.5)
    assert not is_yawning

    # Time 2.5: Mouth stays closed
    is_yawning = detector.update(mar=0.2, current_time=2.5)
    assert not is_yawning


def test_yawn_detection():
    """Verify that a high MAR sustained for >= duration triggers a yawn."""
    detector = YawnDetector(mar_threshold=0.6, yawn_duration=2.0)

    # Time 0.0: Yawn begins
    assert not detector.update(mar=0.8, current_time=0.0)

    # Time 1.0: Still yawning
    assert not detector.update(mar=0.8, current_time=1.0)

    # Time 2.1: Crossed the duration threshold!
    assert detector.update(mar=0.8, current_time=2.1)

    # Time 3.0: Still yawning
    assert detector.update(mar=0.8, current_time=3.0)

    # Time 3.5: Mouth closes, yawn finishes
    assert not detector.update(mar=0.2, current_time=3.5)


def test_fatigue_assessment():
    """Verify that 3 yawns in 10 minutes triggers fatigue."""
    detector = YawnDetector(mar_threshold=0.6, yawn_duration=2.0)

    # We can use the exposed `record_yawn` for testing the downstream logic
    current_time = 1000.0  # seconds

    # 1st yawn at t=600 (within last 10 mins = 600 seconds window: 400->1000)
    detector.record_yawn(timestamp=600.0)
    assert not detector.is_fatigued(current_time=current_time)

    # 2nd yawn at t=800
    detector.record_yawn(timestamp=800.0)
    assert not detector.is_fatigued(current_time=current_time)

    # A yawn that is too old (t=300) should be ignored
    detector.record_yawn(timestamp=300.0)
    assert not detector.is_fatigued(current_time=current_time)

    # 3rd valid yawn at t=950
    detector.record_yawn(timestamp=950.0)

    # We now have 3 yawns within the window [400, 1000]
    assert detector.is_fatigued(current_time=current_time)

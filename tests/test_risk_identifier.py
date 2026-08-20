from src.analytics.risk_identifier import RiskIdentifier


def test_empty_scores_returns_false():
    ri = RiskIdentifier()
    result = ri.evaluate("anon_1", [])
    assert not result.at_risk
    assert not result.declining_trend
    assert result.reason is None


def test_at_risk_consecutive_drops():
    ri = RiskIdentifier(threshold=50.0, consecutive_sessions=3)
    # 3 consecutive sessions below 50
    result = ri.evaluate("anon_42", [40.0, 35.0, 30.0])
    assert result.at_risk is True
    assert result.reason == "below_threshold_3_sessions"


def test_not_at_risk_inconsistent():
    ri = RiskIdentifier(threshold=50.0, consecutive_sessions=3)
    # 3 sessions, but not all below 50
    result = ri.evaluate("anon_43", [50.0, 80.0, 70.0])
    assert result.at_risk is False
    assert result.reason is None


def test_declining_trend_detected():
    ri = RiskIdentifier()
    # older half (80, 75) avg = 77.5
    # newer half (70, 65, 60) avg = 65.0
    # drop is (77.5 - 65) / 77.5 = 16.1%, which is > 10%
    result = ri.evaluate("anon_44", [80.0, 75.0, 70.0, 65.0, 60.0])
    assert result.declining_trend is True


def test_declining_trend_not_detected_if_drop_small():
    ri = RiskIdentifier()
    # older half (80, 75) avg = 77.5
    # newer half (75, 74, 75) avg = 74.6
    # drop is (77.5 - 74.6) / 77.5 = 3.7%, which is < 10%
    result = ri.evaluate("anon_45", [80.0, 75.0, 75.0, 74.0, 75.0])
    assert result.declining_trend is False


def test_recovering_trend_not_flagged():
    ri = RiskIdentifier()
    # older half (40, 45) avg = 42.5
    # newer half (70, 75, 80) avg = 75.0
    # no drop, it's improving
    result = ri.evaluate("anon_46", [40.0, 45.0, 70.0, 75.0, 80.0])
    assert result.declining_trend is False
    assert result.at_risk is False


def test_at_risk_and_declining():
    ri = RiskIdentifier(threshold=50.0, consecutive_sessions=3)
    # older half (60, 55) avg = 57.5
    # newer half (45, 40, 35) avg = 40.0 (drop > 10%)
    # last 3 sessions are all < 50, so at_risk=True
    result = ri.evaluate("anon_47", [60.0, 55.0, 45.0, 40.0, 35.0])
    assert result.declining_trend is True
    assert result.at_risk is True
    assert result.reason == "below_threshold_3_sessions"


def test_requires_at_least_four_sessions_for_trend():
    ri = RiskIdentifier()
    # Only 3 sessions -> no trend evaluation
    result = ri.evaluate("anon_48", [80.0, 60.0, 40.0])
    assert result.declining_trend is False

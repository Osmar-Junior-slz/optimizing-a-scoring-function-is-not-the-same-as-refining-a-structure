from docking_confidence.domain.preservation import PreservationRule


def test_preservation_rule():
    rule = PreservationRule(0.23, 0.05)
    assert rule.label(0.30, 0.34) == 1
    assert rule.label(0.22, 0.24) == 0
    assert rule.label(0.30, 0.36) == 0

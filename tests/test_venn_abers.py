import numpy as np
import pytest
from copy import deepcopy

from docking_confidence.calibration.venn_abers import VennAbersCalibrator


def test_tied_score_golden_case_and_log_minimax_point():
    calibrator = VennAbersCalibrator().fit(
        np.array([0.0, 0.0, 1.0]), np.array([0, 1, 1])
    )
    lower, upper, point = calibrator.predict_interval(np.array([0.0]))
    assert lower[0] == pytest.approx(1 / 3)
    assert upper[0] == pytest.approx(2 / 3)
    assert point[0] == pytest.approx(1 / 2)


def test_permutation_and_export_round_trip_do_not_change_predictions():
    scores = np.array([-2.0, -1.0, -1.0, 0.5, 3.0, 4.0])
    labels = np.array([0, 1, 0, 1, 0, 1])
    queries = np.array([-3.0, -1.0, 0.0, 5.0])
    first = VennAbersCalibrator().fit(scores, labels)
    permutation = np.array([4, 1, 5, 0, 3, 2])
    second = VennAbersCalibrator().fit(scores[permutation], labels[permutation])
    restored = VennAbersCalibrator.from_export(first.export())
    for candidate in (second, restored):
        for actual, expected in zip(
            candidate.predict_interval(queries), first.predict_interval(queries)
        ):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_venn_abers_rejects_non_finite_and_one_class_calibration():
    with pytest.raises(ValueError, match="finite"):
        VennAbersCalibrator().fit(np.array([0.0, np.inf]), np.array([0, 1]))
    with pytest.raises(ValueError, match="both binary classes"):
        VennAbersCalibrator().fit(np.array([0.0, 1.0]), np.array([1, 1]))


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        (("score_groups", 0, "weight"), 1.5),
        (("score_groups", 0, "positive_count"), True),
        (("calibration_example_count",), 3.0),
        (("calibration_positive_count",), False),
    ],
)
def test_import_rejects_non_integer_and_boolean_counts(path, invalid):
    payload = VennAbersCalibrator().fit(
        np.array([0.0, 0.0, 1.0]), np.array([0, 1, 1])
    ).export()
    corrupted = deepcopy(payload)
    target = corrupted
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid
    with pytest.raises(ValueError, match="must be an integer"):
        VennAbersCalibrator.from_export(corrupted)

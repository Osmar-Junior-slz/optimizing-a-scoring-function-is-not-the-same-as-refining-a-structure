import numpy as np
import pytest

from docking_confidence.validation.split import four_way_group_partition


def test_group_partition_is_disjoint_deterministic_and_label_independent():
    groups = np.repeat([f"C{index:02d}" for index in range(12)], 2)
    labels = np.tile([0, 1], 12)
    first = four_way_group_partition(labels, groups, split_folds=4, random_seed=17)
    changed_labels = 1 - labels
    second = four_way_group_partition(
        changed_labels, groups, split_folds=4, random_seed=17
    )
    for role in (
        "model_train",
        "probability_calibration",
        "risk_calibration",
        "locked_test",
    ):
        np.testing.assert_array_equal(getattr(first, role), getattr(second, role))

    sets = [set(values) for values in first.groups_by_role.values()]
    assert set.union(*sets) == set(groups)
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            assert sets[left].isdisjoint(sets[right])

    assert (
        first.export()["method"]
        == "label_independent_assignment_with_postsplit_feasibility_gate_v1"
    )
    assert first.export()["grouping_unit"] == "complex_fingerprint_sha256"


def test_group_partition_fails_closed_after_one_class_feasibility_failure():
    groups = np.array([f"C{index:02d}" for index in range(12)])
    labels = np.zeros(len(groups), dtype=int)
    with pytest.raises(ValueError, match="fail-closed") as error:
        four_way_group_partition(labels, groups, split_folds=4, random_seed=17)
    assert "choose" not in str(error.value).lower()
    assert "retry" not in str(error.value).lower()

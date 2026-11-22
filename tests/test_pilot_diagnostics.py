import numpy as np
import pytest

from docking_confidence.application.pilot_diagnostics import (
    CANDIDATES, PilotCandidate, fit_pilot_fold, grouped_pilot_partitions,
    transform_features,
)


def test_assessment_labels_cannot_change_fitted_model_or_predictions():
    X = np.r_[np.linspace(0, 1, 12), np.linspace(.2, .8, 8)].reshape(-1, 1)
    y = np.tile([0, 1], 10)
    kwargs = dict(
        columns=["viable_fraction_mean"], candidate=CANDIDATES[0],
        training=np.arange(12), calibration=np.arange(12, 16), assessment=np.arange(16, 20),
    )
    first = fit_pilot_fold(X, y, **kwargs)
    changed = y.copy()
    changed[16:] = 1 - changed[16:]
    second = fit_pilot_fold(X, changed, **kwargs)
    assert first["model"] == second["model"]
    for key in ["raw_score", "before_probability", "after_probability", "probability_lower"]:
        np.testing.assert_array_equal(first[key], second[key])


def test_molecular_aliases_never_leak_and_every_group_is_assessed_once():
    groups = np.repeat([f"group-{i:02}" for i in range(25)], 3)
    folds = grouped_pilot_partitions(groups)
    assessed = []
    for fold in folds:
        roles = fold["groups"]
        assert not (set(roles["model_train"]) & set(roles["outer_assessment"]))
        assert not (set(roles["probability_calibration"]) & set(roles["outer_assessment"]))
        for group in np.unique(groups):
            role_memberships = [group in members for members in roles.values()]
            assert sum(role_memberships) == 1
        assessed.extend(roles["outer_assessment"])
    assert sorted(assessed) == sorted(np.unique(groups))


def test_one_class_calibration_stays_unavailable_instead_of_inventing_probability():
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    y = np.r_[np.tile([0, 1], 5), np.ones(10, dtype=int)]
    result = fit_pilot_fold(
        X, y, columns=["viable_fraction_mean"], candidate=CANDIDATES[0],
        training=np.arange(10), calibration=np.arange(10, 15), assessment=np.arange(15, 20),
    )
    assert result["after_probability"] is None
    assert result["probability_lower"] is None
    assert result["calibration_unavailable_reason"]
    assert np.isfinite(result["before_probability"]).all()


@pytest.mark.parametrize("column", ["dockq_full", "dockq_reduced", "preserved", "reference_quality"])
def test_reference_or_outcome_cannot_become_a_pilot_feature(column):
    with pytest.raises(ValueError, match="allowlist"):
        transform_features(np.ones((4, 1)), [column], CANDIDATES[0])


def test_log_counts_changes_only_declared_counts_and_never_mutates_input():
    X = np.array([[.2, 9.0], [.8, 99.0]])
    before = X.copy()
    result = transform_features(
        X, ["viable_fraction_mean", "graph_interface_edge_count"],
        PilotCandidate("test", .1, None, True),
    )
    np.testing.assert_array_equal(X, before)
    np.testing.assert_array_equal(result[:, 0], X[:, 0])
    np.testing.assert_allclose(result[:, 1], np.log1p(X[:, 1]))

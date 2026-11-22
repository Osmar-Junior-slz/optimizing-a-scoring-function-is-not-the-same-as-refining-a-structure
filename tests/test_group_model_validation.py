"""Grouped validation evaluates fixed model inputs without selection or leakage."""
import numpy as np
import pytest

from docking_confidence.modeling import group_validation
from docking_confidence.modeling.hybrid import HybridBaseModel, METHOD, PARAMETERS


def _rows():
    groups = np.repeat(["a", "b", "c", "d", "e", "f"], [1, 2, 3, 4, 5, 6])
    X = np.column_stack([np.arange(len(groups)), np.linspace(-1, 1, len(groups))])
    y = np.linspace(-.3, .6, len(groups))
    return X, y, groups


class RecordingZeroModel:
    calls = []

    def fit(self, X, y, groups):
        self.training_rows = set(X[:, 0])
        self.calls.append((X.copy(), y.copy(), groups.copy()))
        return self

    def raw_score(self, X):
        assert not self.training_rows & set(X[:, 0])
        return np.zeros(len(X))


@pytest.fixture
def fake_model(monkeypatch):
    RecordingZeroModel.calls = []
    monkeypatch.setattr(group_validation, "HybridBaseModel", RecordingZeroModel)
    return RecordingZeroModel


def _equal_complex_mean(values, groups):
    return np.mean([np.mean(values[groups == group]) for group in np.unique(groups)])


def test_group_splits_cover_each_complex_once_and_forward_all_fixed_inputs(fake_model):
    X, y, groups = _rows()
    prediction, report = group_validation.group_cross_validation(X, y, groups)
    np.testing.assert_array_equal(prediction, np.zeros(len(y)))
    assert report["available"] and len(report["folds"]) == 5
    assert report["feature_count"] == X.shape[1]
    assert report["row_count"] == len(y)
    assert report["independent_complex_count"] == len(set(groups))
    assert report["feature_selection"] is False
    assert report["hyperparameter_search"] is False
    assert report["calibration_or_test_used"] is False
    assert report["parameters"] == PARAMETERS
    assert report["base_model_method"] == METHOD
    visited = []
    for fold, (train_X, train_y, train_groups) in zip(report["folds"], fake_model.calls):
        assert set(train_groups) == set(fold["training_groups"])
        assert not set(fold["training_groups"]) & set(fold["validation_groups"])
        assert set(fold["training_groups"]) | set(fold["validation_groups"]) == set(groups)
        rows = np.isin(groups, fold["training_groups"])
        np.testing.assert_array_equal(train_X, X[rows])
        np.testing.assert_array_equal(train_y, y[rows])
        np.testing.assert_array_equal(train_groups, groups[rows])
        visited.extend(fold["validation_groups"])
    assert sorted(visited) == sorted(set(groups))


def test_report_parameters_do_not_share_mutable_state_with_fixed_hybrid(fake_model):
    X, y, groups = _rows()
    _, report = group_validation.group_cross_validation(X, y, groups)
    report["parameters"]["weights"]["ridge"] = .9
    report["parameters"]["ridge"]["alpha"] = 99.
    report["parameters"]["extra_trees"]["max_depth"] = 99
    assert PARAMETERS["weights"] == {"ridge": .5, "extra_trees": .5}
    assert PARAMETERS["ridge"]["alpha"] == 1.
    assert PARAMETERS["extra_trees"]["max_depth"] == 4


def test_losses_and_fold_training_mean_give_equal_weight_per_complex(fake_model):
    X, y, groups = _rows()
    prediction, report = group_validation.group_cross_validation(X, y, groups)
    expected_mse = _equal_complex_mean((prediction - y) ** 2, groups)
    assert report["group_mse"] == pytest.approx(expected_mse)
    assert report["group_mae"] == pytest.approx(_equal_complex_mean(np.abs(prediction - y), groups))
    assert report["group_mse"] != pytest.approx(np.mean(y ** 2))
    baseline = np.empty(len(y))
    for fold in report["folds"]:
        train = np.isin(groups, fold["training_groups"])
        validation = np.isin(groups, fold["validation_groups"])
        baseline[validation] = _equal_complex_mean(y[train], groups[train])
    expected_baseline_mse = _equal_complex_mean((baseline - y) ** 2, groups)
    assert report["training_mean_baseline_group_mse"] == pytest.approx(expected_baseline_mse)
    assert report["skill_against_fold_training_mean"] == pytest.approx(1 - expected_mse / expected_baseline_mse)
    assert report["predicted_negative_rows"] == 0
    assert report["observed_negative_rows"] == int((y < 0).sum())


def test_single_complex_returns_unavailable_without_fitting(fake_model):
    X, y, _ = _rows()
    prediction, report = group_validation.group_cross_validation(X, y, ["one"] * len(y))
    assert prediction is None and report["available"] is False
    assert report["reason"] == "fewer_than_two_proper_training_complexes"
    assert report["folds"] == [] and not fake_model.calls
    assert report["feature_count"] == 2


def test_constant_zero_target_reports_undefined_relative_skill(fake_model):
    X, _, groups = _rows()
    _, report = group_validation.group_cross_validation(X, np.zeros(len(X)), groups)
    assert report["group_mse"] == 0
    assert report["training_mean_baseline_group_mse"] == 0
    assert report["skill_against_fold_training_mean"] is None


@pytest.mark.parametrize("groups", [None, ["a"], ["a", None], ["a", np.nan], ["a", " "], [["a"], ["b"]]])
def test_missing_invalid_and_misaligned_molecular_groups_are_rejected(groups):
    with pytest.raises(ValueError):
        group_validation.group_cross_validation([[0.], [1.]], [0., .1], groups)


@pytest.mark.parametrize("X,y", [
    ([], []), ([0., 1.], [0., 1.]), ([[np.inf], [0.]], [0., 1.]),
    ([[0.], [1.]], [np.nan, 1.]), ([[0.], [1.]], [1.]),
    ([[0.], [1.]], [[0.], [1.]]),
])
def test_invalid_features_and_outcomes_are_rejected(X, y):
    with pytest.raises(ValueError):
        group_validation.group_cross_validation(X, y, ["a", "b"])


@pytest.mark.parametrize("returned", [[np.nan], [[0.]], [0., 0.]])
def test_malformed_model_predictions_are_rejected(monkeypatch, returned):
    class BadModel:
        def fit(self, *args):
            return self
        def raw_score(self, X):
            return returned
    monkeypatch.setattr(group_validation, "HybridBaseModel", BadModel)
    with pytest.raises(ValueError, match="predictions must be finite and aligned"):
        group_validation.group_cross_validation([[0.], [1.]], [0., 1.], ["a", "b"])


def test_real_fixed_hybrid_validation_accepts_fifteen_inputs_and_is_deterministic():
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(30, 15))
    y = .15 * X[:, 0] + .1 * X[:, 3] ** 2
    groups = np.repeat(["a", "b", "c"], 10)
    first, report = group_validation.group_cross_validation(X, y, groups)
    second, repeated = group_validation.group_cross_validation(X, y, groups)
    np.testing.assert_array_equal(first, second)
    assert repeated == report
    assert first.shape == y.shape and np.isfinite(first).all()
    assert report["feature_count"] == 15
    assert len(report["folds"]) == 3
    fold = report["folds"][0]
    training = np.isin(groups, fold["training_groups"])
    validation = np.isin(groups, fold["validation_groups"])
    model = HybridBaseModel().fit(X[training], y[training], groups[training])
    np.testing.assert_array_equal(first[validation], model.raw_score(X[validation]))
    components = model.component_scores(X[validation])
    np.testing.assert_array_equal(first[validation], .5 * components["ridge"] + .5 * components["extra_trees"])

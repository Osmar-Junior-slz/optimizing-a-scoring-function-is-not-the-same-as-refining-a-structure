"""One raw margin score combines fixed linear and nonlinear components."""
from copy import deepcopy
import json

import numpy as np
import pytest

from docking_confidence.modeling.extra_trees import ExtraTreesBaseModel, complex_weights
from docking_confidence.modeling.hybrid import (
    COMBINATION, METHOD, PARAMETERS, WEIGHTING, HybridBaseModel,
    component_scores_from_export, predict_from_export, validate_export,
)
from docking_confidence.modeling.ridge import RidgeBaseModel


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(2026)
    groups = np.repeat([f"complex-{i}" for i in range(12)], [2, 3, 4, 5] * 3)
    X = rng.normal(size=(len(groups), 15))
    X[:, -1] = 7.0
    y = .12 * X[:, 0] + .18 * np.square(X[:, 1]) - .08
    names = [f"descriptor-{i}" for i in range(X.shape[1])]
    model = HybridBaseModel().fit(X, y, groups)
    return model, model.export(names), X, y, groups, rng.normal(size=(9, 15))


def test_hybrid_export_replays_components_and_equal_mean_without_fitting(fitted, monkeypatch):
    model, payload, _, _, _, query = fitted
    scores = model.component_scores(query)
    expected = .5 * scores["ridge"] + .5 * scores["extra_trees"]
    np.testing.assert_array_equal(model.raw_score(query), expected)
    saved = json.loads(json.dumps(payload, allow_nan=False))
    assert saved["method"] == METHOD
    assert saved["parameters"] == PARAMETERS
    assert saved["combination"] == COMBINATION
    assert saved["weighting"] == WEIGHTING
    assert len(saved["feature_columns"]) == 15
    assert set(saved["components"]) == {"ridge", "extra_trees"}
    assert "calibration" not in saved

    def forbidden(*args, **kwargs):
        raise AssertionError("exported scores must not refit either component")

    monkeypatch.setattr(RidgeBaseModel, "fit", forbidden)
    monkeypatch.setattr(ExtraTreesBaseModel, "fit", forbidden)
    restored = component_scores_from_export(saved, query)
    for component in scores:
        np.testing.assert_allclose(restored[component], scores[component], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(predict_from_export(saved, query), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(predict_from_export(saved, query), .5 * restored["ridge"] + .5 * restored["extra_trees"])


def test_hybrid_weights_are_fixed_and_parameter_exports_are_independent(fitted):
    model, payload, _, _, _, _ = fitted
    exported = deepcopy(payload)
    exported["parameters"]["weights"]["ridge"] = .9
    assert model.parameters == PARAMETERS
    assert payload["parameters"]["weights"] == {"ridge": .5, "extra_trees": .5}
    assert model.export(payload["feature_columns"]) == payload
    with pytest.raises(ValueError, match="fixed"):
        validate_export(exported)


def test_component_scaling_uses_equal_complex_weights_only_from_training(fitted):
    model, payload, X, y, groups, query = fitted
    weights = complex_weights(groups, len(X))
    mean = np.average(X, axis=0, weights=weights)
    scale = np.sqrt(np.average(np.square(X - mean), axis=0, weights=weights))
    scale[-1] = 1.0
    np.testing.assert_allclose(payload["components"]["ridge"]["scaler_mean"], mean, atol=1e-12)
    np.testing.assert_allclose(payload["components"]["ridge"]["scaler_scale"], scale, atol=1e-12)
    # Independent weighted Ridge normal equations, including the intercept.
    scaled = (X - mean) / scale
    y_mean = np.average(y, weights=weights)
    coefficients = np.linalg.solve(
        scaled.T @ (weights[:, None] * scaled) + np.eye(X.shape[1]),
        scaled.T @ (weights * (y - y_mean)),
    )
    expected = ((query - mean) / scale) @ coefficients + y_mean
    np.testing.assert_allclose(model.component_scores(query)["ridge"], expected, rtol=1e-11, atol=1e-11)
    assert payload["components"]["ridge"]["weighting"] == WEIGHTING
    for tree in model.extra_trees.model.estimators_:
        assert tree.tree_.weighted_n_node_samples[0] == pytest.approx(weights.sum())
        assert tree.tree_.value[0, 0, 0] == pytest.approx(y_mean)
    before = model.export(payload["feature_columns"])
    model.raw_score(query * 1e3)
    assert model.export(payload["feature_columns"]) == before


def test_hybrid_is_deterministic_and_accepts_implicit_row_groups(fitted):
    model, payload, X, y, groups, query = fitted
    second = HybridBaseModel().fit(X, y, groups)
    assert second.export(payload["feature_columns"]) == payload
    np.testing.assert_array_equal(model.raw_score(query), second.raw_score(query))
    ungrouped = HybridBaseModel().fit(X, y)
    unique_groups = [f"row-{index}" for index in range(len(X))]
    explicit = HybridBaseModel().fit(X, y, unique_groups)
    assert ungrouped.export(payload["feature_columns"]) == explicit.export(payload["feature_columns"])


@pytest.mark.parametrize("outcome", [-.25, 0., .75])
def test_constant_margin_does_not_require_two_classes(outcome):
    X = np.array([[1., 3.], [2., 4.], [9., -2.]])
    model = HybridBaseModel().fit(X, np.full(len(X), outcome), ["a", "a", "b"])
    names = ["first", "second"]
    query = np.array([[100., -50.], [0., 0.]])
    np.testing.assert_allclose(model.raw_score(query), outcome, atol=1e-12)
    np.testing.assert_allclose(predict_from_export(model.export(names), query), outcome, atol=1e-12)


@pytest.mark.parametrize("path,value", [
    (("method",), "other"),
    (("parameters", "weights", "ridge"), .4),
    (("parameters", "weights", "extra_trees"), .6),
    (("parameters", "ridge", "alpha"), 2.0),
    (("parameters", "ridge", "alpha"), True),
    (("parameters", "extra_trees", "max_depth"), 3),
    (("weighting",), "uniform_rows"),
    (("combination",), "mean_of_probabilities"),
    (("feature_columns",), ["duplicate", "duplicate"]),
    (("components", "ridge", "method"), "logistic_regression"),
    (("components", "ridge", "weighting"), "uniform_rows"),
    (("components", "ridge", "alpha"), -1.0),
    (("components", "ridge", "alpha"), True),
    (("components", "ridge", "solver"), "auto"),
    (("components", "ridge", "fit_intercept"), False),
    (("components", "ridge", "score_orientation"), "probability"),
    (("components", "ridge", "intercept"), np.nan),
    (("components", "ridge", "intercept"), "0.1"),
    (("components", "ridge", "scaler_mean"), [0.]),
    (("components", "ridge", "coef"), [0.]),
    (("components", "ridge", "original_scale_intercept"), 1e9),
    (("components", "extra_trees", "parameters", "max_depth"), 3),
    (("components", "extra_trees", "feature_columns"), ["different"]),
])
def test_export_rejects_tampered_components_parameters_and_mixed_units(fitted, path, value):
    saved = deepcopy(fitted[1])
    current = saved
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    with pytest.raises(ValueError):
        validate_export(saved)


def test_export_rejects_nonpositive_scales_nonfinite_vectors_and_extra_components(fitted):
    for column, value in (("scaler_scale", 0.), ("coef", float("inf")), ("scaler_mean", True)):
        saved = deepcopy(fitted[1])
        saved["components"]["ridge"][column][0] = value
        with pytest.raises(ValueError):
            validate_export(saved)
    saved = deepcopy(fitted[1])
    saved["components"]["other"] = {}
    with pytest.raises(ValueError, match="exactly"):
        validate_export(saved)


@pytest.mark.parametrize("query", [
    np.ones(15), np.empty((0, 15)), np.ones((1, 14)),
    np.full((1, 15), np.nan), np.full((1, 15), np.inf), np.full((1, 15), 1e100),
])
def test_queries_reject_missing_nonfinite_and_mismatched_inputs(fitted, query):
    with pytest.raises(ValueError):
        fitted[0].raw_score(query)
    with pytest.raises(ValueError):
        predict_from_export(fitted[1], query)


@pytest.mark.parametrize("X,y,groups", [
    (np.ones((2, 2)), [1.], ["a", "b"]),
    (np.ones((2, 2)), [1., np.nan], ["a", "b"]),
    (np.ones((2, 2)), [[1.], [2.]], ["a", "b"]),
    (np.ones((2, 2)), [1., 2.], ["a"]),
    (np.ones((2, 2)), [1., 2.], ["a", ""]),
    (np.ones((2, 2)), [1., 2.], ["a", None]),
])
def test_invalid_training_shapes_outcomes_and_molecular_groups_are_rejected(X, y, groups):
    with pytest.raises(ValueError):
        HybridBaseModel().fit(X, y, groups)


def test_unfitted_and_mismatched_feature_exports_are_rejected(fitted):
    model = HybridBaseModel()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.raw_score(np.ones((1, 15)))
    with pytest.raises(RuntimeError, match="not fitted"):
        model.export([f"d-{i}" for i in range(15)])
    with pytest.raises(ValueError, match="feature names"):
        fitted[0].export(["too-few"])

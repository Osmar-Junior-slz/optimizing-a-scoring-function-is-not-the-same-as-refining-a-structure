"""Nonlinear margin fitting, equal-complex weights and JSON replay."""
from copy import deepcopy
import json

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor

from docking_confidence.modeling.extra_trees import (
    ExtraTreesBaseModel, complex_weights,
    predict_from_export, validate_export,
)


def _data():
    rng = np.random.default_rng(340)
    X = rng.normal(size=(72, 3))
    y = .2 * X[:, 0] * X[:, 1] + .1 * np.sin(X[:, 2])
    groups = np.repeat([f"complex-{i}" for i in range(24)], 3)
    return X, y, groups


def _payload():
    X, y, groups = _data()
    return ExtraTreesBaseModel(n_estimators=8).fit(X, y, groups).export(["a", "b", "c"])


def test_fitting_matches_sklearn_with_independently_computed_equal_complex_weights():
    X, y, _ = _data()
    groups = np.repeat(["short", "medium", "long"], [7, 23, 42])
    model = ExtraTreesBaseModel(n_estimators=8).fit(X, y, groups)
    counts = {group: int((groups == group).sum()) for group in set(groups)}
    expected_weights = np.array([len(X) / len(counts) / counts[group] for group in groups])
    expected = ExtraTreesRegressor(**model.parameters).fit(X, y, sample_weight=expected_weights)
    queries = np.random.default_rng(345).normal(size=(20, 3))
    np.testing.assert_array_equal(model.raw_score(queries), expected.predict(queries))
    payload = model.export(["a", "b", "c"])
    restored = json.loads(json.dumps(payload, allow_nan=False))
    assert validate_export(restored) == payload
    np.testing.assert_array_equal(predict_from_export(restored, queries), expected.predict(queries))


def test_json_replay_matches_sklearn_at_float32_sensitive_thresholds():
    X = np.array([[0.], [2.]])
    y = np.array([-.3, .4])
    model = ExtraTreesBaseModel(n_estimators=1, max_depth=1, min_samples_leaf=1).fit(X, y)
    # sklearn converts query input to float32 before comparing with its float64
    # thresholds. This first query is > threshold in float64, but routes left.
    threshold = 1.00000005
    model.model.estimators_[0].tree_.threshold[0] = threshold
    queries = np.array([[1.000000055], [np.nextafter(np.float32(1.), np.float32(np.inf))], [.999]])
    assert queries[0, 0] > threshold
    assert np.float32(queries[0, 0]) <= threshold
    payload = json.loads(json.dumps(model.export(["x"])))
    expected = model.raw_score(queries)
    np.testing.assert_allclose(expected, [-.3, .4, -.3], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(predict_from_export(payload, queries), expected)


def test_fixed_regularized_tree_family_can_learn_nonlinear_xor_without_extra_features():
    corners = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    X = np.repeat(corners, 30, axis=0)
    y = np.repeat([-.4, .4, .4, -.4], 30)
    groups = np.array([f"complex-{i}" for i in range(len(X))])
    model = ExtraTreesBaseModel(n_estimators=8).fit(X, y, groups)
    prediction = model.raw_score(corners)
    np.testing.assert_allclose(prediction, [-.4, .4, .4, -.4], atol=1e-12)
    assert model.parameters["max_depth"] == 4 and model.parameters["min_samples_leaf"] == 5


def test_constant_continuous_outcomes_fit_and_replay():
    X, _, groups = _data()
    target = np.full(len(X), -.125)
    model = ExtraTreesBaseModel(n_estimators=8).fit(X, target, groups)
    exported = model.export(["a", "b", "c"])
    queries = np.array([[1., -2., .3], [0., 0., 0.]])
    np.testing.assert_allclose(model.raw_score(queries), [-.125, -.125], atol=1e-15)
    np.testing.assert_allclose(predict_from_export(exported, queries), [-.125, -.125], atol=1e-15)


def test_each_complex_has_equal_total_weight_independent_of_run_count():
    groups = np.array(["a", "b", "b", "c", "c", "c"])
    weights = complex_weights(groups, len(groups))
    np.testing.assert_allclose(weights, [2., 1., 1., 2 / 3, 2 / 3, 2 / 3])
    assert weights.sum() == pytest.approx(len(groups))
    for group in set(groups):
        assert weights[groups == group].sum() == pytest.approx(2.)
    np.testing.assert_array_equal(complex_weights(None, 3), np.ones(3))


@pytest.mark.parametrize("groups", [["a"], [["a"], ["b"]], ["a", ""], ["a", "  "], ["a", None], ["a", np.nan]])
def test_invalid_missing_or_misaligned_molecular_groups_are_rejected(groups):
    with pytest.raises(ValueError):
        complex_weights(groups, 2)


@pytest.mark.parametrize("changes", [
    {"method": "other"}, {"input_dtype": "float64"}, {"aggregation": "median"},
    {"feature_columns": ["a", "a", "c"]}, {"feature_columns": []},
    {"trees": []}, {"weighting": "uniform_rows"}, {"feature_importances": [np.nan, 0., 0.]},
])
def test_invalid_serialized_model_contract_is_rejected(changes):
    payload = _payload()
    payload.update(changes)
    with pytest.raises(ValueError):
        validate_export(payload)


@pytest.mark.parametrize("field,value", [
    ("children_left", 0), ("children_left", -1), ("children_left", 99999),
    ("children_left", True), ("children_left", 1.5),
    ("children_right", 0), ("feature", -1), ("feature", 3),
    ("threshold", np.nan), ("threshold", np.inf), ("value", np.nan), ("value", np.inf),
])
def test_cycles_invalid_indices_and_nonfinite_tree_arrays_are_rejected(field, value):
    payload = _payload()
    tree = payload["trees"][0]
    assert tree["children_left"][0] >= 1
    tree[field][0] = value
    with pytest.raises(ValueError):
        validate_export(payload)


def test_disconnected_nodes_malformed_arrays_and_tree_count_are_rejected():
    first = _payload()
    malformed = []
    disconnected = deepcopy(first)
    tree = disconnected["trees"][0]
    for key, value in [("children_left", -1), ("children_right", -1), ("feature", -2), ("threshold", -2.), ("value", .1)]:
        tree[key].append(value)
    malformed.append(disconnected)
    unaligned = deepcopy(first)
    unaligned["trees"][0]["feature"].pop()
    malformed.append(unaligned)
    wrong_type = deepcopy(first)
    wrong_type["trees"][0]["value"] = None
    malformed.append(wrong_type)
    wrong_leaf = deepcopy(first)
    tree = wrong_leaf["trees"][0]
    leaf = tree["children_left"].index(-1)
    tree["feature"][leaf] = 0
    malformed.append(wrong_leaf)
    wrong_count = deepcopy(first)
    wrong_count["parameters"]["n_estimators"] += 1
    malformed.append(wrong_count)
    for candidate in malformed:
        with pytest.raises(ValueError):
            validate_export(candidate)


@pytest.mark.parametrize("parameters", [
    {"n_estimators": 0}, {"n_estimators": True}, {"min_samples_leaf": 0},
    {"max_depth": 0}, {"max_depth": True}, {"bootstrap": True}, {"n_jobs": 2}, {"n_jobs": True},
    {"unknown": 1}, {"max_features": 0}, {"max_features": np.nan}, {"random_state": -1},
])
def test_invalid_model_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        ExtraTreesBaseModel(**parameters)


@pytest.mark.parametrize("X", [np.array([]), np.ones((0, 2)), np.ones((3, 0)),
                              np.array([[np.nan], [1.]]), np.array([[np.inf], [1.]]),
                              np.array([[1e100], [1.]])])
def test_invalid_or_unrepresentable_feature_arrays_are_rejected(X):
    with pytest.raises(ValueError):
        ExtraTreesBaseModel(n_estimators=8).fit(X, np.zeros(len(X)))


@pytest.mark.parametrize("outcomes", [np.array([.1]), np.ones((3, 1)), np.array([.1, np.nan, .2]), np.array([.1, np.inf, .2])])
def test_invalid_outcome_vectors_are_rejected(outcomes):
    with pytest.raises(ValueError):
        ExtraTreesBaseModel(n_estimators=8).fit(np.ones((3, 2)), outcomes)


def test_unfitted_prediction_and_export_and_wrong_query_shapes_are_rejected():
    model = ExtraTreesBaseModel(n_estimators=8)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.raw_score([[0., 1.]])
    with pytest.raises(RuntimeError, match="not fitted"):
        model.export(["a", "b"])
    X, y, groups = _data()
    model.fit(X, y, groups)
    with pytest.raises(ValueError):
        model.export(["a", "b"])
    with pytest.raises(ValueError):
        model.raw_score(np.ones((2, 4)))
    with pytest.raises(ValueError):
        predict_from_export(model.export(["a", "b", "c"]), np.ones((2, 2)))



import numpy as np
import pytest

from docking_confidence.modeling.ridge import RidgeBaseModel
from docking_confidence.modeling.extra_trees import complex_weights


def test_ridge_predictions_match_independent_normal_equations_and_export():
    X = np.array([[2., 8., 7.], [5., -2., 7.], [1., 4., 7.], [9., 6., 7.], [-3., 2., 7.]])
    y = np.array([-4., 2., 3., 9., -2.])
    query = np.array([[4., 5., 7.], [30., -20., 7.], [-10., 1., 7.]])
    fitted = RidgeBaseModel().fit(X, y)
    saved = fitted.export(["first", "second", "constant"])

    # Independently compute the standardized Ridge normal equations.
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1.
    scaled = (X - mean) / scale
    coefficients = np.linalg.solve(scaled.T @ scaled + np.eye(X.shape[1]), scaled.T @ (y - y.mean()))
    expected = ((query - mean) / scale) @ coefficients + y.mean()
    np.testing.assert_allclose(fitted.raw_score(query), expected, rtol=1e-12, atol=1e-12)
    exported = ((query - saved["scaler_mean"]) / saved["scaler_scale"]) @ np.asarray(saved["coef"]) + saved["intercept"]
    original_units = query @ np.asarray(saved["original_scale_coef"]) + saved["original_scale_intercept"]
    np.testing.assert_allclose(exported, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(original_units, expected, rtol=1e-12, atol=1e-12)
    assert saved["method"] == "standardized_ridge_regression_v1"
    assert saved["alpha"] == 1.0
    assert "positive_class" not in saved and "class_weight" not in saved
    assert (expected > 1).any()  # Regression predictions are not clipped probabilities.


def test_scaling_is_fit_on_training_features_only_and_queries_do_not_mutate_it():
    training = np.array([[0., 10.], [2., 20.], [4., 30.]])
    fitted = RidgeBaseModel().fit(training, np.array([-.1, .2, .4]))
    before = fitted.export(["a", "b"])
    query = np.array([[1e6, -1e6]])
    score = fitted.raw_score(query)
    after = fitted.export(["a", "b"])
    assert before == after
    np.testing.assert_array_equal(before["scaler_mean"], training.mean(axis=0))
    np.testing.assert_array_equal(before["scaler_scale"], training.std(axis=0))
    expected = ((query - before["scaler_mean"]) / before["scaler_scale"]) @ np.asarray(before["coef"]) + before["intercept"]
    np.testing.assert_array_equal(score, expected)


@pytest.mark.parametrize("outcome", [-0.25, 0., 0.75])
def test_constant_outcomes_are_supported(outcome):
    model = RidgeBaseModel().fit(np.array([[1., 2.], [3., 1.], [-4., 7.]]), np.full(3, outcome))
    np.testing.assert_allclose(model.raw_score(np.array([[100., -20.], [0., 0.]])), outcome, atol=1e-12)


@pytest.mark.parametrize("alpha", [0., -1., np.nan, np.inf, True])
def test_invalid_regularization_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        RidgeBaseModel(alpha=alpha)


@pytest.mark.parametrize("X,y", [
    (np.array([1., 2.]), np.array([1., 2.])),
    (np.empty((0, 2)), np.array([])),
    (np.empty((2, 0)), np.array([1., 2.])),
    (np.array([[1.], [np.nan]]), np.array([1., 2.])),
    (np.array([[1.], [np.inf]]), np.array([1., 2.])),
    (np.array([[1.], [2.]]), np.array([1.])),
    (np.array([[1.], [2.]]), np.array([[1.], [2.]])),
    (np.array([[1.], [2.]]), np.array([1., np.nan])),
    (np.array([[1.], [2.]]), np.array([1., np.inf])),
])
def test_invalid_training_arrays_are_rejected(X, y):
    with pytest.raises(ValueError):
        RidgeBaseModel().fit(X, y)


@pytest.mark.parametrize("query", [np.array([1., 2.]), np.array([[np.nan]]), np.array([[np.inf]]), np.empty((0, 1)), np.array([[1., 2.]])])
def test_invalid_query_arrays_are_rejected(query):
    model = RidgeBaseModel().fit(np.array([[1.], [2.]]), np.array([.1, .2]))
    with pytest.raises(ValueError):
        model.raw_score(query)


def test_unfitted_model_and_mismatched_feature_names_are_rejected():
    model = RidgeBaseModel()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.raw_score(np.array([[1.]]))
    with pytest.raises(RuntimeError, match="not fitted"):
        model.export()
    model.fit(np.array([[1.], [2.]]), np.array([.1, .2]))
    with pytest.raises(ValueError, match="feature names"):
        model.export(["too", "many"])


def test_group_weighting_applies_to_both_scaler_and_ridge_normal_equations():
    X = np.array([[0., 2.], [1., 2.], [2., 2.], [10., 2.], [14., 2.]])
    y = np.array([-.3, -.2, -.1, .4, .7])
    groups = ["a", "a", "a", "b", "b"]
    weights = complex_weights(groups, len(X))
    fitted = RidgeBaseModel().fit(X, y, groups)
    saved = fitted.export(["variable", "constant"])
    mean = np.average(X, axis=0, weights=weights)
    scale = np.sqrt(np.average((X - mean) ** 2, axis=0, weights=weights))
    scale[scale == 0] = 1.
    np.testing.assert_allclose(saved["scaler_mean"], mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(saved["scaler_scale"], scale, rtol=1e-12, atol=1e-12)
    assert saved["weighting"] == "equal_total_weight_per_molecular_complex"
    scaled = (X - mean) / scale
    y_mean = np.average(y, weights=weights)
    coef = np.linalg.solve(scaled.T @ (weights[:, None] * scaled) + np.eye(X.shape[1]),
                           scaled.T @ (weights * (y - y_mean)))
    expected = scaled @ coef + y_mean
    np.testing.assert_allclose(fitted.raw_score(X), expected, rtol=1e-12, atol=1e-12)
    # A legacy ungrouped refit removes the molecular-weighting metadata.
    fitted.fit(X, y)
    assert "weighting" not in fitted.export(["variable", "constant"])


@pytest.mark.parametrize("groups", [["a"], ["a", None], ["a", ""], ["a", 3]])
def test_invalid_molecular_groups_are_rejected(groups):
    with pytest.raises(ValueError):
        RidgeBaseModel().fit(np.array([[0.], [1.]]), np.array([.1, .2]), groups)

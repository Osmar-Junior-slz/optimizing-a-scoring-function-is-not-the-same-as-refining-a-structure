"""Numerical checks against independent constrained/isotonic regression oracles."""
import json

import numpy as np
import pytest
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from docking_confidence.calibration.generalized_venn_abers import GeneralizedVennAbersCalibrator
from docking_confidence.calibration.venn_abers import VennAbersCalibrator


def test_continuous_sets_match_independent_augmented_isotonic_oracle():
    scores = np.array([-2, -1, -1, 0, 3, 4], dtype=float)
    outcomes = np.array([0.4, -0.2, 0.1, 0.7, -0.3, 0.4])
    queries = np.array([-3, -1, 0.5, 3, 5], dtype=float)
    candidates = np.array([-0.95, 0, 0.36, 0.77])
    fitted = GeneralizedVennAbersCalibrator((-0.95, 0.77)).fit(scores, outcomes)
    actual = fitted.predict_set(queries, candidates)
    for i, query in enumerate(queries):
        for j, outcome in enumerate(candidates):
            oracle = IsotonicRegression().fit(np.r_[scores, query], np.r_[outcomes, outcome])
            assert actual[i, j] == pytest.approx(oracle.predict([query])[0], abs=1e-14)
    lower, upper = fitted.predict_bounds(queries)
    np.testing.assert_allclose(lower, actual[:, 0], atol=1e-14)
    np.testing.assert_allclose(upper, actual[:, -1], atol=1e-14)
    assert np.all(actual >= lower[:, None] - 1e-14)
    assert np.all(actual <= upper[:, None] + 1e-14)


def test_generalized_query_matches_direct_constrained_quadratic_minimization():
    scores = np.array([0, 1, 1, 2, 4], dtype=float)
    outcomes = np.array([0.3, -0.7, 0.2, 0.1, 0.8])
    query, hypothetical = 1.5, -0.2
    augmented_scores, augmented_y = np.r_[scores, query], np.r_[outcomes, hypothetical]
    unique, inverse = np.unique(augmented_scores, return_inverse=True)
    solution = minimize(
        lambda theta: np.square(theta[inverse] - augmented_y).sum(),
        np.full(len(unique), augmented_y.mean()),
        jac=lambda theta: np.bincount(inverse, 2 * (theta[inverse] - augmented_y), minlength=len(unique)),
        constraints=[{"type": "ineq", "fun": lambda theta: np.diff(theta)}],
        method="SLSQP", options={"ftol": 1e-13, "maxiter": 1000},
    )
    assert solution.success, solution.message
    fitted = GeneralizedVennAbersCalibrator((-1, 1)).fit(scores, outcomes)
    actual = fitted.predict_set([query], [hypothetical])[0, 0]
    assert actual == pytest.approx(solution.x[np.where(unique == query)[0][0]], abs=1e-7)


@pytest.mark.parametrize("labels", [[0, 1, 0, 1, 0, 1], [1] * 6, [0] * 6])
def test_binary_reduction_matches_classic_ivap_without_production_dependency(labels):
    scores = np.array([-2, -1, -1, 0, 3, 4], dtype=float)
    queries = np.array([-3, -1, 0.5, 3, 5], dtype=float)
    generalized = GeneralizedVennAbersCalibrator((0, 1)).fit(scores, labels)
    classic = VennAbersCalibrator().fit(scores, labels, allow_single_class=True)
    for actual, expected in zip(generalized.predict_bounds(queries), classic.predict_interval(queries)[:2]):
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)


def test_constant_calibration_and_point_are_supported():
    scores, outcomes = np.array([-1, 0, 0, 2]), np.array([0.25] * 4)
    fitted = GeneralizedVennAbersCalibrator((-1, 1)).fit(scores, outcomes)
    queries = np.array([-2, 0, 1, 3])
    np.testing.assert_allclose(fitted.predict_point(queries), 0.25, atol=0)
    lower, upper = fitted.predict_bounds(queries)
    assert np.all(lower <= 0.25) and np.all(upper >= 0.25)
    assert np.any(lower < 0.25) and np.any(upper > 0.25)


def test_point_uses_only_fixed_calibration_fit_and_matches_sklearn():
    scores = np.array([0, 1, 1, 3], dtype=float)
    outcomes = np.array([0.2, 0.8, 0.4, 0.9])
    queries = np.array([-2, 0.5, 1, 2, 5])
    fitted = GeneralizedVennAbersCalibrator((0, 1)).fit(scores, outcomes)
    expected = IsotonicRegression(out_of_bounds="clip").fit(scores, outcomes).predict(queries)
    point = fitted.predict_point(queries)
    np.testing.assert_allclose(point, expected, atol=1e-14)
    before = fitted.export()
    fitted.predict_set(queries, [0, 0.3, 1])
    assert before == fitted.export()
    np.testing.assert_array_equal(point, fitted.predict_point(queries))


def test_permutation_ties_and_json_round_trip_preserve_results():
    scores = np.array([0, 0, 0, 1, 1, 2])
    outcomes = np.array([0.1, 0.2, 0.3, -0.3, 0.5, 0.6])
    queries, candidates = [-1, 0, 0.5, 3], [-0.95, 0.2, 0.77]
    fitted = GeneralizedVennAbersCalibrator((-0.95, 0.77)).fit(scores, outcomes)
    order = [5, 2, 4, 0, 3, 1]
    permuted = GeneralizedVennAbersCalibrator((-0.95, 0.77)).fit(scores[order], outcomes[order])
    restored = GeneralizedVennAbersCalibrator.from_export(json.loads(json.dumps(fitted.export())))
    assert len(fitted.score_groups) == 3
    for candidate in [permuted, restored]:
        assert candidate.export() == fitted.export()
        np.testing.assert_array_equal(candidate.predict_set(queries, candidates), fitted.predict_set(queries, candidates))
        np.testing.assert_array_equal(candidate.predict_point(queries), fitted.predict_point(queries))


def test_candidate_order_empty_queries_and_degenerate_known_support():
    fitted = GeneralizedVennAbersCalibrator((0.2, 0.2)).fit([0], [0.2])
    lower, upper = fitted.predict_bounds([0, 1])
    np.testing.assert_array_equal(lower, upper)
    assert fitted.predict_set([], [0.2]).shape == (0, 1)
    other = GeneralizedVennAbersCalibrator((0, 1)).fit([0, 1], [0, 1])
    ordered = other.predict_set([0.5], [0, 0.3, 1])
    np.testing.assert_array_equal(other.predict_set([0.5], [1, 0, 0.3]), ordered[:, [2, 0, 1]])


@pytest.mark.parametrize("bounds", [(1, 0), (0, np.inf), (True, 1), (0,), None])
def test_invalid_declared_support_is_rejected(bounds):
    with pytest.raises(ValueError):
        GeneralizedVennAbersCalibrator(bounds)


@pytest.mark.parametrize("scores,outcomes", [([], []), ([0], []), ([np.inf], [0]), ([0], [np.nan]), ([0], [1.1]), ([0], [-0.1])])
def test_bad_fit_inputs_are_rejected(scores, outcomes):
    with pytest.raises(ValueError):
        GeneralizedVennAbersCalibrator((0, 1)).fit(scores, outcomes)


def test_bad_queries_and_imputed_outcomes_are_rejected():
    fitted = GeneralizedVennAbersCalibrator((0, 1)).fit([0], [0.5])
    for scores, candidates in [([np.nan], [0]), ([0], []), ([0], [-1]), ([0], [np.inf])]:
        with pytest.raises(ValueError):
            fitted.predict_set(scores, candidates)
    with pytest.raises(RuntimeError, match="not fitted"):
        GeneralizedVennAbersCalibrator().predict_bounds([0])


@pytest.mark.parametrize("field,value", [
    ("method", "renamed_old_method"), ("loss", "log_loss"), ("reference", "2015"),
    ("calibration_example_count", True), ("calibration_example_count", 2.0),
    ("calibration_outcome_sum", "1.0"), ("calibration_outcome_sum", 5),
])
def test_bad_export_metadata_is_rejected(field, value):
    payload = GeneralizedVennAbersCalibrator((0, 1)).fit([0, 1], [0, 1]).export()
    payload[field] = value
    with pytest.raises(ValueError):
        GeneralizedVennAbersCalibrator.from_export(payload)


@pytest.mark.parametrize("field,value", [("weight", True), ("weight", 1.5), ("outcome_sum", np.inf), ("outcome_sum", 1.5), ("score_hex", "0x1p+0")])
def test_bad_export_groups_are_rejected(field, value):
    payload = GeneralizedVennAbersCalibrator((0, 1)).fit([0, 1], [0, 1]).export()
    payload["score_groups"][0][field] = value
    with pytest.raises(ValueError):
        GeneralizedVennAbersCalibrator.from_export(payload)

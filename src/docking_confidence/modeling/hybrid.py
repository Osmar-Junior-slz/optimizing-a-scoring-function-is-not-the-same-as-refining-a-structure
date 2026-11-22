"""One margin score from fixed, equally weighted linear and tree predictors.

Both components learn the same continuous outcome using proper-training rows.
The ensemble contains no calibration step: a single GVA calibrates its score.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Real

import numpy as np

from docking_confidence.modeling.extra_trees import (
    ExtraTreesBaseModel, PARAMETERS as TREE_PARAMETERS, _features, complex_weights,
    predict_from_export as predict_trees, validate_export as validate_trees,
)
from docking_confidence.modeling.ridge import RidgeBaseModel


METHOD = "auditable_ridge_extra_trees_margin_ensemble_v1"
PARAMETERS = {
    "ridge": {"alpha": 1.0, "fit_intercept": True, "solver": "svd"},
    "extra_trees": deepcopy(TREE_PARAMETERS),
    "weights": {"ridge": 0.5, "extra_trees": 0.5},
}
WEIGHTING = "equal_total_weight_per_molecular_complex"
COMBINATION = "weighted_mean_of_raw_margin_scores"


def _scalar(value, description):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{description} must be a finite number")
    return float(value)


def _vector(value, size, description):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{description} must match the hybrid feature count")
    return np.asarray([_scalar(v, description) for v in value], dtype=float)


def _validate_ridge(payload, features):
    if (not isinstance(payload, dict) or payload.get("method") != "standardized_ridge_regression_v1" or
        payload.get("feature_columns") != features or payload.get("weighting") != WEIGHTING):
        raise ValueError("linear component disagrees with the hybrid feature or weighting contract")
    if (_scalar(payload.get("alpha"), "ridge alpha") != 1.0 or
        payload.get("fit_intercept") is not True or payload.get("solver") != "svd"):
        raise ValueError("hybrid linear component uses fixed Ridge parameters")
    if payload.get("score_orientation") != "larger_means_larger_preservation_margin":
        raise ValueError("the linear score must predict the continuous preservation margin")
    count = len(features)
    mean = _vector(payload.get("scaler_mean"), count, "ridge scaler mean")
    scale = _vector(payload.get("scaler_scale"), count, "ridge scaler scale")
    coef = _vector(payload.get("coef"), count, "ridge coefficients")
    intercept = _scalar(payload.get("intercept"), "ridge intercept")
    if (scale <= 0).any():
        raise ValueError("ridge scaler scales must be positive")
    original_coef = _vector(payload.get("original_scale_coef"), count, "original ridge coefficients")
    original_intercept = _scalar(payload.get("original_scale_intercept"), "original ridge intercept")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        expected_coef = coef / scale
        expected_intercept = intercept - np.sum(expected_coef * mean)
    if (not np.isfinite(expected_coef).all() or not np.isfinite(expected_intercept) or
        not np.allclose(original_coef, expected_coef, rtol=1e-12, atol=1e-12) or
        not np.isclose(original_intercept, expected_intercept, rtol=1e-12, atol=1e-12)):
        raise ValueError("the two saved linear formulas disagree")
    return payload


def validate_export(payload):
    """Reject altered weights, components, units and malformed exported arrays."""
    if not isinstance(payload, dict) or payload.get("method") != METHOD:
        raise ValueError("unsupported hybrid model")
    features = payload.get("feature_columns")
    if (not isinstance(features, list) or not features or
        any(not isinstance(name, str) or not name or name != name.strip() for name in features) or
        len(set(features)) != len(features)):
        raise ValueError("hybrid feature names must be unique nonempty strings")
    parameters = payload.get("parameters")
    if parameters != PARAMETERS:
        raise ValueError("hybrid parameters and 50/50 weights are fixed")
    if (_scalar(parameters["ridge"]["alpha"], "ridge alpha") != 1.0 or
        parameters["ridge"]["fit_intercept"] is not True):
        raise ValueError("invalid fixed Ridge parameters")
    for weight in parameters["weights"].values():
        if _scalar(weight, "hybrid weight") != 0.5:
            raise ValueError("hybrid weights must both equal 0.5")
    if payload.get("weighting") != WEIGHTING or payload.get("combination") != COMBINATION:
        raise ValueError("unsupported hybrid fitting or combination contract")
    components = payload.get("components")
    if not isinstance(components, dict) or set(components) != {"ridge", "extra_trees"}:
        raise ValueError("hybrid requires exactly its linear and tree components")
    _validate_ridge(components["ridge"], features)
    trees = validate_trees(components["extra_trees"])
    if trees["feature_columns"] != features or trees["parameters"] != parameters["extra_trees"]:
        raise ValueError("tree component disagrees with the hybrid feature or parameter contract")
    return payload


def component_scores_from_export(payload, X):
    """Replay both frozen margin predictors using only their saved public arrays."""
    validate_export(payload)
    values = _features(X)
    if values.shape[1] != len(payload["feature_columns"]):
        raise ValueError("query features disagree with the hybrid model")
    ridge = payload["components"]["ridge"]
    with np.errstate(over="ignore", invalid="ignore"):
        linear = ((values - ridge["scaler_mean"]) / ridge["scaler_scale"]) @ np.asarray(ridge["coef"]) + ridge["intercept"]
    if not np.isfinite(linear).all():
        raise ValueError("the linear margin score is not finite")
    return {"ridge": linear, "extra_trees": predict_trees(payload["components"]["extra_trees"], values)}


def predict_from_export(payload, X):
    scores = component_scores_from_export(payload, X)
    return 0.5 * scores["ridge"] + 0.5 * scores["extra_trees"]


class HybridBaseModel:
    """Fit two fixed predictors of the same margin and expose one raw score."""

    def __init__(self):
        self.parameters = deepcopy(PARAMETERS)
        self.ridge = RidgeBaseModel(alpha=1.0)
        self.extra_trees = ExtraTreesBaseModel()

    def fit(self, X, y, groups=None):
        values = _features(X)
        outcomes = np.asarray(y, dtype=float)
        if outcomes.shape != (len(values),) or not np.isfinite(outcomes).all():
            raise ValueError("hybrid outcomes must be a finite vector aligned with features")
        # Explicit row groups make the weighting contract uniform even for the
        # optional ungrouped API: each row is then one equally weighted unit.
        if groups is None:
            groups = np.asarray([f"row-{index}" for index in range(len(values))])
        complex_weights(groups, len(values))
        self.ridge.fit(values, outcomes, groups=groups)
        self.extra_trees.fit(values, outcomes, groups=groups)
        return self

    def component_scores(self, X):
        values = _features(X)
        return {"ridge": self.ridge.raw_score(values), "extra_trees": self.extra_trees.raw_score(values)}

    def raw_score(self, X):
        scores = self.component_scores(X)
        return 0.5 * scores["ridge"] + 0.5 * scores["extra_trees"]

    def export(self, feature_columns):
        return validate_export({
            "method": METHOD, "parameters": deepcopy(self.parameters),
            "feature_columns": list(feature_columns), "weighting": WEIGHTING,
            "combination": COMBINATION, "components": {
                "ridge": self.ridge.export(feature_columns),
                "extra_trees": self.extra_trees.export(feature_columns),
            },
        })

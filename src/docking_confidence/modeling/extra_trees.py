"""Regularized nonlinear predictor with equal-complex weights and JSON replay."""
from __future__ import annotations

from numbers import Integral

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


METHOD = "auditable_extra_trees_regression_v1"
PARAMETERS = dict(n_estimators=128, max_depth=4, min_samples_leaf=5,
                  max_features=1.0, bootstrap=False, random_state=2026, n_jobs=1)


def _features(X):
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("tree features must be a nonempty finite matrix")
    with np.errstate(over="ignore"):
        single = values.astype(np.float32)
    if not np.isfinite(single).all():
        raise ValueError("tree features must be representable as float32")
    return values


def complex_weights(groups, n_rows):
    if isinstance(n_rows, bool) or not isinstance(n_rows, Integral) or n_rows < 1:
        raise ValueError("molecular weights require a positive example count")
    if groups is None:
        return np.ones(n_rows)
    raw = np.asarray(groups, dtype=object)
    if raw.shape != (n_rows,) or any(not isinstance(v, str) or not v.strip() for v in raw):
        raise ValueError("molecular groups must be nonempty and aligned with examples")
    values = raw.astype(str)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    # Every molecular complex has the same total fitting weight.
    return n_rows / len(counts) / counts[inverse]


class ExtraTreesBaseModel:
    def __init__(self, **parameters):
        self.parameters = {**PARAMETERS, **parameters}
        unknown = set(self.parameters) - set(PARAMETERS)
        if unknown:
            raise ValueError(f"unsupported tree parameters: {sorted(unknown)}")
        for name in ("n_estimators", "min_samples_leaf"):
            value = self.parameters[name]
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            self.parameters[name] = int(value)
        depth = self.parameters["max_depth"]
        if depth is not None and (isinstance(depth, bool) or not isinstance(depth, Integral) or depth < 1):
            raise ValueError("max_depth must be None or a positive integer")
        self.parameters["max_depth"] = None if depth is None else int(depth)
        seed = self.parameters["random_state"]
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("random_state must be a non-negative integer")
        self.parameters["random_state"] = int(seed)
        if not isinstance(self.parameters["max_features"], (float, np.floating)) or self.parameters["max_features"] != 1.0:
            raise ValueError("this auditable model uses all available features")
        if self.parameters["bootstrap"] is not False or type(self.parameters["n_jobs"]) is not int or self.parameters["n_jobs"] != 1:
            raise ValueError("auditable tree fitting requires no bootstrap and one worker")
        self.model = ExtraTreesRegressor(**self.parameters)

    def fit(self, X, y, groups=None):
        values = _features(X)
        outcomes = np.asarray(y, dtype=float)
        if outcomes.shape != (len(values),) or not np.isfinite(outcomes).all():
            raise ValueError("tree outcomes must be a finite vector aligned with features")
        self.model.fit(values, outcomes, sample_weight=complex_weights(groups, len(values)))
        return self

    def raw_score(self, X):
        if not hasattr(self.model, "estimators_"):
            raise RuntimeError("tree model is not fitted")
        return self.model.predict(_features(X))

    def export(self, feature_columns):
        if not hasattr(self.model, "estimators_"):
            raise RuntimeError("tree model is not fitted")
        if len(feature_columns) != self.model.n_features_in_:
            raise ValueError("feature names must match tree feature count")
        trees = []
        for estimator in self.model.estimators_:
            tree = estimator.tree_
            trees.append({"children_left": tree.children_left.tolist(),
                "children_right": tree.children_right.tolist(), "feature": tree.feature.tolist(),
                "threshold": tree.threshold.tolist(), "value": tree.value[:, 0, 0].tolist()})
        payload = {"method": METHOD, "parameters": self.parameters, "feature_columns": list(feature_columns),
            "weighting": "equal_total_weight_per_molecular_complex", "trees": trees,
            "input_dtype": "float32", "aggregation": "arithmetic_mean_of_trees",
            "feature_importances": self.model.feature_importances_.tolist()}
        return validate_export(payload)


def validate_export(payload):
    if not isinstance(payload, dict) or payload.get("method") != METHOD:
        raise ValueError("unsupported tree model")
    if payload.get("input_dtype") != "float32" or payload.get("aggregation") != "arithmetic_mean_of_trees":
        raise ValueError("unsupported tree prediction contract")
    names = payload.get("feature_columns")
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names) or
        len(set(names)) != len(names)):
        raise ValueError("tree feature names must be unique nonempty strings")
    params = payload.get("parameters")
    if not isinstance(params, dict) or set(params) != set(PARAMETERS):
        raise ValueError("invalid tree parameters")
    ExtraTreesBaseModel(**params)
    trees = payload.get("trees")
    if not isinstance(trees, list) or len(trees) != params["n_estimators"]:
        raise ValueError("tree count disagrees with parameters")
    if payload.get("weighting") != "equal_total_weight_per_molecular_complex":
        raise ValueError("unsupported molecular weighting contract")
    importance = np.asarray(payload.get("feature_importances"), dtype=float)
    if (importance.shape != (len(names),) or not np.isfinite(importance).all() or (importance < 0).any() or
        not (np.isclose(importance.sum(), 1.0) or importance.sum() == 0)):
        raise ValueError("tree feature importances must be finite, nonnegative and normalized")
    for tree in trees:
        if not isinstance(tree, dict) or set(tree) != {"children_left", "children_right", "feature", "threshold", "value"}:
            raise ValueError("invalid serialized tree")
        if not isinstance(tree["value"], list):
            raise ValueError("tree values must be a list")
        n = len(tree["value"])
        if n < 1 or any(not isinstance(tree[k], list) or len(tree[k]) != n for k in tree):
            raise ValueError("tree arrays must be nonempty and aligned")
        for key in ("children_left", "children_right", "feature"):
            if any(isinstance(v, bool) or not isinstance(v, Integral) for v in tree[key]):
                raise ValueError("tree indices must be integers")
        for key in ("threshold", "value"):
            if not np.isfinite(np.asarray(tree[key], dtype=float)).all():
                raise ValueError("tree values must be finite")
        parents = np.zeros(n, dtype=int)
        for node, (left, right, feature) in enumerate(zip(tree["children_left"], tree["children_right"], tree["feature"])):
            if left == right == -1:
                if feature != -2:
                    raise ValueError("tree leaf has an invalid feature")
                continue
            if not (node < left < n and node < right < n and left != right and 0 <= feature < len(names)):
                raise ValueError("tree children or split feature are invalid")
            parents[left] += 1
            parents[right] += 1
        if parents[0] or np.any(parents[1:] != 1):
            raise ValueError("tree must be a connected acyclic rooted tree")
    return payload


def predict_from_export(payload, X):
    payload = validate_export(payload)
    values = _features(X).astype(np.float32)
    if values.shape[1] != len(payload["feature_columns"]):
        raise ValueError("query features disagree with the exported trees")
    result = np.zeros(len(values))
    rows = np.arange(len(values))
    for tree in payload["trees"]:
        left = np.asarray(tree["children_left"])
        right = np.asarray(tree["children_right"])
        feature = np.asarray(tree["feature"])
        threshold = np.asarray(tree["threshold"])
        nodes = np.zeros(len(values), dtype=int)
        active = left[nodes] != -1
        while active.any():
            r = rows[active]
            current = nodes[active]
            go_left = values[r, feature[current]] <= threshold[current]
            nodes[active] = np.where(go_left, left[current], right[current])
            active = left[nodes] != -1
        result += np.asarray(tree["value"])[nodes]
    return result / len(payload["trees"])


def _group_mean(values, groups):
    unique = np.unique(groups)
    return float(np.mean([np.mean(values[groups == g]) for g in unique]))



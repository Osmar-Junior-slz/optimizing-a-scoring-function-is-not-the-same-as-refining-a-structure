"""Validate the fixed Ridge/tree hybrid on disjoint molecular training groups."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from sklearn.model_selection import GroupKFold

from docking_confidence.modeling.extra_trees import _features, _group_mean, complex_weights
from docking_confidence.modeling.hybrid import HybridBaseModel, METHOD, PARAMETERS


def group_cross_validation(X, y, groups):
    """Evaluate fixed inputs and parameters; never select features or calibrate.

    Both components are fitted afresh on each fold's training rows. Their raw
    margin scores are combined with fixed equal weights before evaluation;
    no GVA calibrator is fitted here. The caller supplies proper-training rows
    only. Entire molecular complexes
    stay together, and both fitting and reported losses give each complex the
    same total weight. Calibration and test rows are not arguments to this API.
    """
    X = _features(X)
    if groups is None:
        raise ValueError("group validation requires molecular groups")
    complex_weights(groups, len(X))
    y, groups = np.asarray(y, dtype=float), np.asarray(groups, dtype=str)
    if y.shape != (len(X),) or not np.isfinite(y).all():
        raise ValueError("validation outcomes must be finite and aligned")
    count = len(np.unique(groups))
    report = {
        "available": count >= 2,
        "scope": "fixed_model_group_cross_validation_inside_proper_training_only",
        "base_model_method": METHOD,
        "hyperparameter_search": False, "feature_selection": False,
        "calibration_or_test_used": False,
        "weighting": "each_molecular_complex_has_equal_weight",
        "independent_complex_count": count, "row_count": len(y),
        "feature_count": X.shape[1], "parameters": deepcopy(PARAMETERS), "folds": [],
    }
    if count < 2:
        report["reason"] = "fewer_than_two_proper_training_complexes"
        return None, report
    prediction, baseline = np.empty(len(y)), np.empty(len(y))
    splits = GroupKFold(n_splits=min(5, count)).split(X, y, groups)
    for index, (train, validation) in enumerate(splits):
        model = HybridBaseModel().fit(X[train], y[train], groups[train])
        values = np.asarray(model.raw_score(X[validation]), dtype=float)
        if values.shape != (len(validation),) or not np.isfinite(values).all():
            raise ValueError("validation predictions must be finite and aligned")
        prediction[validation] = values
        baseline[validation] = _group_mean(y[train], groups[train])
        report["folds"].append({
            "fold": index, "training_groups": sorted(set(groups[train])),
            "validation_groups": sorted(set(groups[validation])),
            "validation_rows": len(validation),
        })
    error = prediction - y
    mse = _group_mean(error ** 2, groups)
    baseline_mse = _group_mean((baseline - y) ** 2, groups)
    report.update(
        group_mae=_group_mean(np.abs(error), groups), group_mse=mse,
        training_mean_baseline_group_mse=baseline_mse,
        skill_against_fold_training_mean=1 - mse / baseline_mse if baseline_mse > 0 else None,
        predicted_negative_rows=int((prediction < 0).sum()),
        observed_negative_rows=int((y < 0).sum()),
    )
    return prediction, report

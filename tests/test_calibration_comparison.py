import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from docking_confidence.application.calibration_comparison import (
    ComparisonOptions, comparison_partitions, fit_comparison_fold,
    select_validation_threshold,
)
from docking_confidence.domain.preservation import PreservationRule


@pytest.fixture
def study():
    rng = np.random.default_rng(17)
    groups = np.repeat([f"{i:064x}" for i in range(50)], 3)
    X = np.c_[rng.integers(0, 5, len(groups)) / 5 + .1,
              np.tile([.1, .5, .9], 50)]
    y = (X[:, 0] >= .5).astype(int)
    roles = {"model_train": np.arange(72), "probability_calibration": np.arange(72, 108),
             "threshold_validation": np.arange(108, 129), "outer_test": np.arange(129, 150)}
    kwargs = dict(columns=["viable_fraction_mean", "elite_dispersion"], roles=roles,
                  reduced_quality=.2 + .6 * y, full_quality=.4 + .3 * y)
    return X, y, groups, kwargs


def test_outer_outcomes_and_full_quality_cannot_change_models_thresholds_or_predictions(study):
    X, y, groups, kwargs = study
    first = fit_comparison_fold(X, y, groups, **kwargs)
    yy = y.copy(); yy[kwargs["roles"]["outer_test"]] = 1 - yy[kwargs["roles"]["outer_test"]]
    changed = dict(kwargs)
    changed["full_quality"] = kwargs["full_quality"].copy()
    changed["full_quality"][kwargs["roles"]["outer_test"]] = 999
    second = fit_comparison_fold(X, yy, groups, **changed)
    assert first["model"] == second["model"]
    assert first["calibrators"] == second["calibrators"]
    assert first["policies"] == second["policies"]
    for method in first["scores"]:
        np.testing.assert_array_equal(first["scores"][method], second["scores"][method])
        np.testing.assert_array_equal(first["accepted"][method], second["accepted"][method])
    np.testing.assert_array_equal(first["cascade"]["coef"], second["cascade"]["coef"])


def test_validation_changes_decision_only_and_isotonic_matches_independent_fit(study):
    X, y, groups, kwargs = study
    first = fit_comparison_fold(X, y, groups, **kwargs)
    yy = y.copy(); yy[kwargs["roles"]["threshold_validation"]] = 1
    second = fit_comparison_fold(X, yy, groups, **kwargs)
    assert first["model"] == second["model"]
    assert first["calibrators"] == second["calibrators"]
    assert second["policies"]["uncalibrated"]["threshold"] is None
    for method in first["probabilities"]:
        np.testing.assert_array_equal(first["probabilities"][method], second["probabilities"][method])
    model = first["model"]
    raw = ((X - model["scaler_mean"]) / model["scaler_scale"]) @ np.asarray(model["coef"]) + model["intercept"]
    domain = first["domain"]
    indices = kwargs["roles"]["probability_calibration"]
    used = indices[((X[indices] >= domain["lower"]) & (X[indices] <= domain["upper"])).all(axis=1)]
    independent = IsotonicRegression(out_of_bounds="clip").fit(raw[used], y[used])
    np.testing.assert_allclose(first["probabilities"]["isotonic"], independent.predict(raw[kwargs["roles"]["outer_test"]]))


def test_one_class_calibration_is_marked_and_never_creates_venn_abers_probability(study):
    X, y, groups, kwargs = study
    y[kwargs["roles"]["probability_calibration"]] = 1
    result = fit_comparison_fold(X, y, groups, **kwargs)
    assert np.isnan(result["probabilities"]["venn_abers"]).all()
    assert result["calibrators"]["isotonic"]["class_deficient"] is True
    assert not result["accepted"]["venn_abers"].any()
    assert result["policies"]["venn_abers"]["reason"] == "method_unavailable"


def test_four_roles_keep_aliases_and_every_group_is_tested_once(study):
    _, _, groups, _ = study
    seen = []
    for partition in comparison_partitions(groups):
        sets = [set(values) for values in partition["groups"].values()]
        assert sum(map(len, sets)) == len(np.unique(groups))
        assert not any(sets[i] & sets[j] for i in range(4) for j in range(i + 1, 4))
        seen += partition["groups"]["outer_test"]
    assert sorted(seen) == sorted(np.unique(groups))


def test_cross_role_alias_is_rejected(study):
    X, y, groups, kwargs = study
    groups[129] = groups[0]
    with pytest.raises(ValueError, match="crosses"):
        fit_comparison_fold(X, y, groups, **kwargs)


def test_planned_groups_without_valid_features_keep_their_outer_assignment(study):
    _, _, groups, _ = study
    missing = "f" * 64
    partitions = comparison_partitions(groups, planned_groups=[missing])
    assert sum(missing in p["groups"]["outer_test"] for p in partitions) == 1
    rows = np.concatenate([p["indices"]["outer_test"] for p in partitions])
    np.testing.assert_array_equal(np.sort(rows), np.arange(len(groups)))


def test_five_complex_followup_is_only_a_specification_with_sixty_unique_jobs(tmp_path):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1] / "scripts/run_calibration_audit.py"
    spec = importlib.util.spec_from_file_location("audit_driver_for_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    representatives = {f"{i:064x}": {"complex_id": f"case-{i}"} for i in range(9)}
    module.prepare_five_complex_followup({"representatives": representatives}, tmp_path)
    import json
    result = json.loads((tmp_path / "followup_5_spec.json").read_text())
    assert result["planned_jobs"] == 60 and len(result["targets"]) == 5
    assert "not_executed" in result["scope"]
    keys = {(j["complex_id"], j["algorithm"], j["extent"], j["effective_evaluation_budget"]) for j in result["jobs"]}
    assert len(keys) == 60
    assert all(j["effective_evaluation_budget"] == 7600 for j in result["jobs"] if j["extent"] == "expanded")


def test_validation_tuning_does_not_accept_a_high_score_negative_without_accounting_risk():
    options = ComparisonOptions(minimum_validation_accepted_groups=1)
    result = select_validation_threshold([.99, .95, .90], [0, 1, 1], [True] * 3,
                                         ["a", "b", "c"], options=options)
    assert result["threshold"] is None
    assert all(row["empirical_risk"] > .1 for row in result["candidates"] if row["accepted"])


@pytest.mark.parametrize("reduced,full,label", [(.25, .375, 1), (.25, .5, 0), (.125, .125, 0), (.5, .25, 1)])
def test_preservation_direction_quality_floor_and_inclusive_loss(reduced, full, label):
    assert PreservationRule(.25, .125).label(reduced, full) == label

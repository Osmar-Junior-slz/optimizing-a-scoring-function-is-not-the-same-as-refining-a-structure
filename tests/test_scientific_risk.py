"""Statistical oracle checks, clustering and missing-data safety; no invented docking results."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta, binom

from docking_confidence.risk.policy import clopper_pearson_upper
from docking_confidence.risk.scientific import (
    METHODS, load_risk_protocol, risk_coverage_envelope, structural_outcomes,
    validate_risk_protocol, zero_error_sample_requirements,
)


def _frame(n=10):
    return pd.DataFrame({"complex_fingerprint_sha256": [f"complex-{i}" for i in range(n)],
        "dockq_reduced": [.6] * n, "dockq_full": [.65] * n})


def _profile(frame, *, config=None, universe=None, eligible=None):
    return risk_coverage_envelope(frame, {method: np.full(len(frame), .95) for method in METHODS},
        np.ones(len(frame), dtype=bool) if eligible is None else eligible,
        config or load_risk_protocol(), stage="risk_calibration", all_complexes=universe)


@pytest.mark.parametrize("reduced,full,safe,unknown", [
    (.22, .1, False, False), (.23, .23, True, False), (.23, .49, False, False),
    (.49, .23, True, False), (.80, .79, True, False), (.79, .81, False, False),
    (.5, .79, True, False), (np.nan, .5, False, True), (.1, np.nan, False, False),
    (.6, np.nan, False, True),
])
def test_published_quality_cutoffs_and_unknowns(reduced, full, safe, unknown):
    outcome = structural_outcomes(pd.DataFrame({"dockq_reduced": [reduced], "dockq_full": [full]})).iloc[0]
    assert bool(outcome.structural_quality_preserved) is safe
    assert bool(outcome.quality_unknown) is unknown


@pytest.mark.parametrize("errors,trials", [(0, 1), (0, 10), (1, 10), (4, 30), (9, 10), (10, 10)])
def test_exact_bounds_match_scipy_beta_oracle(errors, trials):
    tail = .05 / 600
    oracle = 1. if errors == trials else beta.ppf(1-tail, errors+1, trials-errors)
    assert clopper_pearson_upper(errors, trials, tail) == pytest.approx(oracle, abs=1e-12)


def test_correlated_algorithms_are_not_independent_binomial_trials():
    frame = _frame()
    frame.loc[0, "dockq_reduced"] = .1
    single = _profile(frame)["curves"]["venn_abers"][0]
    triple = _profile(pd.concat([frame] * 3, ignore_index=True))["curves"]["venn_abers"][0]
    assert triple["accepted_runs"] == 30 and triple["false_accepted_runs_observed"] == 3
    assert triple["accepted_complexes"] == single["accepted_complexes"] == 10
    assert triple["worst_case_adverse_complexes"] == 1
    assert triple["simultaneous_complex_risk_upper_bound"] == single["simultaneous_complex_risk_upper_bound"]


def test_family_includes_all_methods_thresholds_and_both_stages():
    config = load_risk_protocol()
    profile = _profile(_frame(), config=config)
    assert profile["family_size"] == 600
    assert profile["error_probability_per_bound"] == pytest.approx(.05 / 600)
    assert profile["selected_threshold"] is None and not profile["decision_authorized"]
    assert profile["curves"]["original"][0]["simultaneous_complex_risk_upper_bound"] > .60
    assert profile["curves"]["original"][-1]["simultaneous_complex_risk_upper_bound"] is None
    assert profile["curves"]["original"][-1]["empirical_worst_case_complex_risk"] is None


def test_unknown_quality_is_not_dropped_or_counted_as_success():
    frame = _frame(3)
    frame.loc[0, "dockq_full"] = np.nan
    row = _profile(frame, universe=["complex-0", "complex-1", "complex-2", "failed-reduced"])["curves"]["venn_abers"][0]
    assert row["total_complexes"] == 4 and row["accepted_complexes"] == 3
    assert row["empirical_complex_coverage"] == .75
    assert row["observed_adverse_complexes"] == 0
    assert row["unknown_accepted_complexes"] == row["worst_case_adverse_complexes"] == 1


def test_all_reduced_failures_have_zero_coverage_and_unavailable_risk():
    row = _profile(_frame(0), universe=["failed-complex"])["curves"]["venn_abers"][0]
    assert row["empirical_complex_coverage"] == 0
    assert row["empirical_worst_case_complex_risk"] is None
    assert row["simultaneous_complex_risk_upper_bound"] is None


def test_zero_error_planning_matches_exact_test_not_a_fixed_sample_floor():
    cfg = load_risk_protocol()
    tail = .05 / 600
    for row in zero_error_sample_requirements(cfg):
        n = row["minimum_accepted_independent_complexes_if_zero_errors"]
        alpha = row["illustrative_risk_level"]
        assert clopper_pearson_upper(0, n, tail) <= alpha
        assert clopper_pearson_upper(0, n-1, tail) > alpha
        assert not row["approved_tolerance"]


def test_exact_bound_has_nominal_binomial_coverage_by_enumeration():
    # Enumerate the binomial sampling distribution, not a noisy simulation.
    for n in (3, 12):
        for truth in (.15, .5, .9):
            tail = .05 / 6
            false_coverage = sum(binom.pmf(k, n, truth) for k in range(n+1)
                                 if clopper_pearson_upper(k, n, tail) < truth)
            assert false_coverage <= tail + 1e-12


def test_config_cannot_silently_approve_a_risk_tolerance():
    cfg = deepcopy(load_risk_protocol())
    cfg["maximum_accepted_error_rate"] = .20
    with pytest.raises(ValueError, match="not been approved"):
        validate_risk_protocol(cfg)
    cfg = deepcopy(load_risk_protocol())
    cfg["quality_cutoffs"][0] = .1
    with pytest.raises(ValueError, match="cited DockQ"):
        validate_risk_protocol(cfg)


def test_missing_or_invalid_grouping_and_scores_are_rejected():
    frame = _frame()
    frame.loc[0, "complex_fingerprint_sha256"] = None
    with pytest.raises(ValueError, match="grouping"):
        _profile(frame)
    with pytest.raises(ValueError, match="binary"):
        _profile(_frame(), eligible=[np.nan] * 10)

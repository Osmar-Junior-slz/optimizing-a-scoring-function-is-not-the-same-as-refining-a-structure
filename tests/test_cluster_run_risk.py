import copy

import numpy as np
import pandas as pd
import pytest

from docking_confidence.risk.policy import (
    assess_cluster_counts,
    assess_threshold,
    audit_locked_policy,
    clopper_pearson_lower,
    select_policy,
)


def _config(**overrides):
    return {
        "method": "cluster_bootstrap_run_risk",
        "risk_target": 0.20,
        "familywise_confidence_level": 0.95,
        "risk_error_budget_fraction": 0.80,
        "threshold_min": 0.90,
        "threshold_max": 0.90,
        "threshold_step": 0.05,
        "minimum_complex_coverage": 0.10,
        "minimum_evaluated_complexes": 30,
        "minimum_authorized_complexes": 30,
        "bootstrap_repetitions": 2_000,
        "bootstrap_seed": 20_260_909,
        **overrides,
    }


def _record(index, accepted, false, *, total=None, eligible=None):
    return {
        "complex_fingerprint_sha256": f"{index:064x}",
        "total_run_count": max(accepted, 1) if total is None else total,
        "eligible_run_count": accepted if eligible is None else eligible,
        "accepted_run_count": accepted,
        "false_accepted_run_count": false,
    }


def _assess(records, **config_overrides):
    return assess_cluster_counts(
        records, 0.90, cfg=_config(**config_overrides), hypothesis_count=1, role="test"
    )


def _supported_records():
    return [_record(index, 3, int(index < 6)) for index in range(60)]


def test_primary_risk_counts_runs_and_records_keep_unaccepted_complexes():
    records = [
        _record(0, 3, 1, total=4, eligible=4),
        _record(1, 1, 0, total=3, eligible=2),
        _record(2, 0, 0, total=3, eligible=1),
    ]
    result = _assess(records)
    assert result["empirical_run_risk"] == 1 / 4
    assert result["empirical_conditional_complex_risk"] == 1 / 2
    assert result["total_row_count"] == 10
    assert result["eligible_row_count"] == 7
    assert result["accepted_row_count"] == 4
    assert result["false_acceptance_row_count"] == 1
    assert result["total_complex_count"] == 3
    assert result["accepted_complex_count"] == 2
    assert result["largest_accepted_cluster_share"] == 3 / 4
    assert result["cluster_counts"] == records
    assert result["risk_bound_method"] == "cluster_bootstrap_percentile"
    assert result["confidence_guarantee"] == "asymptotic_not_finite_sample"
    assert "risk_p_value" not in result
    assert "risk_p_value_bonferroni" not in result


def test_pooled_run_risk_is_not_mean_of_cluster_error_rates():
    result = _assess([_record(0, 1, 1), _record(1, 4, 1)])
    assert result["empirical_run_risk"] == 2 / 5
    assert result["empirical_run_risk"] != (1 + 1 / 4) / 2


def test_deterministic_replay_matches_whole_cluster_bootstrap_reference():
    records = [
        _record(0, 3, 1),
        _record(1, 1, 0),
        _record(2, 0, 0),
        _record(3, 2, 0),
    ]
    result = _assess(records)
    config = _config()
    rng = np.random.Generator(np.random.PCG64(config["bootstrap_seed"]))
    indices = rng.integers(0, 4, size=(config["bootstrap_repetitions"], 4))
    # Independent scalar oracle: each sampled index contributes its complete
    # accepted/false-accepted run counts, including repeated draws and zeros.
    risks = []
    undefined = 0
    for sampled in indices:
        accepted = sum(records[index]["accepted_run_count"] for index in sampled)
        false = sum(records[index]["false_accepted_run_count"] for index in sampled)
        risks.append(false / accepted if accepted else 1.0)
        undefined += int(accepted == 0)
    expected = np.quantile(risks, 0.96, method="higher")
    assert result["simultaneous_risk_upper_bound"] == expected
    assert result["bootstrap_zero_denominator_repetitions"] == undefined
    assert undefined > 0
    assert result["bootstrap_completed_repetitions"] == len(risks)
    assert _assess(list(reversed(records))) == result


def test_duplicating_correlated_runs_does_not_create_independent_evidence():
    records = _supported_records()
    repeated = copy.deepcopy(records)
    for record in repeated:
        for key in (
            "total_run_count", "eligible_run_count", "accepted_run_count",
            "false_accepted_run_count",
        ):
            record[key] *= 10
    first, second = _assess(records), _assess(repeated)
    for field in (
        "empirical_run_risk", "simultaneous_risk_upper_bound", "total_complex_count",
        "accepted_complex_count", "simultaneous_complex_coverage_lower_bound", "supported",
    ):
        assert first[field] == second[field]
    assert second["accepted_row_count"] == 10 * first["accepted_row_count"]


@pytest.mark.parametrize(
    "records,status,zero_denominators",
    [
        ([_record(index, 0, 0) for index in range(40)], "no_accepted_runs", 2_000),
        ([_record(index, 3, 0) for index in range(40)], "no_observed_failures", 0),
        ([_record(index, 2, 1) for index in range(40)], "degenerate_cluster_risk", 0),
    ],
)
def test_no_information_or_degenerate_bootstrap_fails_closed(records, status, zero_denominators):
    result = _assess(records, risk_target=1.0)
    assert result["bootstrap_status"] == status
    assert result["simultaneous_risk_upper_bound"] == 1.0
    assert result["supported"] is False
    assert result["bootstrap_zero_denominator_repetitions"] == zero_denominators


def test_coverage_remains_exact_at_independent_complex_level():
    result = _assess(_supported_records())
    assert result["simultaneous_complex_coverage_lower_bound"] == clopper_pearson_lower(
        60, 60, result["coverage_error_probability_per_hypothesis"]
    )
    assert result["supported"] is True
    for override in (
        {"minimum_evaluated_complexes": 61},
        {"minimum_authorized_complexes": 61},
        {"minimum_complex_coverage": 1.0},
        {"risk_target": 0.0},
    ):
        assert _assess(_supported_records(), **override)["supported"] is False


def test_multiple_thresholds_reduce_error_budget_and_do_not_loosen_upper_bound():
    records = _supported_records()
    single = _assess(records)
    multiple = assess_cluster_counts(
        records, 0.90, cfg=_config(), hypothesis_count=5, role="test"
    )
    assert multiple["risk_error_probability_per_hypothesis"] == pytest.approx(
        single["risk_error_probability_per_hypothesis"] / 5
    )
    assert multiple["simultaneous_risk_upper_bound"] >= single["simultaneous_risk_upper_bound"]
    assert multiple["simultaneous_complex_coverage_lower_bound"] <= single[
        "simultaneous_complex_coverage_lower_bound"
    ]


def test_insufficient_monte_carlo_tail_resolution_fails_closed():
    result = assess_cluster_counts(
        _supported_records(), 0.90, cfg=_config(risk_target=1.0),
        hypothesis_count=10, role="test",
    )
    assert result["bootstrap_status"] == "insufficient_bootstrap_tail_resolution"
    assert result["bootstrap_expected_tail_repetitions"] < 20
    assert result["simultaneous_risk_upper_bound"] == 1.0
    assert result["supported"] is False


def test_threshold_assessment_matches_counts_replay_and_respects_eligibility():
    frame = pd.DataFrame(
        {
            "complex_fingerprint_sha256": [f"{index:064x}" for index in (0, 0, 0, 1, 1, 2)],
            "preserved": [0, 1, 0, 1, 0, 0],
        }
    )
    probabilities = np.array([0.90, 0.99, 0.99, 0.90, 0.89, 0.89])
    eligibility = np.array([True, True, False, True, True, True])
    result = assess_threshold(
        frame, probabilities, eligibility, 0.90,
        cfg=_config(), hypothesis_count=1, role="test",
    )
    expected = [
        _record(0, 2, 1, total=3, eligible=2),
        _record(1, 1, 0, total=2, eligible=2),
        _record(2, 0, 0, total=1, eligible=1),
    ]
    assert result == _assess(expected)
    shuffled = np.array([5, 2, 3, 0, 4, 1])
    replay = assess_threshold(
        frame.iloc[shuffled], probabilities[shuffled], eligibility[shuffled], 0.90,
        cfg=_config(), hypothesis_count=1, role="test",
    )
    assert replay == result


def test_policy_selection_uses_new_method_and_unsupported_threshold_is_none():
    rows = []
    for index in range(60):
        for run in range(3):
            rows.append({
                "complex_fingerprint_sha256": f"{index:064x}",
                "preserved": int(not (index < 6 and run == 0)),
            })
    frame = pd.DataFrame(rows)
    probabilities = np.full(len(frame), 0.90)
    eligible = np.ones(len(frame), dtype=bool)
    selected, candidates = select_policy(frame, probabilities, eligible, _config())
    assert selected["supported"] is True
    assert selected["threshold"] == 0.90
    assert selected["method"] == "cluster_bootstrap_run_risk_v1"
    assert "run-weighted" in selected["loss_definition"]
    assert selected["confidence_guarantee"] == "asymptotic_not_finite_sample"
    assert "risk_p_value" not in selected
    assert len(candidates) == 1
    audited = audit_locked_policy(frame, probabilities, eligible, 0.90, _config())
    assert audited["role"] == "locked_test"
    assert audited["empirical_run_risk"] == selected["empirical_run_risk"]

    frame["preserved"] = 1
    unsupported, _ = select_policy(frame, probabilities, eligible, _config())
    assert unsupported["supported"] is False
    assert unsupported["threshold"] is None
    assert unsupported["best_observed_candidate"]["bootstrap_status"] == "no_observed_failures"


@pytest.mark.parametrize(
    "field,value",
    [
        ("complex_fingerprint_sha256", "display-name"),
        ("complex_fingerprint_sha256", None),
        ("complex_fingerprint_sha256", "A" * 64),
        ("total_run_count", 0),
        ("total_run_count", 2**63),
        ("eligible_run_count", 4),
        ("accepted_run_count", 4),
        ("false_accepted_run_count", 4),
        ("accepted_run_count", 1.0),
        ("false_accepted_run_count", True),
        ("false_accepted_run_count", -1),
        ("false_accepted_run_count", None),
    ],
)
def test_counts_replay_rejects_malformed_or_inconsistent_records(field, value):
    record = _record(0, 3, 1)
    record[field] = value
    with pytest.raises(ValueError):
        _assess([record])


@pytest.mark.parametrize("records", [[], [None], [_record(0, 1, 0)] * 2])
def test_counts_replay_rejects_empty_invalid_or_duplicate_clusters(records):
    with pytest.raises(ValueError):
        _assess(records)


def test_new_assessment_rejects_missing_fingerprint_instead_of_dropping_it():
    frame = pd.DataFrame({"complex_fingerprint_sha256": [None], "preserved": [1]})
    with pytest.raises(ValueError, match="complex_fingerprint_sha256"):
        assess_threshold(
            frame, np.array([0.95]), np.array([True]), 0.90,
            cfg=_config(), hypothesis_count=1, role="test",
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"bootstrap_repetitions": True},
        {"bootstrap_repetitions": 10},
        {"bootstrap_repetitions": 1_000_001},
        {"bootstrap_seed": -1},
        {"familywise_confidence_level": 1.0},
        {"risk_error_budget_fraction": 0.0},
        {"minimum_evaluated_complexes": 0},
        {"minimum_authorized_complexes": 1.0},
    ],
)
def test_counts_replay_rejects_invalid_bootstrap_or_evidence_config(overrides):
    with pytest.raises(ValueError):
        _assess(_supported_records(), **overrides)

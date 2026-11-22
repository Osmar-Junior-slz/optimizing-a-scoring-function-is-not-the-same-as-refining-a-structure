import numpy as np
import pandas as pd
import pytest

from docking_confidence.risk.policy import (
    assess_threshold,
    binomial_cdf,
    clopper_pearson_lower,
    clopper_pearson_upper,
    operational_acceptance_mask,
    select_policy,
    threshold_grid,
)


def _policy_config() -> dict:
    return {
        "risk_target": 0.5,
        "familywise_confidence_level": 0.8,
        "risk_error_budget_fraction": 0.5,
        "threshold_min": 0.0,
        "threshold_max": 0.8,
        "threshold_step": 0.4,
        "minimum_complex_coverage": 0.0,
        "minimum_evaluated_complexes": 1,
        "minimum_authorized_complexes": 1,
    }


def test_exact_binomial_endpoints_match_closed_form_edges():
    assert binomial_cdf(0, 2, 0.5) == pytest.approx(0.25)
    assert clopper_pearson_upper(0, 10, 0.05) == pytest.approx(
        1 - 0.05 ** (1 / 10)
    )
    assert clopper_pearson_lower(10, 10, 0.05) == pytest.approx(
        0.05 ** (1 / 10)
    )


def test_complex_level_evidence_is_unchanged_by_duplicate_runs():
    frame = pd.DataFrame(
        {
            "complex_fingerprint_sha256": ["A", "B", "C"],
            "preserved": [1, 0, 1],
        }
    )
    probabilities = np.array([0.9, 0.8, 0.2])
    eligible = np.ones(3, dtype=bool)
    first = assess_threshold(
        frame,
        probabilities,
        eligible,
        0.5,
        cfg=_policy_config(),
        hypothesis_count=3,
        role="test",
    )

    duplicated = pd.concat([frame, frame.iloc[[0, 1]]], ignore_index=True)
    duplicated_probabilities = np.array([0.9, 0.8, 0.2, 0.9, 0.8])
    second = assess_threshold(
        duplicated,
        duplicated_probabilities,
        np.ones(5, dtype=bool),
        0.5,
        cfg=_policy_config(),
        hypothesis_count=3,
        role="test",
    )
    for field in (
        "total_complex_count",
        "accepted_complex_count",
        "adverse_accepted_complex_count",
        "empirical_conditional_complex_risk",
        "simultaneous_risk_upper_bound",
        "complex_coverage",
        "simultaneous_complex_coverage_lower_bound",
        "supported",
    ):
        assert second[field] == pytest.approx(first[field])


def test_threshold_grid_is_predeclared_from_config():
    np.testing.assert_array_equal(threshold_grid(_policy_config()), [0.0, 0.4, 0.8])


def test_fixed_threshold_includes_boundary_and_preserves_eligibility():
    accepted = operational_acceptance_mask(
        np.array([np.nextafter(0.9, 0.0), 0.9, 0.99, 0.9, 0.99]),
        np.array([True, True, True, False, False]),
        0.9,
    )
    np.testing.assert_array_equal(accepted, [False, True, True, False, False])


@pytest.mark.parametrize("preserved,supported", [(1, True), (0, False)])
def test_fixed_threshold_still_requires_statistical_support(preserved, supported):
    config = _policy_config()
    config.update(threshold_min=0.9, threshold_max=0.9)
    frame = pd.DataFrame(
        {
            "complex_fingerprint_sha256": [f"C{index}" for index in range(10)],
            "preserved": [preserved] * 10,
        }
    )

    selected, candidates = select_policy(
        frame, np.full(10, 0.9), np.ones(10, dtype=bool), config
    )

    assert selected["candidate_grid"] == [0.9]
    assert selected["tested_hypothesis_count"] == 1
    assert len(candidates) == 1
    assert candidates[0]["threshold"] == 0.9
    assert candidates[0]["accepted_complex_count"] == 10
    assert selected["supported"] is supported
    assert selected["threshold"] == (0.9 if supported else None)
    if not supported:
        assert selected["best_observed_candidate"] == candidates[0]


def test_risk_gate_uses_p_value_and_keeps_cp_upper_bound_for_audit(monkeypatch):
    import docking_confidence.risk.policy as policy_module

    frame = pd.DataFrame(
        {
            "complex_fingerprint_sha256": [f"C{index}" for index in range(10)],
            "preserved": [1] * 10,
        }
    )
    monkeypatch.setattr(policy_module, "clopper_pearson_upper", lambda *args: 1.0)
    result = assess_threshold(
        frame,
        np.ones(10),
        np.ones(10, dtype=bool),
        0.5,
        cfg=_policy_config(),
        hypothesis_count=1,
        role="test",
    )
    assert result["risk_p_value"] <= result["risk_error_probability_per_hypothesis"]
    assert result["simultaneous_risk_upper_bound"] > result["risk_target"]
    assert result["supported"] is True
    assert "minimum_row_coverage" not in result


def test_policy_selection_uses_predeclared_lexicographic_order(monkeypatch):
    import docking_confidence.risk.policy as policy_module

    assessments = {
        0.0: (0.30, 0.20),
        0.4: (0.30, 0.10),
        0.8: (0.30, 0.10),
    }

    def fake_assessment(df, probability_lower, eligible, threshold, **kwargs):
        coverage_lower, risk_upper = assessments[threshold]
        return {
            "supported": True,
            "threshold": threshold,
            "simultaneous_complex_coverage_lower_bound": coverage_lower,
            "simultaneous_risk_upper_bound": risk_upper,
        }

    monkeypatch.setattr(policy_module, "assess_threshold", fake_assessment)
    frame = pd.DataFrame(
        {"complex_fingerprint_sha256": ["A"], "preserved": [1]}
    )
    selected, _ = select_policy(
        frame,
        np.array([1.0]),
        np.array([True]),
        _policy_config(),
    )
    assert selected["threshold"] == 0.8
    assert "i.i.d." in selected["statistical_assumptions"]
    assert "protocol is fixed" in selected["statistical_assumptions"]


def test_policy_rejects_missing_complex_fingerprint():
    frame = pd.DataFrame({"complex_id": ["display-only"], "preserved": [1]})
    with pytest.raises(ValueError, match="complex_fingerprint_sha256"):
        assess_threshold(
            frame,
            np.array([1.0]),
            np.array([True]),
            0.5,
            cfg=_policy_config(),
            hypothesis_count=1,
            role="test",
        )

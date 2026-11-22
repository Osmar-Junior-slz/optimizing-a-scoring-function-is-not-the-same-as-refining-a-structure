import json

import numpy as np
import pandas as pd
import pytest

from docking_confidence.application.probability_reporting import (
    DiagnosticOptions, _cluster_draws, _resampled_metrics, descriptor_comparison, frozen_prediction_report,
)
from docking_confidence.application.runtime import operational_cost_assessment
from docking_confidence.validation.metrics import (
    probability_curves, probability_metrics, reliability_bins, sigmoid_raw_score,
)
from docking_confidence.validation.split import four_way_group_partition


def test_pr_auc_and_ap_are_distinct_and_prevalence_is_explicit():
    result = probability_metrics(np.array([0, 1, 1]), np.array([0.8, 0.5, 0.2]))
    assert result["roc_auc"] == 0
    assert result["pr_auc_trapezoidal"] == pytest.approx(5 / 12)
    assert result["average_precision"] == pytest.approx(7 / 12)
    assert result["positive_prevalence"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("labels,reason", [([], "no_rows"), ([1, 1], "single_class"), ([0, 0], "single_class")])
def test_missing_class_or_rows_does_not_invent_discrimination(labels, reason):
    result = probability_metrics(np.array(labels), np.full(len(labels), 0.4))
    for metric in ("roc_auc", "pr_auc_trapezoidal", "average_precision"):
        assert result[metric] is None
    assert result["ranking_unavailable_reason"] == reason
    curves = probability_curves(np.array(labels), np.full(len(labels), 0.4))
    assert curves["roc"] == curves["precision_recall"] == []
    if not labels:
        assert result["brier_score"] is result["log_loss"] is None
    else:
        assert result["brier_score"] is not None


def test_reliability_edges_and_sigmoid_logits():
    bins = reliability_bins(np.array([0, 1, 1]), np.array([0, 0.1, 1.0]))
    assert bins[0]["count"] == bins[1]["count"] == bins[-1]["count"] == 1
    assert bins[2]["mean_probability"] is None
    assert sum(row["count"] for row in bins) == 3
    np.testing.assert_allclose(sigmoid_raw_score(np.array([-1000, 0, 1000])), [0, 0.5, 1])
    with pytest.raises(ValueError, match="probabilities"):
        probability_metrics(np.array([0, 1]), np.array([-2, 3]))


def test_bootstrap_scalar_implementation_matches_reference_metrics_with_score_ties():
    rng = np.random.default_rng(8)
    for _ in range(20):
        labels = rng.integers(0, 2, 20)
        probability = rng.choice([0, 0.1, 0.3, 0.5, 0.9, 1], 20)
        expected = probability_metrics(labels, probability)
        actual = _resampled_metrics(labels, probability)
        for metric, value in actual.items():
            assert value == pytest.approx(expected[metric], abs=1e-12)


def _saved_frame():
    return pd.DataFrame({
        "complex_fingerprint_sha256": np.repeat(list("abcdef"), 2),
        "preserved": np.tile([0, 1], 6), "base_raw_score": np.tile([-0.2, 0.2], 6),
        "probability": np.tile([0.2, 0.8], 6),
    })


def test_before_after_uses_sigmoid_and_group_bootstrap_replays_after_reordering():
    options = DiagnosticOptions(100, 42, 0.95)
    frame = _saved_frame()
    report = frozen_prediction_report(frame, options=options)
    assert report == frozen_prediction_report(frame.sample(frac=1, random_state=8), options=options)
    before = report["evaluations"]["before_venn_abers"]["metrics"]["brier_score"]
    assert before == pytest.approx(float(np.mean((frame.preserved - sigmoid_raw_score(frame.base_raw_score)) ** 2)))
    assert report["paired_after_minus_before"]["brier_score"]["estimate"] < 0
    assert report["evaluations"]["after_venn_abers"]["confidence_intervals"]["roc_auc"]["valid_repetitions"] == 100
    json.dumps(report, allow_nan=False)
    groups = np.array(["a", "b", "b", "c", "c", "c"])
    for indices in _cluster_draws(groups, options):
        for group, size in (("a", 1), ("b", 2), ("c", 3)):
            assert np.count_nonzero(groups[indices] == group) % size == 0


def test_one_complex_cannot_supply_a_complex_confidence_interval():
    frame = _saved_frame().iloc[:2]
    result = frozen_prediction_report(frame, options=DiagnosticOptions(100))
    interval = result["evaluations"]["after_venn_abers"]["confidence_intervals"]["brier_score"]
    assert interval["lower"] is None
    assert interval["unavailable_reason"] == "fewer_than_two_complexes"


def test_before_calibration_ranking_keeps_raw_logit_order_when_sigmoid_saturates():
    frame = _saved_frame().iloc[:2].copy()
    frame["base_raw_score"] = [40, 41]
    assert len(np.unique(sigmoid_raw_score(frame.base_raw_score))) == 1
    result = frozen_prediction_report(frame, options=DiagnosticOptions(100))
    assert result["evaluations"]["before_venn_abers"]["metrics"]["roc_auc"] == 1


def test_operational_cost_includes_full_fallback_and_preserves_scope():
    frame = pd.DataFrame({"runtime_reduced_sec": [2, 3], "runtime_full_sec": [10, 20]})
    result = operational_cost_assessment(frame, np.array([True, False]))
    assert result["paired_strategy_sec"] == 25
    assert result["baseline_full_sec"] == 30
    assert result["saved_sec"] == 5
    assert result["fallback_row_count"] == 1
    assert result["overhead_included"] is False
    assert operational_cost_assessment(frame, np.array([False, False]))["saved_sec"] == -5
    assert operational_cost_assessment(frame, np.array([True, True]), overhead_sec=2)["paired_strategy_sec"] == 7
    with pytest.raises(ValueError, match="nonnegative"):
        operational_cost_assessment(frame.assign(runtime_reduced_sec=[-1, 3]), np.array([True, False]))


def test_descriptor_variants_do_not_fit_on_locked_test_outcomes():
    frame = pd.DataFrame({"complex_fingerprint_sha256": np.repeat([f"g{i:02}" for i in range(20)], 2),
        "preserved": np.tile([0, 1], 20), "elite_dispersion": np.tile([0.2, 0.8], 20),
        "graph_interface_density": np.tile([0.1, 0.9], 20)})
    split = four_way_group_partition(frame.preserved.to_numpy(), frame.complex_fingerprint_sha256.to_numpy(), split_folds=5, random_seed=99)
    config = {"model": {"C": 1, "max_iter": 1000, "class_weight": "balanced"},
              "applicability_domain": {"lower_quantile": 0, "upper_quantile": 1}}
    saved = frame.iloc[split.locked_test].assign(base_raw_score=lambda f: f.preserved * 2 - 1,
                                             probability=lambda f: f.preserved * 0.6 + 0.2)
    options = DiagnosticOptions(100)
    before = descriptor_comparison(frame, split, ["elite_dispersion", "graph_interface_density"], config, split.probability_calibration, saved, options=options)
    changed = frame.copy()
    changed.loc[split.locked_test, "preserved"] = 1 - changed.loc[split.locked_test, "preserved"]
    changed.loc[split.locked_test, "elite_dispersion"] += 100
    changed.loc[split.locked_test, "graph_interface_density"] += 200
    after = descriptor_comparison(changed, split, ["elite_dispersion", "graph_interface_density"], config, split.probability_calibration, saved, options=options)
    for family in ("population", "interface"):
        for component in ("model", "calibration", "applicability_domain"):
            assert before["variants"][family][component] == after["variants"][family][component]
    assert before["variants"]["population"]["diagnostics"] != after["variants"]["population"]["diagnostics"]

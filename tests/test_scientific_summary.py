"""Unit tests only for the independent scientific calculations."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "scientific_summary", Path(__file__).resolve().parents[1] / "scripts" / "scientific_summary.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def case(score, action="ACCEPT_REDUCED", state="SUCCEEDED", algorithm="AG"):
    return {"dockq_reduced": score, "action": action, "reduced_state": state, "algorithm": algorithm}


def test_metrics_denominators_and_failures():
    rows = [case(.23), case(.1), case(.9), case(None),
            case(.8, "RUN_FULL"), case(None, "RUN_FULL", "FAILED")]
    result = summary.calculate_metrics(rows)
    assert result == {
        "total_decisions": 6, "accepted_reduced": 4, "coverage_percent": 100 * 4 / 6,
        "failed_reduced": 1, "accepted_dockq_mean": pytest.approx(1.23 / 3),
        "accepted_dockq_median": .23, "accepted_dockq_ge_0_23": 2,
        "accepted_dockq_ge_0_23_percent": 50, "accepted_dockq_lt_0_23": 1,
        "accepted_dockq_missing": 1}


def test_empty_and_no_accepted():
    empty = summary.calculate_metrics([])
    assert empty["coverage_percent"] is None
    assert empty["total_decisions"] == 0
    result = summary.calculate_metrics([case(.7, "RUN_FULL")])
    assert result["coverage_percent"] == 0
    for key in ("accepted_dockq_mean", "accepted_dockq_median", "accepted_dockq_ge_0_23_percent"):
        assert result[key] is None


def test_even_median_zero_score_and_algorithm_groups():
    result = summary.consolidate([case(0), case(.8), case(.3, algorithm="PSO"),
                                  case(None, "RUN_FULL", "FAILED", "Hybrid")])
    assert result["by_algorithm"]["AG"]["accepted_dockq_median"] == .4
    assert result["by_algorithm"]["PSO"]["accepted_reduced"] == 1
    assert result["by_algorithm"]["Hybrid"]["failed_reduced"] == 1
    assert result["overall"]["total_decisions"] == 4


@pytest.mark.parametrize("reduced,full,y", [(.23, .28, 1), (.3, .35, 1),
    (.23, .28000001, 0), (.229999, .2, 0), (.5, .3, 1), (0, 0, 0)])
def test_preservation_boundaries(reduced, full, y):
    result = summary.preservation(reduced, full)
    assert result["Y"] == y
    assert result["dockq_full_minus_reduced"] == pytest.approx(full - reduced)


@pytest.mark.parametrize("reduced,full", [(None, .5), (.5, None), (None, None), (.1, None)])
def test_missing_scores_do_not_invent_preservation(reduced, full):
    result = summary.preservation(reduced, full)
    assert result["Y"] is None
    assert result["dockq_full_minus_reduced"] is None
    assert "não pode ser determinada" in result["preservation_status"]
    assert result["dockq_full"] == full


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.01, 1.01, True, "0.5"])
def test_invalid_dockq_is_not_silently_counted(value):
    with pytest.raises(ValueError):
        summary.calculate_metrics([case(value)])

"""The cost axis and score direction must match the archived generation events."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from revision_convergence import align_curve, checkpoints


def events(pairs):
    return [json.dumps({"name": name, "value": value, "generation": generation}).encode()
            for generation, (count, score) in enumerate(pairs)
            for name, value in (("best_score", score), ("n_eval", count))]


def test_generation_regression_and_irregular_costs_use_only_completed_checkpoints():
    points = checkpoints(events([(100, -20), (78, -3), (30, -9), (0, -2)]), budget=208)
    assert align_curve(points, [100, 150, 178, 200, 208]) == [-20, -20, -3, -3, -2]
    with pytest.raises(ValueError, match="beyond observed"):
        align_curve(points, [0, 100])


def test_truncated_or_duplicated_metrics_do_not_make_plausible_curves():
    complete = events([(100, 1), (75, 2)])
    with pytest.raises(ValueError, match="Incomplete generation"):
        checkpoints(complete[:-1], 175)
    with pytest.raises(ValueError, match="duplicated"):
        checkpoints(complete + [complete[-1]], 175)
    with pytest.raises(ValueError, match="expected 200"):
        checkpoints(complete, 200)

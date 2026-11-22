"""Exact squared-loss generalized Venn–Abers, Algorithm 2 (ICML 2025).

For every candidate outcome, isotonic regression is refit after augmenting the
calibration data with the query and that outcome. The resulting set contains an
oracle marginally calibrated prediction under the paper's assumptions. Its range
is not a prediction interval for the observed response or a confidence interval
for a conditional mean. Binary outcomes are a special case of this algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np


REFERENCE = "https://arxiv.org/abs/2502.05676v3"


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return 0.0 if number == 0.0 else number


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _vector(values, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float).reshape(-1).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite numbers")
    result[result == 0.0] = 0.0
    return result


def _finite_sum(values) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise ValueError("outcome sums exceed finite floating-point range") from exc
    if not math.isfinite(result):
        raise ValueError("outcome sums must remain finite")
    return result


@dataclass(frozen=True)
class _Block:
    start: int
    stop: int
    weight: int
    outcome_sum: float

    @property
    def mean(self) -> float:
        return self.outcome_sum / self.weight


def _pav(groups: list[dict]) -> np.ndarray:
    """Weighted least-squares isotonic fit on ordered, tie-aggregated scores."""
    blocks: list[_Block] = []
    for index, group in enumerate(groups):
        blocks.append(_Block(index, index + 1, group["weight"], group["outcome_sum"]))
        while len(blocks) >= 2 and blocks[-2].mean >= blocks[-1].mean:
            left, right = blocks[-2:]
            blocks[-2:] = [_Block(
                left.start, right.stop, left.weight + right.weight,
                _finite_sum([left.outcome_sum, right.outcome_sum]),
            )]
    fitted = np.empty(len(groups), dtype=float)
    for block in blocks:
        fitted[block.start:block.stop] = block.mean
    return fitted


class GeneralizedVennAbersCalibrator:
    """Generalized mean calibration on a declared finite outcome support.

    ``predict_bounds`` computes the exact range of Algorithm 2 under squared
    loss: the isotonic fitted value at the query is nondecreasing in its imputed
    outcome, so support endpoints attain both extrema. ``predict_point`` is a
    separate, fixed ordinary isotonic calibration; no oracle guarantee is claimed
    for that point and no midpoint/minimax rule is imposed on general responses.
    """

    method = "generalized_venn_abers_squared_v1"
    loss = "squared_error"
    reference = REFERENCE

    def __init__(self, outcome_bounds=(0.0, 1.0)):
        if not isinstance(outcome_bounds, (list, tuple)) or len(outcome_bounds) != 2:
            raise ValueError("outcome_bounds must contain two finite endpoints")
        low, high = [_finite_number(v, "outcome bound") for v in outcome_bounds]
        if low > high:
            raise ValueError("outcome_bounds must be ordered")
        self.outcome_bounds = (low, high)
        self.score_groups: list[dict] | None = None
        self._point_values: np.ndarray | None = None

    def _validate_outcomes(self, outcomes, name="outcomes") -> np.ndarray:
        values = _vector(outcomes, name)
        low, high = self.outcome_bounds
        if np.any(values < low) or np.any(values > high):
            raise ValueError(f"{name} must lie inside declared outcome_bounds")
        return values

    def fit(self, scores, outcomes) -> "GeneralizedVennAbersCalibrator":
        score_values = _vector(scores, "calibration scores")
        responses = self._validate_outcomes(outcomes)
        if not len(score_values) or len(score_values) != len(responses):
            raise ValueError("calibration scores and outcomes must be nonempty and aligned")
        tied: dict[float, list[float]] = {}
        for score, outcome in zip(score_values, responses):
            tied.setdefault(float(score), []).append(float(outcome))
        self.score_groups = [
            {"score": score, "score_hex": score.hex(), "weight": len(values),
             "outcome_sum": _finite_sum(values)}
            for score, values in sorted(tied.items())
        ]
        self._point_values = _pav(self.score_groups)
        return self

    def _require_fitted(self) -> list[dict]:
        if self.score_groups is None:
            raise RuntimeError("generalized Venn–Abers calibrator is not fitted")
        return self.score_groups

    def predict_set(self, scores, candidate_outcomes) -> np.ndarray:
        """Return shape (n_queries, n_candidates), preserving candidate order."""
        groups = self._require_fitted()
        queries = _vector(scores, "query scores")
        candidates = self._validate_outcomes(candidate_outcomes, "candidate outcomes")
        if not len(candidates):
            raise ValueError("candidate outcomes must not be empty")
        result = np.empty((len(queries), len(candidates)), dtype=float)
        for row, query in enumerate(queries):
            tied = {g["score"]: (g["weight"], g["outcome_sum"]) for g in groups}
            weight, total = tied.get(float(query), (0, 0.0))
            ordered_scores = sorted(set(tied) | {float(query)})
            position = ordered_scores.index(float(query))
            for column, outcome in enumerate(candidates):
                augmented = []
                for score in ordered_scores:
                    if score == query:
                        count, value = weight + 1, _finite_sum([total, float(outcome)])
                    else:
                        count, value = tied[score]
                    augmented.append({"weight": count, "outcome_sum": value})
                result[row, column] = _pav(augmented)[position]
        return result

    def predict_bounds(self, scores) -> tuple[np.ndarray, np.ndarray]:
        endpoints = self.predict_set(scores, self.outcome_bounds)
        lower, upper = endpoints[:, 0], endpoints[:, 1]
        if np.any(lower > upper):
            raise RuntimeError("generalized isotonic endpoint monotonicity was violated")
        return lower, upper

    def predict_point(self, scores) -> np.ndarray:
        """Ordinary isotonic point estimate, fitted only on calibration data."""
        groups = self._require_fitted()
        queries = _vector(scores, "query scores")
        return np.interp(queries, [g["score"] for g in groups], self._point_values)

    def export(self) -> dict:
        groups = self._require_fitted()
        return {
            "method": self.method, "reference": self.reference, "loss": self.loss,
            "outcome_bounds": list(self.outcome_bounds),
            "point_rule": "ordinary_calibration_only_isotonic_linear_interpolation",
            "bounds_rule": "augmented_isotonic_at_declared_support_endpoints",
            "tie_policy": "exact_score_aggregation",
            "score_groups": [dict(group) for group in groups],
            "calibration_example_count": sum(g["weight"] for g in groups),
            "calibration_outcome_sum": _finite_sum(g["outcome_sum"] for g in groups),
        }

    @classmethod
    def from_export(cls, payload: dict) -> "GeneralizedVennAbersCalibrator":
        if not isinstance(payload, dict):
            raise ValueError("generalized Venn–Abers export must be an object")
        expected = {
            "method": cls.method, "reference": cls.reference, "loss": cls.loss,
            "point_rule": "ordinary_calibration_only_isotonic_linear_interpolation",
            "bounds_rule": "augmented_isotonic_at_declared_support_endpoints",
            "tie_policy": "exact_score_aggregation",
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"unsupported generalized Venn–Abers {key}")
        instance = cls(payload.get("outcome_bounds"))
        exported_groups = payload.get("score_groups")
        if not isinstance(exported_groups, list) or not exported_groups:
            raise ValueError("score_groups must be a nonempty list")
        checked = []
        previous = None
        low, high = instance.outcome_bounds
        for group in exported_groups:
            if not isinstance(group, dict):
                raise ValueError("score groups must be objects")
            score = _finite_number(group.get("score"), "score")
            weight = _integer(group.get("weight"), "weight", minimum=1)
            total = _finite_number(group.get("outcome_sum"), "outcome_sum")
            if previous is not None and score <= previous:
                raise ValueError("scores must be strictly ordered and ties aggregated")
            try:
                hexadecimal = float.fromhex(group["score_hex"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("score_hex must encode its score") from exc
            if hexadecimal != score:
                raise ValueError("score_hex disagrees with score")
            mean = total / weight
            # A rounded block sum can put its mean one ulp beyond the support.
            if ((mean < low and not math.isclose(mean, low, rel_tol=8e-15, abs_tol=0)) or
                (mean > high and not math.isclose(mean, high, rel_tol=8e-15, abs_tol=0))):
                raise ValueError("score-group outcomes exceed declared outcome_bounds")
            checked.append({"score": score, "score_hex": score.hex(), "weight": weight, "outcome_sum": total})
            previous = score
        count = _integer(payload.get("calibration_example_count"), "calibration_example_count", minimum=1)
        total = _finite_number(payload.get("calibration_outcome_sum"), "calibration_outcome_sum")
        if count != sum(g["weight"] for g in checked):
            raise ValueError("calibration_example_count disagrees with score groups")
        if total != _finite_sum(g["outcome_sum"] for g in checked):
            raise ValueError("calibration_outcome_sum disagrees with score groups")
        instance.score_groups = checked
        instance._point_values = _pav(checked)
        return instance

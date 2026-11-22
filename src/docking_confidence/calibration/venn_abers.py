from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _artifact_integer(value: object, *, name: str) -> int:
    """Read an artifact count without accepting booleans or numeric coercion."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"artifact {name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class RationalProbability:
    positive_count: int
    weight: int

    @property
    def value(self) -> float:
        return self.positive_count / self.weight


@dataclass
class _Block:
    positive_count: int
    weight: int
    contains_query: bool


def _validate_scores(scores: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(scores, dtype=float).reshape(-1).copy()
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    values[values == 0.0] = 0.0  # Canonicalize negative zero.
    return values


def _validate_labels(labels: np.ndarray, expected_size: int) -> np.ndarray:
    values = np.asarray(labels).reshape(-1)
    if len(values) != expected_size:
        raise ValueError("scores and labels must have the same length")
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Venn-Abers labels must be exactly 0 or 1")
    return values.astype(int)


def _aggregate_scores(scores: np.ndarray, labels: np.ndarray) -> list[dict]:
    grouped: dict[float, list[int]] = {}
    for score, label in zip(scores, labels):
        item = grouped.setdefault(float(score), [0, 0])
        item[0] += 1
        item[1] += int(label)
    return [
        {
            "score": score,
            "score_hex": score.hex(),
            "weight": counts[0],
            "positive_count": counts[1],
        }
        for score, counts in sorted(grouped.items())
    ]


def _pav_endpoint(
    score_groups: list[dict],
    query_score: float,
    hypothetical_label: int,
) -> RationalProbability:
    nodes = {
        float(group["score"]): [
            int(group["weight"]),
            int(group["positive_count"]),
            False,
        ]
        for group in score_groups
    }
    query_node = nodes.setdefault(float(query_score), [0, 0, False])
    query_node[0] += 1
    query_node[1] += hypothetical_label
    query_node[2] = True

    blocks: list[_Block] = []
    for _, (weight, positive_count, contains_query) in sorted(nodes.items()):
        blocks.append(
            _Block(
                positive_count=int(positive_count),
                weight=int(weight),
                contains_query=bool(contains_query),
            )
        )
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            # Exact cross-products avoid floating-point decisions in the PAV.
            if left.positive_count * right.weight < right.positive_count * left.weight:
                break
            blocks[-2:] = [
                _Block(
                    positive_count=left.positive_count + right.positive_count,
                    weight=left.weight + right.weight,
                    contains_query=left.contains_query or right.contains_query,
                )
            ]

    query_blocks = [block for block in blocks if block.contains_query]
    if len(query_blocks) != 1:
        raise RuntimeError("PAV did not retain exactly one query block")
    block = query_blocks[0]
    return RationalProbability(block.positive_count, block.weight)


class VennAbersCalibrator:
    """Auditable inductive Venn-Abers calibrator with a manual weighted PAV.

    The scoring model must already be frozen. This class stores only calibration
    score groups and, for every query, performs the two hypothetical-label fits.
    """

    method = "inductive_venn_abers_pav_v1"
    point_rule = "log_loss_minimax"
    tie_policy = "exact_score_aggregation"

    def __init__(self) -> None:
        self.score_groups: list[dict] | None = None

    def fit(
        self, raw_scores: np.ndarray, y: np.ndarray, *, allow_single_class: bool = False,
    ) -> "VennAbersCalibrator":
        scores = _validate_scores(raw_scores, name="calibration scores")
        if len(scores) == 0:
            raise ValueError("Venn-Abers calibration set cannot be empty")
        labels = _validate_labels(y, len(scores))
        if not allow_single_class and len(np.unique(labels)) < 2:
            raise ValueError("Venn-Abers calibration requires both binary classes")
        self.score_groups = _aggregate_scores(scores, labels)
        return self

    @classmethod
    def from_export(cls, payload: dict, *, allow_single_class: bool = False) -> "VennAbersCalibrator":
        if payload.get("method") != cls.method:
            raise ValueError(f"unsupported calibration method: {payload.get('method')!r}")
        if payload.get("point_rule") != cls.point_rule:
            raise ValueError("unsupported Venn-Abers point rule")
        if payload.get("tie_policy") != cls.tie_policy:
            raise ValueError("unsupported Venn-Abers tie policy")
        groups = payload.get("score_groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError("calibration score_groups must be a non-empty list")
        instance = cls()
        checked: list[dict] = []
        previous: float | None = None
        for group in groups:
            score = float(group["score"])
            weight = _artifact_integer(group["weight"], name="score-group weight")
            positive_count = _artifact_integer(
                group["positive_count"], name="score-group positive_count"
            )
            if not np.isfinite(score):
                raise ValueError("artifact calibration scores must be finite")
            if weight < 1 or not 0 <= positive_count <= weight:
                raise ValueError("invalid calibration score-group counts")
            if previous is not None and score <= previous:
                raise ValueError("artifact calibration score groups must be strictly ordered")
            if "score_hex" in group and float.fromhex(str(group["score_hex"])) != score:
                raise ValueError("calibration score and score_hex disagree")
            checked.append(
                {
                    "score": score,
                    "score_hex": score.hex(),
                    "weight": weight,
                    "positive_count": positive_count,
                }
            )
            previous = score
        total_weight = sum(group["weight"] for group in checked)
        total_positive = sum(group["positive_count"] for group in checked)
        if not allow_single_class and total_positive in {0, total_weight}:
            raise ValueError("artifact Venn-Abers calibration must contain both classes")
        calibration_example_count = _artifact_integer(
            payload.get("calibration_example_count"),
            name="calibration_example_count",
        )
        calibration_positive_count = _artifact_integer(
            payload.get("calibration_positive_count"),
            name="calibration_positive_count",
        )
        if calibration_example_count != total_weight:
            raise ValueError("artifact Venn-Abers calibration example count disagrees")
        if calibration_positive_count != total_positive:
            raise ValueError("artifact Venn-Abers positive count disagrees")
        instance.score_groups = checked
        return instance

    def _require_fitted(self) -> list[dict]:
        if self.score_groups is None:
            raise RuntimeError("Venn-Abers calibrator is not fitted")
        return self.score_groups

    def predict_interval(
        self, raw_scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        trace = self.predict_with_trace(raw_scores)
        return trace["lower"], trace["upper"], trace["point"]

    def predict_with_trace(self, raw_scores: np.ndarray) -> dict[str, np.ndarray]:
        groups = self._require_fitted()
        queries = _validate_scores(raw_scores, name="query scores")
        lower = np.empty(len(queries), dtype=float)
        upper = np.empty(len(queries), dtype=float)
        point = np.empty(len(queries), dtype=float)
        lower_positive = np.empty(len(queries), dtype=np.int64)
        lower_weight = np.empty(len(queries), dtype=np.int64)
        upper_positive = np.empty(len(queries), dtype=np.int64)
        upper_weight = np.empty(len(queries), dtype=np.int64)
        point_numerator = np.empty(len(queries), dtype=np.int64)
        point_denominator = np.empty(len(queries), dtype=np.int64)

        for index, query in enumerate(queries):
            p0 = _pav_endpoint(groups, float(query), 0)
            p1 = _pav_endpoint(groups, float(query), 1)
            if p0.positive_count * p1.weight >= p1.positive_count * p0.weight:
                raise RuntimeError("Venn-Abers invariant p0 < p1 was violated")
            lower[index] = p0.value
            upper[index] = p1.value
            # Equivalent to p1 / (1 - p0 + p1), evaluated as integer counts.
            numerator = p1.positive_count * p0.weight
            denominator = (
                (p0.weight - p0.positive_count) * p1.weight
                + p1.positive_count * p0.weight
            )
            point[index] = numerator / denominator
            lower_positive[index] = p0.positive_count
            lower_weight[index] = p0.weight
            upper_positive[index] = p1.positive_count
            upper_weight[index] = p1.weight
            point_numerator[index] = numerator
            point_denominator[index] = denominator

        return {
            "lower": lower,
            "upper": upper,
            "point": point,
            "lower_positive_count": lower_positive,
            "lower_weight": lower_weight,
            "upper_positive_count": upper_positive,
            "upper_weight": upper_weight,
            "point_numerator": point_numerator,
            "point_denominator": point_denominator,
        }

    def export(self) -> dict:
        groups = self._require_fitted()
        return {
            "method": self.method,
            "point_rule": self.point_rule,
            "tie_policy": self.tie_policy,
            "score_groups": groups,
            "calibration_example_count": int(
                sum(group["weight"] for group in groups)
            ),
            "calibration_positive_count": int(
                sum(group["positive_count"] for group in groups)
            ),
        }

"""Auditable complex-level evidence graph for preservation calibration v5.

The graph is intentionally small and non-learned: every node is one independent
``complex_id`` represented by the coordinate-wise median of its standardized
REDUCED features.  Edges and inference use Euclidean distance only.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping, Sequence


DISTANCE_METRIC = "euclidean_standardized"
SUPPORT_SCORE_DEFINITION = "1/(1+kth_distance)"


def _finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _finite_vector(values: Sequence[Any], *, field: str) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{field} must be a numeric sequence")
    converted = [
        _finite_float(value, field=f"{field}[{index}]")
        for index, value in enumerate(values)
    ]
    if not converted:
        raise ValueError(f"{field} cannot be empty")
    return converted


def _strict_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer") from exc
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError(f"{field} must be a non-negative integer")
    return int(numeric)


def _euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("evidence graph vectors must have equal dimensions")
    # ``math.dist`` uses a scaled hypot implementation and therefore avoids the
    # intermediate squaring overflow of ``sum(delta**2)``.  A mathematically
    # unrepresentable distance is saturated only in distance space (never in a
    # predictor) so it remains an auditable, unequivocal out-of-domain case.
    distance = math.dist(
        [float(value) for value in left], [float(value) for value in right]
    )
    return float(distance if math.isfinite(distance) else sys.float_info.max)


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute a graph distance quantile without values")
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("quantile must be within [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class ComplexEvidenceNode:
    """One independent complex represented by a robust feature prototype."""

    complex_id: str
    feature_vector: list[float]
    n_runs: int
    positive_count: int
    negative_count: int
    preservation_rate: float

    def __post_init__(self) -> None:
        if not str(self.complex_id).strip():
            raise ValueError("evidence node complex_id cannot be empty")
        vector = _finite_vector(self.feature_vector, field="feature_vector")
        object.__setattr__(self, "feature_vector", vector)
        n_runs = _strict_nonnegative_int(self.n_runs, field="n_runs")
        positive = _strict_nonnegative_int(
            self.positive_count, field="positive_count"
        )
        negative = _strict_nonnegative_int(
            self.negative_count, field="negative_count"
        )
        if n_runs < 1 or positive < 0 or negative < 0 or positive + negative != n_runs:
            raise ValueError("evidence node class counts must sum to n_runs")
        expected_rate = positive / float(n_runs)
        rate = _finite_float(self.preservation_rate, field="preservation_rate")
        if not 0.0 <= rate <= 1.0 or not math.isclose(
            rate, expected_rate, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("evidence node preservation_rate is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "complex_id": str(self.complex_id),
            "feature_vector": [float(value) for value in self.feature_vector],
            "n_runs": int(self.n_runs),
            "positive_count": int(self.positive_count),
            "negative_count": int(self.negative_count),
            "preservation_rate": float(self.preservation_rate),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComplexEvidenceNode":
        if not isinstance(payload, Mapping):
            raise ValueError("evidence graph node must be an object")
        return cls(
            complex_id=str(payload.get("complex_id") or ""),
            feature_vector=list(payload.get("feature_vector") or []),
            n_runs=payload.get("n_runs", 0),
            positive_count=payload.get("positive_count", 0),
            negative_count=payload.get("negative_count", 0),
            preservation_rate=float(payload.get("preservation_rate", math.nan)),
        )


@dataclass(frozen=True)
class GraphEvidence:
    """Local evidence returned for one standardized REDUCED observation."""

    p_graph: float
    neighbor_ids: list[str]
    neighbor_distances: list[float]
    neighbor_preservation_rates: list[float]
    nearest_distance: float
    kth_distance: float
    effective_neighbor_count: float
    support_score: float
    domain_threshold: float
    domain_supported: bool
    disagreement: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_graph": float(self.p_graph),
            "neighbor_ids": list(self.neighbor_ids),
            "neighbor_distances": [float(value) for value in self.neighbor_distances],
            "neighbor_preservation_rates": [
                float(value) for value in self.neighbor_preservation_rates
            ],
            "nearest_distance": float(self.nearest_distance),
            "kth_distance": float(self.kth_distance),
            "effective_neighbor_count": float(self.effective_neighbor_count),
            "support_score": float(self.support_score),
            "support_score_definition": SUPPORT_SCORE_DEFINITION,
            "domain_threshold": float(self.domain_threshold),
            "domain_supported": bool(self.domain_supported),
            "disagreement": (
                float(self.disagreement) if self.disagreement is not None else None
            ),
        }


class EvidenceGraph:
    """Complex-prototype k-NN graph without a learned embedding or GNN."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        nodes: Sequence[ComplexEvidenceNode],
        k: int,
        domain_threshold: float,
        epsilon: float = 1e-12,
        distance_metric: str = DISTANCE_METRIC,
    ) -> None:
        names = [str(value) for value in feature_names]
        if not names or len(set(names)) != len(names):
            raise ValueError("evidence graph feature_names must be unique and non-empty")
        resolved_nodes = list(nodes)
        if len(resolved_nodes) < 2:
            raise ValueError("evidence graph requires at least two complex nodes")
        if len({node.complex_id for node in resolved_nodes}) != len(resolved_nodes):
            raise ValueError("evidence graph nodes must use distinct complex_ids")
        if any(len(node.feature_vector) != len(names) for node in resolved_nodes):
            raise ValueError("evidence graph node dimension must match feature_names")
        resolved_k = int(k)
        if resolved_k < 1 or resolved_k > len(resolved_nodes):
            raise ValueError("evidence graph k must be between 1 and node count")
        threshold = _finite_float(domain_threshold, field="domain_threshold")
        if threshold < 0.0:
            raise ValueError("domain_threshold must be non-negative")
        eps = _finite_float(epsilon, field="epsilon")
        if eps < 1e-15:
            raise ValueError("epsilon must be at least 1e-15 for stable graph weights")
        if str(distance_metric) != DISTANCE_METRIC:
            raise ValueError(f"unsupported evidence graph distance metric: {distance_metric!r}")
        self.feature_names = names
        self.nodes = resolved_nodes
        self.k = resolved_k
        self.domain_threshold = threshold
        self.epsilon = eps
        self.distance_metric = DISTANCE_METRIC

    @classmethod
    def fit(
        cls,
        feature_vectors: Sequence[Sequence[float]],
        *,
        complex_ids: Sequence[str],
        labels: Sequence[int],
        feature_names: Sequence[str],
        k: int,
        domain_threshold: float | None = None,
        domain_quantile: float = 0.95,
        epsilon: float = 1e-12,
    ) -> "EvidenceGraph":
        vectors = [
            _finite_vector(value, field=f"feature_vectors[{index}]")
            for index, value in enumerate(feature_vectors)
        ]
        ids = [str(value) for value in complex_ids]
        outcomes: list[int] = []
        for index, raw_value in enumerate(labels):
            if isinstance(raw_value, bool):
                raise ValueError(
                    f"labels[{index}] must be the numeric binary value 0 or 1"
                )
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"labels[{index}] must be numeric") from exc
            if not math.isfinite(numeric_value) or numeric_value not in (0.0, 1.0):
                raise ValueError("evidence graph labels must be exactly binary (0 or 1)")
            outcomes.append(int(numeric_value))
        if not vectors or len(vectors) != len(ids) or len(vectors) != len(outcomes):
            raise ValueError("feature_vectors, complex_ids, and labels must align")
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("all evidence graph feature vectors must align")
        groups: dict[str, list[int]] = {}
        for index, complex_id in enumerate(ids):
            if not complex_id:
                raise ValueError("complex_ids cannot contain empty values")
            groups.setdefault(complex_id, []).append(index)
        nodes: list[ComplexEvidenceNode] = []
        for complex_id in sorted(groups):
            indices = groups[complex_id]
            prototype = [
                float(median(vectors[index][column] for index in indices))
                for column in range(dimension)
            ]
            positive = sum(outcomes[index] for index in indices)
            nodes.append(
                ComplexEvidenceNode(
                    complex_id=complex_id,
                    feature_vector=prototype,
                    n_runs=len(indices),
                    positive_count=positive,
                    negative_count=len(indices) - positive,
                    preservation_rate=positive / float(len(indices)),
                )
            )
        resolved_k = int(k)
        if resolved_k < 1 or resolved_k > len(nodes):
            raise ValueError(
                "evidence graph k is infeasible: "
                f"requested={resolved_k}, available_nodes={len(nodes)}"
            )
        if domain_threshold is None:
            if resolved_k > len(nodes) - 1:
                raise ValueError(
                    "learning a graph domain threshold requires at least k+1 complexes"
                )
            domain_threshold = cls.learn_domain_threshold(
                nodes,
                k=resolved_k,
                quantile=domain_quantile,
            )
        return cls(
            feature_names=feature_names,
            nodes=nodes,
            k=resolved_k,
            domain_threshold=float(domain_threshold),
            epsilon=epsilon,
        )

    @staticmethod
    def leave_one_complex_out_kth_distances(
        nodes: Sequence[ComplexEvidenceNode], *, k: int
    ) -> list[float]:
        resolved = list(nodes)
        if len(resolved) < 2:
            raise ValueError("LOCO graph distances require at least two nodes")
        neighbor_count = int(k)
        maximum_k = len(resolved) - 1
        if neighbor_count < 1 or neighbor_count > maximum_k:
            raise ValueError(
                "LOCO graph k is infeasible: "
                f"requested={neighbor_count}, maximum={maximum_k}"
            )
        distances: list[float] = []
        for index, node in enumerate(resolved):
            candidates = sorted(
                _euclidean(node.feature_vector, other.feature_vector)
                for other_index, other in enumerate(resolved)
                if other_index != index
            )
            distances.append(float(candidates[neighbor_count - 1]))
        return distances

    @classmethod
    def learn_domain_threshold(
        cls,
        nodes: Sequence[ComplexEvidenceNode],
        *,
        k: int,
        quantile: float,
    ) -> float:
        distances = cls.leave_one_complex_out_kth_distances(nodes, k=k)
        return float(_quantile(distances, float(quantile)))

    def evaluate(
        self,
        standardized_feature_vector: Sequence[float],
        *,
        p_stat: float | None = None,
    ) -> GraphEvidence:
        vector = _finite_vector(
            standardized_feature_vector, field="standardized_feature_vector"
        )
        if len(vector) != len(self.feature_names):
            raise ValueError("prediction vector dimension does not match evidence graph")
        ranked = sorted(
            (
                (_euclidean(vector, node.feature_vector), node.complex_id, node)
                for node in self.nodes
            ),
            key=lambda item: (item[0], item[1]),
        )[: self.k]
        distances = [float(item[0]) for item in ranked]
        neighbor_nodes = [item[2] for item in ranked]
        exact_indices = [
            index for index, distance in enumerate(distances) if distance == 0.0
        ]
        if exact_indices:
            normalized_weights = [
                1.0 / len(exact_indices) if index in exact_indices else 0.0
                for index in range(len(distances))
            ]
        else:
            raw_weights = [1.0 / (distance + self.epsilon) for distance in distances]
            weight_sum = sum(raw_weights)
            normalized_weights = (
                [weight / weight_sum for weight in raw_weights]
                if weight_sum > 0.0 and math.isfinite(weight_sum)
                else [1.0 / len(raw_weights)] * len(raw_weights)
            )
        p_graph = sum(
            weight * node.preservation_rate
            for weight, node in zip(normalized_weights, neighbor_nodes)
        )
        effective_neighbor_count = 1.0 / sum(
            weight**2 for weight in normalized_weights
        )
        kth_distance = distances[-1]
        disagreement = None
        if p_stat is not None:
            probability = _finite_float(p_stat, field="p_stat")
            if not 0.0 <= probability <= 1.0:
                raise ValueError("p_stat must be within [0, 1]")
            disagreement = abs(probability - p_graph)
        return GraphEvidence(
            p_graph=float(p_graph),
            neighbor_ids=[node.complex_id for node in neighbor_nodes],
            neighbor_distances=distances,
            neighbor_preservation_rates=[
                float(node.preservation_rate) for node in neighbor_nodes
            ],
            nearest_distance=float(distances[0]),
            kth_distance=float(kth_distance),
            effective_neighbor_count=float(effective_neighbor_count),
            support_score=float(1.0 / (1.0 + kth_distance)),
            domain_threshold=float(self.domain_threshold),
            domain_supported=bool(kth_distance <= self.domain_threshold),
            disagreement=(float(disagreement) if disagreement is not None else None),
        )

    def edges(self) -> list[dict[str, Any]]:
        """Return the directed k-NN adjacency used only for audit/inspection."""

        edge_count = min(self.k, len(self.nodes) - 1)
        edges: list[dict[str, Any]] = []
        for source in sorted(self.nodes, key=lambda item: item.complex_id):
            neighbors = sorted(
                (
                    (_euclidean(source.feature_vector, target.feature_vector), target)
                    for target in self.nodes
                    if target.complex_id != source.complex_id
                ),
                key=lambda item: (item[0], item[1].complex_id),
            )[:edge_count]
            for distance, target in neighbors:
                edges.append(
                    {
                        "source": source.complex_id,
                        "target": target.complex_id,
                        "distance": float(distance),
                    }
                )
        return edges

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_unit": "complex",
            "prototype_aggregation": "coordinate_wise_median",
            "distance_metric": self.distance_metric,
            "feature_names": list(self.feature_names),
            "k": int(self.k),
            "domain_threshold": float(self.domain_threshold),
            "epsilon": float(self.epsilon),
            "support_score_definition": SUPPORT_SCORE_DEFINITION,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": self.edges(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceGraph":
        if not isinstance(payload, Mapping):
            raise ValueError("evidence_graph must be an object")
        if payload.get("node_unit") != "complex":
            raise ValueError("evidence graph node_unit must be 'complex'")
        if payload.get("prototype_aggregation") not in {
            None,
            "coordinate_wise_median",
        }:
            raise ValueError("unsupported evidence graph prototype aggregation")
        raw_nodes = payload.get("nodes")
        if not isinstance(raw_nodes, list):
            raise ValueError("evidence graph nodes must be a list")
        return cls(
            feature_names=list(payload.get("feature_names") or []),
            nodes=[ComplexEvidenceNode.from_dict(item) for item in raw_nodes],
            k=int(payload.get("k", 0)),
            domain_threshold=float(payload.get("domain_threshold", math.nan)),
            epsilon=float(payload.get("epsilon", 1e-12)),
            distance_metric=str(payload.get("distance_metric") or ""),
        )


__all__ = [
    "ComplexEvidenceNode",
    "DISTANCE_METRIC",
    "EvidenceGraph",
    "GraphEvidence",
    "SUPPORT_SCORE_DEFINITION",
]

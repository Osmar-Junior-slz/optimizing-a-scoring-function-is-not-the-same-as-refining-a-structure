from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


def _group_digest(groups: tuple[str, ...]) -> str:
    payload = json.dumps(list(groups), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class GroupPartition:
    model_train: np.ndarray
    probability_calibration: np.ndarray
    risk_calibration: np.ndarray
    locked_test: np.ndarray
    groups_by_role: dict[str, tuple[str, ...]]
    random_seed: int
    split_folds: int

    def export(self) -> dict:
        return {
            "method": "label_independent_assignment_with_postsplit_feasibility_gate_v1",
            "grouping_unit": "complex_fingerprint_sha256",
            "random_seed": self.random_seed,
            "split_folds": self.split_folds,
            "roles": {
                role: {
                    "group_count": len(groups),
                    "groups": list(groups),
                    "groups_sha256": _group_digest(groups),
                }
                for role, groups in self.groups_by_role.items()
            },
        }


def four_way_group_partition(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    split_folds: int,
    random_seed: int,
) -> GroupPartition:
    """Partition complete complex fingerprints without inspecting labels/features.

    Fold 0 is the locked test, fold 1 calibrates the LTT policy, fold 2
    calibrates Venn-Abers, and all remaining folds train the base score/domain.
    """

    labels = np.asarray(y, dtype=int).reshape(-1)
    group_values = np.asarray(groups, dtype=str).reshape(-1)
    if len(labels) != len(group_values):
        raise ValueError("labels and groups must have the same length")
    unique_groups = np.unique(group_values)
    if len(unique_groups) < split_folds:
        raise ValueError(
            f"training requires at least {split_folds} independent complex fingerprints"
        )

    rng = np.random.default_rng(random_seed)
    shuffled = rng.permutation(unique_groups)
    folds = [tuple(sorted(part.tolist())) for part in np.array_split(shuffled, split_folds)]
    role_groups = {
        "locked_test": folds[0],
        "risk_calibration": folds[1],
        "probability_calibration": folds[2],
        "model_train": tuple(sorted(group for fold in folds[3:] for group in fold)),
    }

    indices = {
        role: np.flatnonzero(np.isin(group_values, np.asarray(role_group, dtype=str)))
        for role, role_group in role_groups.items()
    }
    if any(len(index) == 0 for index in indices.values()):
        raise ValueError("every group partition must contain at least one row")

    # These are mathematical prerequisites, not criteria used to retry/select a split.
    for role in ("model_train", "probability_calibration"):
        if len(np.unique(labels[indices[role]])) < 2:
            raise ValueError(
                "label-independent assignment failed its post-split two-class "
                f"feasibility gate for {role}; calibration is fail-closed until "
                "more independent complex fingerprints are collected under the predeclared "
                "partition protocol"
            )

    role_sets = [set(values) for values in role_groups.values()]
    for left in range(len(role_sets)):
        for right in range(left + 1, len(role_sets)):
            if role_sets[left] & role_sets[right]:
                raise RuntimeError(
                    "complex_fingerprint_sha256 leakage detected between partitions"
                )

    return GroupPartition(
        model_train=indices["model_train"],
        probability_calibration=indices["probability_calibration"],
        risk_calibration=indices["risk_calibration"],
        locked_test=indices["locked_test"],
        groups_by_role=role_groups,
        random_seed=int(random_seed),
        split_folds=int(split_folds),
    )

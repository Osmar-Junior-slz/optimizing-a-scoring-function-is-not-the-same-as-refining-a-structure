"""Bind paired observed quality to an explicit structural metric convention.

REDUCED-only prediction does not require experimental reference outcomes. This
contract applies to observed labels used in fitting and evaluation.
"""
from __future__ import annotations

import hashlib
import json


def quality_metric_sha256(metric: dict) -> str:
    return hashlib.sha256(json.dumps(metric, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def validate_quality_metric(metric: dict) -> dict:
    if not isinstance(metric, dict):
        raise ValueError("quality_metric must declare the structural evaluation convention")
    version = metric.get("implementation_version")
    if not isinstance(version, str) or not version or not version.startswith("2."):
        raise ValueError("quality_metric requires an explicit supported DockQ v2 version")
    expected = {"schema_version": "structural_quality_metric_v1",
        "method": "official_dockq_v2_protein_protein", "implementation": "DockQ",
        "implementation_version": version, "chain_mapping": "explicit_receptor_ligand",
        "capri_peptide": False, "contact_cutoff_angstrom": 5, "interface_cutoff_angstrom": 10}
    # Canonical serialized comparison also rejects 5.0 vs 5 and Boolean numbers.
    if quality_metric_sha256(metric) != quality_metric_sha256(expected):
        raise ValueError("quality_metric is not the explicit official protein-protein DockQ convention")
    return dict(expected)


def validate_paired_quality_context(frame, metric: dict | None) -> None:
    if metric is None:
        return  # Historical contexts retain their historical interpretation.
    expected = validate_quality_metric(metric)
    required = ("quality_metric_id", "quality_metric_sha256")
    if any(column not in frame for column in required):
        raise ValueError("Paired standard-quality examples require quality_metric_id and quality_metric_sha256; historical scores cannot be relabelled silently")
    if not frame["quality_metric_id"].eq(expected["method"]).all():
        raise ValueError("Paired quality metric identity differs from the training convention")
    if not frame["quality_metric_sha256"].eq(quality_metric_sha256(expected)).all():
        raise ValueError("Paired quality metric digest differs from the training convention")

"""Verify REDUCED pose features shared by training and live inference."""
from __future__ import annotations

import math
from pathlib import Path

from docking_confidence.application.batch import _file_sha256, _load_json_object
from docking_confidence.domain.structure import CanonicalComplexFingerprint, RunBinding
from docking_confidence.features.contact_graph import (
    CONTACT_GRAPH_FEATURE_COLUMNS, CONTACT_GRAPH_METADATA_COLUMNS,
    CONTACT_GRAPH_NULLABLE_AUDIT_COLUMNS, CONTACT_GRAPH_OPTIONAL_METADATA_COLUMNS,
    ContactGraphConfig, describe_interface_contact_graph,
)
from docking_confidence.io.pdb import read_complex_pose

_GRAPH_COLUMNS = (
    *CONTACT_GRAPH_METADATA_COLUMNS, *CONTACT_GRAPH_FEATURE_COLUMNS,
    *CONTACT_GRAPH_NULLABLE_AUDIT_COLUMNS, *CONTACT_GRAPH_OPTIONAL_METADATA_COLUMNS,
)


class MissingReducedFeaturesError(ValueError):
    def __init__(self, features: list[str]) -> None:
        self.features = features
        super().__init__(f"required REDUCED graph features are absent or non-finite: {features}")


def _artifact_path(directory: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("missing batch artifact path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("batch artifact path must be relative and remain in its task")
    path = (directory / relative).resolve(strict=True)
    path.relative_to(directory)
    if not path.is_file():
        raise ValueError(f"batch artifact is not a file: {path}")
    return path


def read_reduced_graph(directory: Path, receipt: dict, config: dict) -> dict:
    result, target, job = receipt["engine_result"], receipt["target"], receipt["job"]
    graph_path = _artifact_path(directory, result.get("graph_features_path"))
    graph = _load_json_object(graph_path)
    missing = [key for key in config["features"]["columns"] if key.startswith("graph_")
               and (isinstance(graph.get(key), bool)
                    or not isinstance(graph.get(key), (int, float))
                    or not math.isfinite(graph[key]))]
    if missing:
        raise MissingReducedFeaturesError(missing)
    pose_path = _artifact_path(directory, result.get("complex_pose_pdb_path"))
    if graph.get("graph_source_file_sha256") != _file_sha256(pose_path):
        raise ValueError("contact_graph source PDB hash mismatch")
    graph_cfg = ContactGraphConfig.from_mapping(config["contact_graph"])
    # Reconstruct the same REDUCED pose descriptor as batch postprocessing. No
    # experimental structure or FULL result participates in this calculation.
    pose = read_complex_pose(
        pose_path, receptor_chains=[target["receptor_chain"]],
        ligand_chains=[target["ligand_chain"]],
        run_binding=RunBinding(complex_id=target["complex_id"], run_id=job["job_id"],
                               algorithm=job["algorithm"], seed=job["seed"],
                               protocol="reduced", engine=result["engine"]),
        complex_fingerprint=CanonicalComplexFingerprint.from_sequences(
            target["receptor_sequence"], target["ligand_sequence"],
            expected_sha256=target["complex_fingerprint_sha256"],
            schema=target["complex_fingerprint_schema"],
        ),
        include_hetero=graph_cfg.include_hetero, exclude_waters=graph_cfg.exclude_waters,
        heavy_atoms_only=graph_cfg.heavy_atoms_only, model_id=graph_cfg.model_id,
        pose_id=f"{job['job_id']}-best-pose", pose_rank=1,
        selection_method="dockingpp_best_computed_pose", pose_source="docking_engine",
    )
    observed = describe_interface_contact_graph(pose, graph_cfg)
    for key in _GRAPH_COLUMNS:
        actual, expected = graph.get(key), observed.get(key)
        if key in CONTACT_GRAPH_METADATA_COLUMNS and key not in graph:
            raise ValueError(f"contact_graph missing provenance: {key}")
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            matches = (not isinstance(actual, bool) and isinstance(actual, (int, float))
                       and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12))
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(f"contact_graph disagrees with its REDUCED pose: {key}")
    return {key: graph.get(key) for key in _GRAPH_COLUMNS}



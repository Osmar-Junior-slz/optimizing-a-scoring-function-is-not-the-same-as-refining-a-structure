"""Input-pose retention, computed without an experimental complex or FULL run."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from time import perf_counter

ALGORITHM_FEATURE_COLUMNS = ("algorithm_ag", "algorithm_pso", "algorithm_hybrid")
EXECUTION_CONTEXT_FEATURE_COLUMNS = (*ALGORITHM_FEATURE_COLUMNS, "input_pose_retention")
EXECUTION_CONTEXT_METADATA_COLUMNS = (
    "input_pose_retention_reference_role", "input_pose_retention_receptor_sha256",
    "input_pose_retention_ligand_sha256", "input_pose_retention_pose_sha256",
)
INPUT_POSE_REFERENCE_ROLE = "supplied_model_pose_only"


def local_execution_context(
    *, algorithm: str, receptor_model_pdb, ligand_template_pdb, pose_json,
    expected_receptor_sha256: str, expected_ligand_sha256: str,
) -> tuple[dict, dict]:
    if algorithm not in {"ag", "pso", "hybrid"}:
        raise ValueError("execution context requires a supported REDUCED algorithm")
    paths = [Path(value).resolve(strict=True) for value in
             (receptor_model_pdb, ligand_template_pdb, pose_json)]
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    if hashes[:2] != [expected_receptor_sha256, expected_ligand_sha256]:
        raise ValueError("input-pose retention input structures changed from the task snapshot")
    # This API has no experimental-reference argument. The supplied ligand is
    # the reference for a predictor of retention, not the experimental outcome.
    from dockingpp.evaluation.structural_metrics import evaluate_saved_pose
    started = perf_counter()
    metrics = evaluate_saved_pose(
        receptor_pdb=paths[0], native_peptide_pdb=paths[1], pose_json=paths[2],
        contact_cutoff_angstrom=4.0, interface_cutoff_angstrom=8.0,
    ).as_dict()
    retention = metrics["dockq"]
    if isinstance(retention, bool) or not isinstance(retention, (int, float)) or not math.isfinite(retention) or not 0 <= retention <= 1:
        raise ValueError("input-pose retention must be a finite value in [0, 1]")
    features = {
        column: float(algorithm == column.removeprefix("algorithm_"))
        for column in ALGORITHM_FEATURE_COLUMNS
    }
    features["input_pose_retention"] = float(retention)
    metadata = {
        "input_pose_retention_reference_role": INPUT_POSE_REFERENCE_ROLE,
        "input_pose_retention_receptor_sha256": hashes[0],
        "input_pose_retention_ligand_sha256": hashes[1],
        "input_pose_retention_pose_sha256": hashes[2],
    }
    return {**features, **metadata}, {
        "method": "supplied_pose_retention_v1", "reference_role": INPUT_POSE_REFERENCE_ROLE,
        "experimental_reference_used": False, "full_result_used": False,
        "calculation_runtime_sec": perf_counter() - started,
    }

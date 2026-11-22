"""Experimental REDUCED-only execution descriptors, never experimental labels."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import numpy as np

PILOT_EXECUTION_FEATURE_COLUMNS = (
    "pilot_input_pose_retention", "pilot_viable_fraction_last",
    "pilot_elite_score_cv", "pilot_elite_gap_relative",
    "pilot_score_improvement_relative", "pilot_plateau_fraction",
    "pilot_final_score_per_residue", "pilot_cheap_score_per_residue",
    "pilot_rescore_gap_per_residue", "pilot_translation_fraction",
    "pilot_rotation_fraction", "pilot_boundary_fraction",
)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"missing or non-finite reduced execution descriptor: {name}")
    return float(value)


def extract_pilot_execution_features(directory: Path, receipt: dict) -> tuple[dict, dict]:
    """Read only the REDUCED result; receipt DockQ/reference/FULL are never read."""
    root = directory.resolve(strict=True)
    if receipt["job"]["protocol"] != "reduced":
        raise ValueError("execution descriptors require a REDUCED receipt")
    relative = Path(receipt["engine_result"]["engine_result_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("REDUCED result path escapes its task")
    path = (root / relative).resolve(strict=True)
    path.relative_to(root)
    payload = path.read_bytes()
    result = json.loads(payload.decode("utf-8-sig"))
    if result.get("structural_metrics_reference_role") != "input_peptide_as_native_reference":
        raise ValueError("input-pose retention must use the supplied peptide, never the experimental reference")
    if result.get("search_space_mode") != "reduced":
        raise ValueError("execution descriptors cannot use an expanded search")
    summaries = result.get("population_confidence_by_pocket")
    if not isinstance(summaries, list) or len(summaries) != 1:
        raise ValueError("pilot local-refinement descriptors require one explicit pocket summary")
    population = summaries[0]
    meta = result["best_pose_meta"]
    if meta.get("input_relative_placement_used") is not True:
        raise ValueError("local-refinement descriptors require the supplied input frame")
    low, high, vector = (
        np.asarray(meta[key], dtype=float) for key in
        ["search_domain_low", "search_domain_high", "solution_vector"]
    )
    if any(value.shape != (6,) or not np.isfinite(value).all() for value in (low, high, vector)):
        raise ValueError("execution descriptor requires finite six-dimensional pose/bounds")
    half_width = (high - low) / 2
    if (half_width <= 0).any():
        raise ValueError("pose domain must have positive widths")
    fractions = np.abs((vector - (high + low) / 2) / half_width)
    residue_count = len(receipt["target"]["ligand_sequence"])
    if residue_count < 1:
        raise ValueError("execution descriptors require the mobile-partner sequence")
    mean = _finite(population["elite_score_mean_last"], "elite_mean")
    best = _finite(population["best_score_final"], "best_score")
    generations = _finite(population["completed_generations"], "completed_generations")
    if generations < 1:
        raise ValueError("execution descriptors require a completed generation")
    final = _finite(result["final_rescore"], "final_rescore")
    cheap = _finite(result["triage_score"], "triage_score")
    values = (
        _finite(result["dockq"], "input_pose_retention"),
        _finite(population["viable_fraction_last"], "viable_fraction_last"),
        _finite(population["elite_score_std_last"], "elite_std") / (abs(mean) + 1e-12),
        _finite(population["elite_score_gap_top1_topk_last"], "elite_gap") / (abs(mean) + 1e-12),
        _finite(population["best_score_improvement_last_window"], "improvement") / (abs(best) + 1e-12),
        _finite(population["plateau_generations_last_window"], "plateau") / generations,
        final / residue_count, cheap / residue_count, (cheap - final) / residue_count,
        float(np.linalg.norm(fractions[:3]) / np.sqrt(3)),
        float(np.linalg.norm(fractions[3:]) / np.sqrt(3)),
        float(np.mean(fractions >= .98)),
    )
    features = dict(zip(PILOT_EXECUTION_FEATURE_COLUMNS, values))
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("derived execution descriptors must be finite")
    trace = {
        "schema_version": "pilot_reduced_execution_descriptors_v1",
        "source_result": str(path), "source_result_sha256": hashlib.sha256(payload).hexdigest(),
        "input_retention_reference": "supplied_peptide_only",
        "experimental_reference_used": False, "full_result_used": False,
        "production_feature_schema_changed": False,
    }
    return features, trace

"""Evaluate one frozen Generalized Venn–Abers margin calibrator retrospectively.

The dataset must have no exact molecular-fingerprint overlap with either fitting
partition. Existing project development data remain retrospective evidence. This
script does not fit models, compare calibrators, choose thresholds or interpret
the generalized prediction set as a coverage/confidence interval.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.metrics import roc_auc_score

from docking_confidence.application.generalized_calibrator import (
    load_calibrator, predict_calibrator, preservation_margin,
)
from docking_confidence.audit.manifest import read_bytes_once, sha256_bytes
from docking_confidence.io.dataset import load_paired_dataset


def _digest_argument(value: str) -> str:
    normalized = value.lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise argparse.ArgumentTypeError("expected a SHA-256 hexadecimal digest")
    return normalized


def _optional_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate(artifact_path, dataset_path, out_dir, *, expected_dataset_sha256=None) -> dict:
    """Apply a verified immutable model to disjoint, unchanged paired examples."""
    artifact_file = Path(artifact_path).resolve(strict=True)
    dataset_file = Path(dataset_path).resolve(strict=True)
    output_root = Path(out_dir).resolve()
    model_root = artifact_file.parent
    if (output_root == model_root or model_root in output_root.parents or
        output_root in model_root.parents):
        raise ValueError("Evaluation outputs must be separate from the immutable model directory.")
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError("Evaluation output directory must be new or empty; existing files are not overwritten.")

    model = load_calibrator(artifact_file)
    artifact = model["artifact"]
    artifact_sha256 = sha256_bytes(model["downloads"]["artifact.json"])
    source_bytes = read_bytes_once(dataset_file)
    source_sha256 = sha256_bytes(source_bytes)
    if expected_dataset_sha256 is not None and source_sha256 != expected_dataset_sha256:
        raise ValueError("Evaluation dataset differs from the expected SHA-256 snapshot.")

    settings = artifact["settings"]
    context_columns = settings["features"]["columns"]
    model_columns = artifact.get("source_feature_columns", artifact["feature_columns"])
    required_base = [column for column in context_columns if column in model_columns]
    frame = load_paired_dataset(
        BytesIO(source_bytes), required_base, settings["contact_graph"],
        docking_task=settings["docking_task"], feature_schema=settings["features"]["schema_version"],
        context_feature_columns=context_columns,
    )
    if "config_sha256" in frame and not frame.config_sha256.eq(settings["source_config_sha256"]).all():
        raise ValueError("Evaluation examples use a different configuration from the frozen model.")
    fit_groups = set().union(*(set(role["groups"]) for role in artifact["partition"].values()))
    evaluation_groups = set(frame.complex_fingerprint_sha256.astype(str))
    overlap = fit_groups & evaluation_groups
    if overlap:
        raise ValueError(f"Evaluation dataset overlaps fitting/calibration molecular fingerprints: {len(overlap)}")

    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "predictions.csv"
    prediction_frame = predict_calibrator(artifact_file, dataset_file, predictions_path)
    if sha256_bytes(read_bytes_once(dataset_file)) != source_sha256:
        raise ValueError("Evaluation dataset changed while predictions were being generated.")
    verified_again = load_calibrator(artifact_file)
    if model["downloads"] != verified_again["downloads"]:
        raise ValueError("Model bundle changed while predictions were being generated.")
    identity = ["complex_fingerprint_sha256", "complex_id", "run_id", "algorithm", "seed"]
    if (len(frame) != len(prediction_frame) or
        not frame[identity].astype(str).reset_index(drop=True).equals(
            prediction_frame[identity].astype(str).reset_index(drop=True))):
        raise ValueError("Predictions are not aligned with evaluation examples.")

    outcome = preservation_margin(frame.dockq_reduced, frame.dockq_full, artifact["target"]["definition"])
    point = prediction_frame.prediction.to_numpy(float)
    lower = prediction_frame.prediction_lower.to_numpy(float)
    upper = prediction_frame.prediction_upper.to_numpy(float)
    if not np.isfinite(np.r_[outcome, point, lower, upper]).all() or np.any(upper < lower):
        raise ValueError("Evaluation outcomes and generalized predictions must be finite and ordered.")
    difference = point - outcome
    residual_sum_squares = float(np.square(difference).sum())
    centered_sum_squares = float(np.square(outcome - outcome.mean()).sum())
    positive = outcome >= 0
    predicted_positive = point >= 0
    tp = int((positive & predicted_positive).sum())
    tn = int((~positive & ~predicted_positive).sum())
    fp = int((~positive & predicted_positive).sum())
    fn = int((positive & ~predicted_positive).sum())
    two_classes = bool(positive.any() and (~positive).any())
    prediction_frame["observed_margin"] = outcome
    prediction_frame["observed_preserved"] = positive
    prediction_frame["predicted_nonnegative_margin_diagnostic_only"] = predicted_positive
    prediction_frame["margin_error"] = difference
    prediction_frame.to_csv(predictions_path, index=False)
    report = {
        "schema_version": "generalized_venn_abers_retrospective_evaluation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "single frozen model; retrospective evaluation on preexisting project development data; not prospective validation",
        "retraining": False, "other_calibrators_compared": False, "new_docking_executions": 0,
        "artifact": str(artifact_file), "artifact_sha256": artifact_sha256,
        "model_manifest_sha256": sha256_bytes(model["downloads"]["audit_manifest.json"]),
        "fitting_dataset_snapshot": artifact["training_dataset_snapshot"],
        "dataset": str(dataset_file), "dataset_sha256": source_sha256,
        "dataset_size_bytes": len(source_bytes),
        "expected_dataset_sha256": expected_dataset_sha256,
        "dataset_unchanged_during_prediction": True,
        "model_bundle_unchanged_during_prediction": True,
        "n_complexes": len(evaluation_groups), "n_runs": len(frame),
        "overlapping_fit_complexes": 0, "target": artifact["target"],
        "regression_metrics": {
            "mae": float(np.abs(difference).mean()),
            "rmse": float(np.sqrt(np.square(difference).mean())),
            "bias_prediction_minus_observed": float(difference.mean()),
            "r2": 1 - residual_sum_squares / centered_sum_squares if centered_sum_squares > 0 else None,
            "r2_unavailable_reason": None if centered_sum_squares > 0 else "constant_observed_margin",
            "mean_observed_margin": float(outcome.mean()), "mean_prediction": float(point.mean()),
        },
        "sign_diagnostics": {
            "interpretation": "positive means margin >= 0; descriptive diagnostic, not an acceptance policy",
            "observed_preserved_runs": int(positive.sum()),
            "observed_not_preserved_runs": int((~positive).sum()),
            "predicted_nonnegative_margin_runs": int(predicted_positive.sum()),
            "predicted_negative_margin_runs": int((~predicted_positive).sum()),
            "agreement": float((positive == predicted_positive).mean()),
            "confusion": {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn},
            "preserved_detection_sensitivity": _optional_ratio(tp, tp + fn),
            "nonpreserved_detection_sensitivity": _optional_ratio(tn, tn + fp),
            "preservation_fraction_among_nonnegative_predictions": _optional_ratio(tp, tp + fp),
            "roc_auc_ranking_by_continuous_prediction": float(roc_auc_score(positive, point)) if two_classes else None,
            "roc_auc_unavailable_reason": None if two_classes else "single_observed_sign_class",
        },
        "generalized_set_diagnostics": {
            "mean_set_width": float((upper - lower).mean()),
            "minimum_set_width": float((upper - lower).min()),
            "maximum_set_width": float((upper - lower).max()),
            "interpretation": "range of generalized Venn prediction set; no response-coverage or confidence level inferred",
        },
        "weighting": "pooled runs; repeated seeds and algorithm runs within each complex are correlated",
        "limitations": [
            "Preexisting development examples were previously available to the project; this is not a prospective or preregistered test.",
            "Exact molecular fingerprints are disjoint; homology, source-level dependence and distribution shift are not ruled out.",
            "Algorithm and seed repetitions are correlated; run counts are not counts of independent molecular inputs.",
            "Only supplied valid paired docking rows are evaluated; unpaired execution failures are not included in these regression diagnostics.",
            "Sign agreement is not probability calibration, and generalized set width is not an individual-outcome coverage guarantee.",
        ],
        "outputs": {"predictions": str(predictions_path), "predictions_sha256": sha256_bytes(read_bytes_once(predictions_path))},
        "evaluator_sha256": sha256_bytes(read_bytes_once(Path(__file__).resolve())),
    }
    report_path = output_root / "evaluation.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="Verified frozen Generalized Venn-Abers artifact.json")
    parser.add_argument("--dataset", required=True, help="Paired docking CSV, disjoint from both fitting partitions")
    parser.add_argument("--out-dir", required=True, help="New or empty evaluation directory, separate from the model")
    parser.add_argument("--expected-dataset-sha256", type=_digest_argument)
    args = parser.parse_args()
    report = evaluate(args.artifact, args.dataset, args.out_dir, expected_dataset_sha256=args.expected_dataset_sha256)
    print(json.dumps({key: report[key] for key in [
        "n_complexes", "n_runs", "overlapping_fit_complexes", "regression_metrics",
        "sign_diagnostics", "generalized_set_diagnostics", "outputs",
    ]}, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()

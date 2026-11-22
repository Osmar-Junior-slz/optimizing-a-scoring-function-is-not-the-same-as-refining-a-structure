"""Explicit, immutable standard-DockQ rescoring of verified saved paired runs.

This is a new measurement, never a relabeling of a historical quality column.
The source CSV's REDUCED descriptors and group identities stay unchanged. A
complete source match is required; an error invalidates the entire conversion.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from docking_confidence.application.batch import _validate_manifest
from docking_confidence.application.operational import validate_training_regime
from docking_confidence.application.reduced_evidence import _artifact_path, read_reduced_graph
from docking_confidence.application.training_dataset import _read_receipt
from docking_confidence.evaluation.official_dockq import (
    METRIC_ID, RESULT_SCHEMA, canonical_sha256, evaluate_standard_complex,
    official_dockq_preflight, quality_metric_descriptor, quality_metric_sha256,
)
from docking_confidence.io.config import config_sha256, load_config, validate_config
from docking_confidence.io.dataset import load_paired_dataset


SCHEMA = "standard_paired_dataset_conversion_v1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def _same(actual, expected):
    if expected is None:
        return actual is None or (not isinstance(actual, (list, dict)) and pd.isna(actual))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (not isinstance(actual, bool) and isinstance(actual, (int, float))
                and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10))
    if isinstance(expected, (list, dict)) and isinstance(actual, str):
        try:
            return json.loads(actual) == expected
        except ValueError:
            return False
    return actual == expected


def _source_pose(directory, receipt, row, config):
    """Prefer actual inference evidence; a fallback stub is never real evidence."""
    result = receipt["engine_result"]
    evidence_path = directory / "confidence/evidence.json"
    source = "postprocess_saved_reduced_pose_v1"
    evidence_sha = None
    graph_receipt = deepcopy(receipt)
    if evidence_path.is_file():
        evidence = _read(evidence_path)
        if evidence.get("schema_version") == "reference_free_reduced_evidence_v1":
            if _sha(evidence_path) != receipt.get("confidence_evidence_sha256"):
                raise ValueError("REDUCED confidence evidence differs from its receipt")
            expected = {"complex_id": receipt["complex_id"], "run_id": receipt["job"]["job_id"],
                "complex_fingerprint_sha256": receipt["target"]["complex_fingerprint_sha256"],
                "engine_pose_json": result["pose_artifact_path"]}
            if any(evidence.get(k) != v for k, v in expected.items()):
                raise ValueError("Reference-free evidence identifies another run")
            for path_key, hash_key in (("engine_pose_json", "engine_pose_sha256"),
                                       ("pose_path", "pose_sha256"), ("graph_path", "graph_sha256")):
                if _sha(_artifact_path(directory, evidence[path_key])) != evidence[hash_key]:
                    raise ValueError("Reference-free evidence artifact SHA-256 mismatch")
            if evidence["pose_sha256"] == row["graph_source_file_sha256"]:
                graph_receipt["engine_result"].update(complex_pose_pdb_path=evidence["pose_path"],
                    graph_features_path=evidence["graph_path"])
                source, evidence_sha = evidence["schema_version"], _sha(evidence_path)
    model = _artifact_path(directory, graph_receipt["engine_result"]["complex_pose_pdb_path"])
    if _sha(model) != row["graph_source_file_sha256"]:
        raise ValueError("Source row does not match the saved REDUCED pose hash; refusing another protocol/pose")
    graph = read_reduced_graph(directory, graph_receipt, config)
    for key, value in graph.items():
        if key in row and not _same(row[key], value):
            raise ValueError(f"Source REDUCED descriptor disagrees with its verified pose: {key}")
    return model, {"feature_source": source, "confidence_evidence_sha256": evidence_sha,
        "graph_path": str(_artifact_path(directory, graph_receipt["engine_result"]["graph_features_path"])),
        "graph_sha256": _sha(_artifact_path(directory, graph_receipt["engine_result"]["graph_features_path"]))}


def _find_reference(target, input_root, reference_roots):
    relative = Path(target["reference_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Reference path must be a safe manifest-relative path")
    checked = []
    for root in dict.fromkeys([str(input_root), *map(str, reference_roots)]):
        candidate = Path(root).expanduser().resolve() / relative
        if candidate.is_file():
            checked.append(str(candidate))
            if _sha(candidate) == target["reference_sha256"]:
                return candidate
    raise ValueError(f"No reference with the manifest SHA-256 was found for {target['complex_id']}: {checked}")


def _pair_context(row, batches, config, reference_roots, source_digest):
    matches = []
    for batch in batches:
        target = batch["targets"].get(row["complex_id"])
        if target is None or target["complex_fingerprint_sha256"] != row["complex_fingerprint_sha256"]:
            continue
        for key in ("complex_fingerprint_schema", "receptor_sequence", "ligand_sequence"):
            if row.get(key) != target.get(key):
                raise ValueError(f"Source row and manifest disagree on component identity: {key}")
        job = batch["jobs"].get(row["run_id"])
        if job is None or job["protocol"] != "reduced":
            continue
        if job["algorithm"] != row["algorithm"] or job["seed"] != row["seed"]:
            continue
        directory = batch["root"] / "targets" / row["complex_id"] / "jobs" / row["run_id"]
        if not (directory / "receipt.json").is_file():
            continue
        reduced = _read_receipt(directory, batch["manifest"], target, job)
        try:
            model, feature_evidence = _source_pose(directory, reduced, row, config)
        except (ValueError, OSError, KeyError) as exc:
            batch.setdefault("match_errors", []).append(str(exc))
            continue
        full_jobs = [j for j in batch["jobs"].values() if j["protocol"] == "full"
            and j["algorithm"] == job["algorithm"] and j["seed"] == job["seed"]
            and j["profile"].get("initialization") == job["profile"].get("initialization")]
        if len(full_jobs) != 1:
            raise ValueError("Expected exactly one FULL counterpart in the same execution plan")
        full_job = full_jobs[0]
        full_dir = batch["root"] / "targets" / row["complex_id"] / "jobs" / full_job["job_id"]
        full = _read_receipt(full_dir, batch["manifest"], target, full_job)
        if full is None:
            raise ValueError("Matching FULL receipt is absent")
        for mode, receipt in (("reduced", reduced), ("full", full)):
            result = receipt["engine_result"]
            for source, column in (("dockq", f"dockq_{mode}"), ("runtime_sec", f"runtime_{mode}_sec"),
                                   ("engine", f"engine_{mode}"), ("scientific", f"scientific_{mode}")):
                if not _same(row[column], result[source]):
                    raise ValueError(f"Source CSV disagrees with its {mode} receipt: {column}")
        for key in config["features"]["columns"]:
            if not key.startswith("graph_") and not _same(row[key], reduced["engine_result"].get(key)):
                raise ValueError(f"Source descriptor disagrees with REDUCED engine result: {key}")
        reference = _find_reference(target, batch["manifest"]["plan"]["input_root"], reference_roots)
        arms = {}
        for mode, receipt, arm_dir, pose in (("reduced", reduced, directory, model),
                ("full", full, full_dir, _artifact_path(full_dir, full["engine_result"]["complex_pose_pdb_path"]))):
            path = arm_dir / "receipt.json"
            binding = {"source_dataset_sha256": source_digest, "batch_id": batch["manifest"]["batch_id"],
                "complex_id": row["complex_id"], "run_id": receipt["job"]["job_id"],
                "algorithm": row["algorithm"], "seed": int(row["seed"]), "protocol": mode,
                "complex_fingerprint_sha256": row["complex_fingerprint_sha256"],
                "source_receipt_path": str(path), "source_receipt_sha256": _sha(path),
                "source_receipt_record_sha256": receipt["record_sha256"]}
            if mode == "reduced":
                binding.update(feature_evidence)
            arms[mode] = {"model": pose, "model_sha256": _sha(pose), "binding": binding}
        matches.append({"target": target, "reference": reference, "arms": arms,
            "manifest_path": batch["manifest_path"], "manifest_sha256": batch["file_sha256"]})
    if len(matches) != 1:
        errors = [e for b in batches for e in b.get("match_errors", [])][-3:]
        raise ValueError(f"Expected one exact source pair, found {len(matches)}; details={errors}")
    return matches[0]


def verify_standard_quality_records(frame, dataset_root, metric):
    """Validate observed-quality sidecars or raise a UI-safe ``ValueError``.

    This is never called for REDUCED-only prediction.
    """
    required = {"complex_id", "run_id", "algorithm", "seed", "complex_fingerprint_sha256",
        "quality_metric_id", "quality_metric_sha256", "quality_reference_sha256"}
    required.update(f"{prefix}_{mode}{suffix}" for mode in ("reduced", "full")
        for prefix, suffix in (("quality", "_record_path"), ("quality", "_record_sha256"),
                               ("quality", "_model_sha256"), ("dockq", "")))
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Standard paired quality requires evaluator records; missing columns: {sorted(missing)}")
    try:
        return _verify_standard_quality_records(frame, dataset_root, metric)
    except (OSError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError(f"Standard quality evaluator record is absent or malformed: {exc}") from exc


def _verify_standard_quality_records(frame, dataset_root, metric):
    root = Path(dataset_root).resolve(strict=True)
    metric_digest = quality_metric_sha256(metric)
    for row in frame.to_dict("records"):
        if row.get("quality_metric_id") != METRIC_ID or row.get("quality_metric_sha256") != metric_digest:
            raise ValueError("Paired row does not declare the frozen standard quality metric")
        for mode in ("reduced", "full"):
            # Accept older Windows snapshots while writing portable new paths.
            relative = Path(row[f"quality_{mode}_record_path"].replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Quality record path must remain inside its dataset snapshot")
            path = (root / relative).resolve(strict=True)
            path.relative_to(root)
            record = _read(path)
            if _sha(path) != row[f"quality_{mode}_record_sha256"]:
                raise ValueError("Quality record file SHA-256 mismatch")
            core = {k:v for k,v in record.items() if k != "record_sha256"}
            if record.get("record_sha256") != canonical_sha256(core):
                raise ValueError("Quality record content integrity failed")
            if (record.get("schema_version") != RESULT_SCHEMA or record.get("quality_metric_id") != METRIC_ID
                    or record.get("quality_metric") != metric
                    or quality_metric_sha256(record["quality_metric"]) != metric_digest
                    or record.get("quality_metric_sha256") != metric_digest):
                raise ValueError("Observed quality was not measured by the frozen standard metric")
            binding = record.get("source_binding", {})
            for key, value in {"complex_id": row["complex_id"], "algorithm": row["algorithm"],
                    "seed": int(row["seed"]), "protocol": mode,
                    "complex_fingerprint_sha256": row["complex_fingerprint_sha256"]}.items():
                if binding.get(key) != value:
                    raise ValueError(f"Quality record identifies another run: {key}")
            if mode == "reduced" and binding.get("run_id") != row["run_id"]:
                raise ValueError("Quality record identifies another REDUCED run")
            if not _same(row[f"dockq_{mode}"], record["metrics"]["dockq"]):
                raise ValueError("Paired quality differs from its official evaluator record")
            if record["inputs"]["model_sha256"] != row[f"quality_{mode}_model_sha256"]:
                raise ValueError("Paired model SHA-256 differs from the scored pose")
            if record["inputs"]["reference_sha256"] != row["quality_reference_sha256"]:
                raise ValueError("Paired reference SHA-256 differs from the scored reference")
    return {"valid": True, "rows": len(frame), "records": 2 * len(frame), "quality_metric_sha256": metric_digest}


def build_standard_paired_dataset(source_dataset_path, batch_dirs, output_dir, *, source_config_path,
        reference_roots=(), python_executable=None, dependency_path=None, max_workers=1,
        expected_source_sha256=None, progress_callback=None):
    """Rescore exact saved pairs, retaining every source row or invalidating all."""
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or not 1 <= max_workers <= 32:
        raise ValueError("max_workers must be an integer in [1,32]")
    source = Path(source_dataset_path).resolve(strict=True)
    destination = Path(output_dir).resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("Standard dataset output must be new or empty")
    if destination == source.parent or destination in source.parents:
        raise ValueError("Standard output must be separate from source data")
    snapshot = source.read_bytes()
    source_digest = hashlib.sha256(snapshot).hexdigest()
    if expected_source_sha256 is not None and expected_source_sha256 != source_digest:
        raise ValueError("Source dataset changed after its selection")
    config = load_config(source_config_path)
    frame = load_paired_dataset(BytesIO(snapshot), config["features"]["columns"], config["contact_graph"],
        docking_task=config.get("docking_task"), feature_schema=config["features"]["schema_version"])
    if not frame.config_sha256.eq(config_sha256(config)).all():
        raise ValueError("Source CSV configuration SHA-256 differs from the supplied source config")
    preflight = official_dockq_preflight(python_executable=python_executable, dependency_path=dependency_path)
    if not preflight["available"]:
        raise RuntimeError(f"Official DockQ dependency preflight failed: {preflight}")
    metric = quality_metric_descriptor(preflight["implementation_version"])
    metric_digest = quality_metric_sha256(metric)
    if "quality_metric" in config and quality_metric_sha256(config["quality_metric"]) != metric_digest:
        raise ValueError("Source configuration requests a different quality metric")
    new_config = validate_config({**deepcopy(config), "quality_metric": metric})
    if isinstance(batch_dirs, (str, Path)):
        batch_dirs = [batch_dirs]
    batches = []
    for value in batch_dirs:
        root = Path(value).resolve(strict=True)
        if destination == root or root in destination.parents or destination in root.parents:
            raise ValueError("Standard dataset output must be separate from all batch inputs")
        manifest_path = root / "batch_manifest.json"
        manifest = _read(manifest_path)
        _validate_manifest(manifest)
        validate_training_regime(config, manifest["plan"])
        batches.append({"root": root, "manifest": manifest, "manifest_path": str(manifest_path),
            "file_sha256": _sha(manifest_path),
            "targets": {t["complex_id"]:t for t in manifest["plan"]["targets"]},
            "jobs": {j["job_id"]:j for j in manifest["plan"]["jobs"]}})
    if not batches:
        raise ValueError("At least one verified batch is required")
    destination.mkdir(parents=True, exist_ok=True)
    config_path = destination / "resolved_config.json"
    _write(config_path, new_config)
    report = {"schema_version": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid": False, "dataset_path": None, "source_dataset_path": str(source),
        "source_dataset_sha256": source_digest, "source_config_sha256": config_sha256(config),
        "config_path": str(config_path), "config_sha256": config_sha256(new_config),
        "quality_metric": metric, "quality_metric_sha256": metric_digest,
        "implementation_preflight": preflight, "source_batches": [{"manifest_path":b["manifest_path"],
            "file_sha256":b["file_sha256"]} for b in batches], "issues": [], "pairs": []}
    original_rows = pd.read_csv(BytesIO(snapshot), float_precision="round_trip").to_dict("records")
    completed = {}

    def convert(index, row):
        context = _pair_context(row, batches, config, reference_roots, source_digest)
        output = dict(row)
        output.update(config=str(config_path), config_sha256=config_sha256(new_config),
            quality_metric_id=METRIC_ID, quality_metric_sha256=metric_digest,
            execution_contract_sha256=canonical_sha256(new_config.get("execution_contract")),
            quality_reference_sha256=context["target"]["reference_sha256"])
        evidence = {"row_index": index, "complex_id": row["complex_id"], "run_id": row["run_id"],
            "complex_fingerprint_sha256": row["complex_fingerprint_sha256"], "arms": {}}
        for mode in ("reduced", "full"):
            arm = context["arms"][mode]
            result = evaluate_standard_complex(arm["model"], context["reference"],
                receptor_chain=context["target"]["receptor_chain"], ligand_chain=context["target"]["ligand_chain"],
                output_dir=destination / "quality" / f"{index:05d}" / mode,
                python_executable=python_executable, dependency_path=dependency_path,
                expected_metric_sha256=metric_digest, expected_model_sha256=arm["model_sha256"],
                expected_reference_sha256=context["target"]["reference_sha256"],
                expected_version=metric["implementation_version"],
                source_binding=arm["binding"])
            if _sha(arm["binding"]["source_receipt_path"]) != arm["binding"]["source_receipt_sha256"]:
                raise ValueError("Source receipt changed during rescoring")
            path = Path(result["metrics_path"])
            output.update({f"historical_quality_{mode}": row[f"dockq_{mode}"],
                f"dockq_{mode}": result["dockq"], f"quality_{mode}_record_path": path.relative_to(destination).as_posix(),
                f"quality_{mode}_record_sha256": _sha(path),
                f"quality_{mode}_record_content_sha256": result["record_sha256"],
                f"quality_{mode}_model_sha256": arm["model_sha256"]})
            evidence["arms"][mode] = {**arm["binding"], "quality_record_path": path.relative_to(destination).as_posix(),
                "quality_record_sha256": _sha(path), "model_sha256": arm["model_sha256"]}
        if _sha(context["manifest_path"]) != context["manifest_sha256"]:
            raise ValueError("Batch manifest changed during rescoring")
        # Old model predictions/class labels have no meaning after remeasurement.
        for column in list(output):
            if column.startswith("probability") or column in {"base_raw_score", "base_ridge_score", "base_extra_trees_score",
                    "prediction", "prediction_lower", "prediction_upper", "preserved", "in_applicability_domain",
                    "unseen_complex", "trusted_reduced_input_provenance", "decision", "action"}:
                del output[column]
        return output, evidence

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(convert, i, row): i for i, row in enumerate(original_rows)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                completed[index] = future.result()
            except Exception as exc:
                report["issues"].append({"row_index": index, "complex_id": original_rows[index]["complex_id"],
                    "run_id": original_rows[index]["run_id"], "error_type": type(exc).__name__, "message": str(exc)})
            if progress_callback is not None:
                progress_callback(len(completed) / len(original_rows), f"Official DockQ: {len(completed)}/{len(original_rows)} pairs")
    rows = [completed[i][0] for i in sorted(completed)]
    report["pairs"] = [completed[i][1] for i in sorted(completed)]
    report["counts"] = {"source_rows": len(original_rows), "rescored_pairs": len(rows),
        "failed_rows": len(report["issues"]), "source_complexes": int(frame.complex_fingerprint_sha256.nunique())}
    if _sha(source) != source_digest:
        report["issues"].append({"message": "Source dataset changed during rescoring"})
    if rows:
        candidate_path = destination / "paired_dataset.csv"
        paired = pd.DataFrame(rows)
        with candidate_path.open("x", encoding="utf-8", newline="") as handle:
            paired.to_csv(handle, index=False, lineterminator="\n")
        report.update(candidate_dataset_path=str(candidate_path), dataset_sha256=_sha(candidate_path))
        try:
            checked = load_paired_dataset(candidate_path, new_config["features"]["columns"], new_config["contact_graph"],
                docking_task=new_config.get("docking_task"), feature_schema=new_config["features"]["schema_version"])
            verify_standard_quality_records(checked, destination, metric)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            report["issues"].append({"message": f"Converted dataset verification failed: {exc}"})
        if not report["issues"] and len(rows) == len(original_rows):
            report.update(valid=True, dataset_path=str(candidate_path))
    report["report_sha256"] = canonical_sha256(report)
    _write(destination / "conversion_manifest.json", report)
    return report

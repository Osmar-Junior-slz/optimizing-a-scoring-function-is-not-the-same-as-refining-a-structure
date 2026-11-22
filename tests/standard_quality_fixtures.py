"""Explicit synthetic official-record fixtures, never production observations."""
import hashlib
import json
from pathlib import Path

from docking_confidence.evaluation.official_dockq import canonical_sha256, quality_metric_sha256


def bind_standard_quality_records(frame, dataset_root, metric):
    """Add coherent sidecars to a synthetic paired frame, returning a copy."""
    frame = frame.copy()
    root = Path(dataset_root)
    digest = quality_metric_sha256(metric)
    frame["quality_metric_id"], frame["quality_metric_sha256"] = metric["method"], digest
    for index, row in frame.iterrows():
        reference_hash = hashlib.sha256((str(row["complex_fingerprint_sha256"]) + "reference-fixture").encode()).hexdigest()
        frame.loc[index, "quality_reference_sha256"] = reference_hash
        for mode in ("reduced", "full"):
            model_hash = hashlib.sha256((str(row["complex_id"]) + str(row["run_id"]) + mode).encode()).hexdigest()
            record = {"schema_version": "official_dockq_reference_evaluation_v1", "fixture": True,
                "quality_metric_id": metric["method"], "quality_metric": metric, "quality_metric_sha256": digest,
                "reference_used_for_search": False,
                "inputs": {"model_sha256": model_hash, "reference_sha256": reference_hash},
                "metrics": {"dockq": float(row[f"dockq_{mode}"])},
                "source_binding": {"complex_id": row["complex_id"], "run_id": row["run_id"] if mode == "reduced" else str(row["run_id"]) + "-full-fixture",
                    "algorithm": row["algorithm"], "seed": int(row["seed"]), "protocol": mode,
                    "complex_fingerprint_sha256": row["complex_fingerprint_sha256"]}}
            record["record_sha256"] = canonical_sha256(record)
            path = root / "quality_fixture" / str(index) / f"{mode}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            frame.loc[index, f"quality_{mode}_record_path"] = str(path.relative_to(root))
            frame.loc[index, f"quality_{mode}_record_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            frame.loc[index, f"quality_{mode}_model_sha256"] = model_hash
    return frame

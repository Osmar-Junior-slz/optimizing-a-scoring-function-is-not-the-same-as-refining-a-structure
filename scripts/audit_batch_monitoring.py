"""Read-only audit of the selected manifest and input directory inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from docking_confidence.application.batch import _load_json_object, _validate_manifest
from docking_confidence.application.batch_monitor import load_batch_monitor
from docking_confidence.application.batch_telemetry import utc_now


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.batch / "batch_manifest.json"
    manifest = _load_json_object(manifest_path)
    _validate_manifest(manifest)
    view = load_batch_monitor(args.batch)
    if view is None:
        raise ValueError("No persisted batch status")
    plan = view["plan"]
    directories = sorted(p.name for p in Path(plan["input_root"]).iterdir() if p.is_dir())
    expected = set(plan["complex_ids"])
    report = {
        "observed_at_utc": utc_now(), "manifest_path": str(manifest_path.resolve()),
        "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "manifest_integrity_verified": True, "plan": plan,
        "input_directory_count": len(directories), "input_directories": directories,
        "manifest_ids_missing_from_input": sorted(expected - set(directories)),
        "input_directories_absent_from_manifest": sorted(set(directories) - expected),
        "counts": view["counts"], "processed": view["processed"],
        "global_fraction": view["global_fraction"], "state": view["state"],
        "health": view["health"], "worker_pid": view["worker_pid"],
        "status_updated_at_utc": view["status_updated_at_utc"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "input_directory_count", "counts", "worker_pid", "health",
        "manifest_ids_missing_from_input", "input_directories_absent_from_manifest",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

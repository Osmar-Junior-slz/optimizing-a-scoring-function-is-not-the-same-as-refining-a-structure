"""Derive probability diagnostics from saved predictions; never run docking or fit."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import pandas as pd

from docking_confidence.application.probability_reporting import DiagnosticOptions, saved_test_diagnostics
from docking_confidence.application.training_results import load_training_result
from docking_confidence.audit.manifest import sha256_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--training-dir", help="Verified immutable training bundle")
    source.add_argument("--predictions", help="Standalone saved test CSV; provenance is not verified")
    parser.add_argument("--output", required=True, help="New JSON file; an existing file is never overwritten")
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20261010)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()
    options = DiagnosticOptions(args.bootstrap_repetitions, args.bootstrap_seed, args.confidence_level)
    if args.training_dir:
        verified = load_training_result(args.training_dir)
        payload = verified["downloads"]["locked_test_predictions.csv"]
        source_path = verified["paths"]["locked_test_predictions.csv"]
        provenance = "verified_training_bundle"
    else:
        source_path = str(Path(args.predictions).resolve())
        payload = Path(source_path).read_bytes()
        provenance = "standalone_csv_provenance_not_verified"
    frame = pd.read_csv(BytesIO(payload), float_precision="round_trip")
    if "partition" in frame and not frame["partition"].eq("locked_test").all():
        raise ValueError("diagnostic evaluation requires locked_test predictions; calibration-fit rows are not a test")
    result = saved_test_diagnostics(frame, options=options)
    result["source"] = {"path": source_path, "sha256": sha256_bytes(payload), "verification": provenance}
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(destination)


if __name__ == "__main__":
    main()

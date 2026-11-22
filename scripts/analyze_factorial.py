"""Analyze an existing factorial batch without importing or running docking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Permit the manual command from a source checkout without package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docking_confidence.application.factorial_reporting import (
    FactorialAnalysisError, analyze_factorial_batch, write_factorial_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, help="Defaults to BATCH/factorial_analysis")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    try:
        report = analyze_factorial_batch(args.batch, bootstrap_repetitions=args.bootstrap_repetitions,
                                         bootstrap_seed=args.bootstrap_seed)
        paths = write_factorial_report(report, args.output_dir or args.batch / "factorial_analysis",
                                       plots=not args.no_plots)
    except (FactorialAnalysisError, OSError, ValueError) as exc:
        parser.exit(2, f"Factorial analysis failed: {exc}\n")
    print(json.dumps({"interpretation": report["interpretation"], "counts": report["counts"],
                      "files": paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

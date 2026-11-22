"""Compare fixed pilot candidates without training an authorized docking gate."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docking_confidence.application.pilot_diagnostics import run_pilot_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = run_pilot_comparison(args.preparation_dir, args.output_dir)
    print(json.dumps(report["counts"]), flush=True)
    for name, result in report["candidates"].items():
        before, after = result["before"], result["after_on_common_rows"]
        print(json.dumps({
            "candidate": name,
            "before_brier": before["preservation"]["brier_score"],
            "before_roc_auc": before["preservation"]["roc_auc"],
            "after_brier": after["preservation"]["brier_score"],
            "after_roc_auc": after["preservation"]["roc_auc"],
            "violation_ap": after["violation"]["average_precision"],
            "domain_eligible_rows": result["domain_eligible_rows"],
            "policy_at_0.9": result["hypothetical_policy_on_common_rows"][-1],
        }), flush=True)
    print("Reference Brier:", report["reference"]["training_prevalence"]["preservation"]["brier_score"])


if __name__ == "__main__":
    main()

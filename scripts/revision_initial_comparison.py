"""Compare supplied poses with the 600 factorial outcomes and 100 uniform-control winners."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "output" / "revisao_professor"
INITIAL = REVISION / "pose_inicial"
THRESHOLD = 0.23
DELTA_TOLERANCE = 1e-12
METHODS = ("ag", "pso", "hybrid", "random_control")
EXTENTS = ("reduced", "expanded")
INITIALIZATIONS = ("uniform", "seeded")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dockq(value):
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"DockQ outside [0,1]: {value!r}")
    return score


def initial_scores(path):
    scores = {}
    for row in read_csv(path):
        target = row["complex_id"]
        if not target or target in scores:
            raise ValueError(f"Missing or repeated initial complex: {target!r}")
        scores[target] = dockq(row["initial_dockq"])
    if len(scores) != 50:
        raise ValueError(f"Expected 50 initial scores, received {len(scores)}")
    return scores


def historical_outcomes(factorial, control):
    for row in read_csv(factorial):
        extent, initialization = row["condition"].split("_")
        expected_state = "scored" if row["valid_pose"] == "1" else "no_valid_pose"
        if (row["outcome_state"] != expected_state or row["seed"] != "1"
                or row["actual_eval_count"] != "7600"):
            raise ValueError(f"Incomplete or incompatible factorial run: {row['job_id']}")
        yield row, row["algorithm"], extent, initialization, row["primary"]
    for row in read_csv(control):
        if row["seed"] != "1" or row["objective_evaluations"] != "7600":
            raise ValueError(f"Incompatible control for {row['complex_id']}")
        if row["valid_pose"] == "0" and not (row["extent"] == "reduced" and row["hard_clash"] == "True"):
            raise ValueError(f"Unknown reason for an invalid control winner: {row['complex_id']}")
        yield row, "random_control", row["extent"], "uniform", row["success"]


def paired_row(raw, method, extent, initialization, stored_success, initial):
    if raw["valid_pose"] not in {"0", "1"} or stored_success not in {"0", "1"}:
        raise ValueError(f"Invalid outcome flags for {raw['complex_id']}")
    valid = int(raw["valid_pose"])
    if bool(raw["dockq"].strip()) != bool(valid):
        raise ValueError(f"Final DockQ availability disagrees with pose validity for {raw['complex_id']}")
    final = dockq(raw["dockq"]) if valid else None
    final_success = final is not None and final >= THRESHOLD
    if final_success != bool(int(stored_success)):
        raise ValueError(f"Stored outcome disagrees with recomputed success for {raw['complex_id']}")
    acceptable = initial >= THRESHOLD
    transition = {
        (True, True): "maintained", (False, True): "rescued",
        (True, False): "lost", (False, False): "neither",
    }[acceptable, final_success]
    delta, direction = None, None
    if valid:
        delta = final - initial
        if abs(delta) <= DELTA_TOLERANCE:
            direction = "unchanged"
        else:
            direction = "improved" if delta > 0 else "worsened"
    return {
        "complex_id": raw["complex_id"], "method": method, "extent": extent,
        "initialization": initialization, "initial_dockq": initial,
        "initial_acceptable": int(acceptable), "final_valid": valid, "final_rejected": 1 - valid,
        "final_dockq": final, "final_success": int(final_success), "transition": transition,
        "delta_dockq": delta, "score_direction": direction,
    }


def summarize_condition(condition, rows):
    valid = [r for r in rows if r["final_valid"]]
    transitions = Counter(r["transition"] for r in rows)
    directions = Counter(r["score_direction"] for r in valid)
    rejected = Counter(r["transition"] for r in rows if r["final_rejected"])
    return {
        "method": condition[0], "extent": condition[1], "initialization": condition[2],
        "n_planned": len(rows), "initial_acceptable": sum(r["initial_acceptable"] for r in rows),
        "final_success": sum(r["final_success"] for r in rows),
        **{k: transitions[k] for k in ("maintained", "rescued", "lost", "neither")},
        "final_valid": len(valid), "final_rejected": len(rows) - len(valid),
        "lost_rejected": rejected["lost"], "neither_rejected": rejected["neither"],
        **{k + "_valid": directions[k] for k in ("improved", "unchanged", "worsened")},
        "median_initial_dockq_all": statistics.median(r["initial_dockq"] for r in rows),
        "median_initial_dockq_valid_pairs": statistics.median(r["initial_dockq"] for r in valid),
        "median_final_dockq_valid": statistics.median(r["final_dockq"] for r in valid),
        "median_delta_dockq_valid": statistics.median(r["delta_dockq"] for r in valid),
    }


def check_published_counts(summary, factorial, control):
    method, extent, initialization = (summary[k] for k in ("method", "extent", "initialization"))
    if method == "random_control":
        cell = control["results"][extent]
        expected_success = cell["successes"]
    else:
        cell = factorial["by_method"][method]["cells"][extent + "_" + initialization]
        expected_success = cell["acceptable_poses"]
    if (summary["n_planned"], summary["final_success"], summary["final_valid"]) != (
            cell["planned"], expected_success, cell["valid_poses"]):
        raise ValueError(f"Final counts differ from published results: {(method, extent, initialization)}")


def analyze(initial_path, factorial_dir, control_dir, output):
    inputs = {
        "initial_scores": initial_path,
        "factorial_observations": factorial_dir / "factorial_observations.csv",
        "factorial_results": factorial_dir / "factorial_results.json",
        "control_observations": control_dir / "observations.csv",
        "control_summary": control_dir / "summary.json",
    }
    exports = ("paired_comparisons.csv", "condition_summary.csv", "summary.json")
    if {path.resolve() for path in inputs.values()} & {(output / name).resolve() for name in exports}:
        raise ValueError("The comparison output would overwrite an input file")
    sources = {k: digest(path) for k, path in inputs.items()}
    factorial = json.loads(inputs["factorial_results"].read_text(encoding="utf-8-sig"))
    control = json.loads(inputs["control_summary"].read_text(encoding="utf-8-sig"))
    if not factorial["complete"] or factorial["observations_sha256"] != sources["factorial_observations"]:
        raise ValueError("Factorial observations do not match the published complete analysis")
    initial = initial_scores(initial_path)
    groups = defaultdict(dict)
    for raw, method, extent, init, success in historical_outcomes(
            inputs["factorial_observations"], inputs["control_observations"]):
        target = raw["complex_id"]
        group = groups[method, extent, init]
        if target not in initial or target in group:
            raise ValueError(f"Unknown or repeated paired observation: {(target, method, extent, init)}")
        group[target] = paired_row(raw, method, extent, init, success, initial[target])
    conditions = [(method, extent, init) for method in METHODS for extent in EXTENTS
                  for init in (("uniform",) if method == "random_control" else INITIALIZATIONS)]
    if set(groups) != set(conditions):
        raise ValueError("Expected the twelve factorial and two control conditions")
    comparisons, summaries = [], []
    for condition in conditions:
        if groups[condition].keys() != initial.keys():
            raise ValueError(f"Missing paired complexes in {condition}")
        rows = [groups[condition][target] for target in sorted(initial)]
        summary = summarize_condition(condition, rows)
        check_published_counts(summary, factorial, control)
        comparisons.extend(rows)
        summaries.append(summary)

    ordered = sorted(initial.values())
    q1, median, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    initial_ok = sum(value >= THRESHOLD for value in ordered)
    result = {
        "schema": "article_initial_paired_comparison_v1", "inputs_sha256": sources,
        "paired_observations": len(comparisons), "n_conditions": len(summaries),
        "initial_quality": {
            "n_complexes": len(ordered), "acceptable": initial_ok, "below_threshold": len(ordered) - initial_ok,
            "acceptable_rate": initial_ok / len(ordered), "min": ordered[0], "q1": q1, "median": median,
            "q3": q3, "max": ordered[-1], "mean": statistics.mean(ordered),
        },
        "definitions": {
            "threshold_inclusive": THRESHOLD,
            "initial_acceptable": "DockQ of the supplied pose at or above the threshold.",
            "final_success": "Valid final winner with DockQ at or above the threshold, under its original regional gate.",
            "transitions": "maintained = initial acceptable/final success; rescued = initial below/final success; "
                           "lost = initial acceptable/final failure; neither = initial below/final failure.",
            "transition_denominator": "All 50 scheduled complexes in each condition; structural rejection is failure.",
            "paired_score_denominator": "final_valid in each condition; missing final DockQ, delta and direction on rejection.",
            "delta_dockq": "Final minus initial DockQ for the same complex; reported median is the median of paired differences.",
            "direction_tolerance_absolute": DELTA_TOLERANCE,
            "unchanged": "Absolute paired difference at most the direction tolerance; "
                         "tolerance is not used to classify acceptability.",
            "units": "Each condition reuses the same 50 starting poses. Descriptive comparison, no independent replication across conditions.",
            "quartiles": "Linear interpolation at (n-1)p (inclusive sample quantiles).",
        },
        "published_final_counts_verified": True, "conditions": summaries,
    }
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as stage:
        stage = Path(stage)
        write_csv(stage / "paired_comparisons.csv", comparisons)
        write_csv(stage / "condition_summary.csv", summaries)
        result["output_sha256"] = {name: digest(stage / name) for name in exports if name.endswith(".csv")}
        (stage / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name in exports:
            (stage / name).replace(output / name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, default=INITIAL / "entrada/initial_scores.csv")
    parser.add_argument("--factorial-dir", type=Path, default=ROOT / "output/fatorial_dockq_20261119")
    parser.add_argument("--control-dir", type=Path, default=REVISION / "controle")
    parser.add_argument("--output", type=Path, default=INITIAL / "comparacao")
    args = parser.parse_args()
    result = analyze(args.initial, args.factorial_dir, args.control_dir, args.output)
    print(f"{result['paired_observations']} paired observations across {result['n_conditions']} conditions: {args.output}")


if __name__ == "__main__":
    main()

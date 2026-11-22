"""Compare input and winning poses using DockQ >= 0.23 and minimum distance >= 1.2 A."""
import argparse
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from revision_initial_comparison import INITIAL, REVISION, ROOT, THRESHOLD, digest, read_csv, write_csv
from revision_random_control import verified_record


DISTANCE = 1.2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def checked_gate(path, expected_hash, valid):
    data = path.read_bytes()
    import hashlib
    require(hashlib.sha256(data).hexdigest() == expected_hash, f"Winner evidence changed: {path}")
    artifact = json.loads(data)
    if path.name == "result.json":
        require(not valid and artifact["failure_reason"] == "no_valid_reduced_pose"
                and artifact["final_pose_available"] is False, f"Unexpected rejected winner: {path}")
        pocket, = artifact["per_pocket_results"]
        meta = pocket["best_pose_meta"]
    else:
        meta = artifact["meta"]
    gate = meta["final_hard_clash_gate"]
    minimum = gate["min_distance"]
    require(gate["enabled"] is True and gate["threshold"] == DISTANCE
            and isinstance(minimum, (float, int)) and math.isfinite(minimum) and minimum >= 0
            and gate["triggered"] is (minimum < DISTANCE), f"Invalid recorded gate: {path}")
    return minimum


def final_evidence(workspace, eligible_index, control_dir, row):
    target, method, extent, init = (row[key] for key in ("complex_id", "method", "extent", "initialization"))
    valid = bool(int(row["final_valid"]))
    if method == "random_control":
        trial = control_dir / "trials" / target / extent
        receipt = verified_record(trial / "trial.json")
        winner = verified_record(trial / "selected.json")
        source = "artifacts/best_pose.json"
        path = trial / source
        expected_hash = receipt["files"][source]
        registered = receipt["row"]
        require(registered["complex_id"] == target and registered["extent"] == extent
                and registered["valid_pose"] == int(valid) and registered["dockq"] ==
                (float(row["final_dockq"]) if valid else None)
                and receipt["files"]["selected.json"] == digest(trial / "selected.json")
                and expected_hash == winner["winner_sha256"], f"Control binding mismatch: {trial}")
        minimum = checked_gate(path, expected_hash, valid)
        require(registered["hard_clash"] is (minimum < DISTANCE), f"Control steric mismatch: {path}")
        harmonized_success = registered["success_common_gate"]
    else:
        record = eligible_index[target, method, extent + "_" + init]
        job = workspace / "runs/fatorial_dockq_20261119" / record["batch_shard"] / "batch/targets" / target / "jobs" / record["job_id"]
        path = (job / record["gate_evidence_path_relative_to_job"]).resolve(strict=True)
        require(path.is_relative_to(job) and record["distance_evidence"] == "recorded_minimum"
                and record["native_valid_pose"] == row["final_valid"] and record["dockq"] == row["final_dockq"]
                and float(record["threshold_angstrom"]) == DISTANCE, f"Factorial binding mismatch: {path}")
        expected_hash = record["gate_evidence_sha256"]
        minimum = checked_gate(path, expected_hash, valid)
        require(minimum == float(record["min_distance_angstrom"])
                and int(record["gate_triggered"]) == int(minimum < DISTANCE), f"Factorial steric mismatch: {path}")
        harmonized_success = int(record["primary"])
    return minimum, harmonized_success, path.relative_to(workspace).as_posix(), expected_hash


def run(workspace):
    workspace = workspace.resolve(strict=True)
    factorial = workspace / "output/fatorial_dockq_20261119"
    revision = workspace / REVISION.relative_to(ROOT)
    initial = workspace / INITIAL.relative_to(ROOT)
    entry, out = initial / "entrada", initial / "comparacao"
    sterics = entry / "initial_sterics.csv"
    before = read(entry / "initial_sterics_summary.json")
    require(digest(sterics) == before["sterics_csv_sha256"] and
            before["threshold_angstrom_strictly_below"] == DISTANCE, "Initial sterics summary or threshold mismatch")
    initial_rows = {r["complex_id"]: r for r in read_csv(sterics)}
    require(len(initial_rows) == before["n_complexes"] == 50, "Incomplete initial distances")
    harmonized = factorial / "harmonized_observations.csv"
    require(digest(harmonized) == read(factorial / "harmonized_results.json")["observations_sha256"],
            "Harmonized factorial table differs from the archived summary")
    index = {(r["complex_id"], r["algorithm"], r["condition"]): r for r in read_csv(harmonized)}
    require(len(index) == 600, "Incomplete factorial steric evidence")
    pair_file = out / "paired_comparisons.csv"
    require(digest(pair_file) == read(out / "summary.json")["output_sha256"][pair_file.name], "Paired table changed")
    observations = []
    groups = defaultdict(list)
    for row in read_csv(pair_file):
        target = row["complex_id"]
        source = initial_rows[target]
        initial_minimum = float(source["min_interpartner_distance_angstrom"])
        initial_dockq = float(row["initial_dockq"])
        start = initial_dockq >= THRESHOLD and initial_minimum >= DISTANCE
        require(initial_dockq == float(source["initial_dockq"]) and math.isfinite(initial_minimum) and initial_minimum >= 0
                and int(initial_minimum < DISTANCE) == int(source["hard_clash_below_1_2"])
                and int(start) == int(source["acceptable_with_common_gate"]), f"Initial scores or sterics differ for {target}")
        minimum, published, path, sha = final_evidence(workspace, index, revision / "controle", row)
        valid = int(row["final_valid"])
        final = bool(valid and float(row["final_dockq"]) >= THRESHOLD and minimum >= DISTANCE)
        require(int(final) == published, f"Common-gate success differs from the published assessment: {path}")
        transition = { (True, True): "maintained", (False, True): "rescued",
                       (True, False): "lost", (False, False): "neither" }[start, final]
        result = {k: row[k] for k in ("complex_id", "method", "extent", "initialization")}
        result.update(initial_dockq=initial_dockq, initial_min_distance_angstrom=initial_minimum,
                      initial_hard_clash=int(initial_minimum < DISTANCE), initial_composite_success=int(start),
                      final_valid=valid, final_rejected=1-valid, final_dockq=row["final_dockq"],
                      final_candidate_min_distance_angstrom=minimum, final_hard_clash=int(minimum < DISTANCE),
                      final_composite_success=int(final), transition=transition, final_evidence_path=path,
                      final_evidence_sha256=sha)
        observations.append(result)
        groups[row["method"], row["extent"], row["initialization"]].append(result)
    require(len(observations) == 700 and len(groups) == 14, "Expected 700 pairs in 14 conditions")
    summaries = []
    for (method, extent, init), rows in groups.items():
        require(len(rows) == 50 and {r["complex_id"] for r in rows} == initial_rows.keys(), "Incomplete condition")
        counts = Counter(r["transition"] for r in rows)
        summary = {"method": method, "extent": extent, "initialization": init, "n_planned": len(rows),
                   "initial_composite_success": sum(r["initial_composite_success"] for r in rows),
                   "final_composite_success": sum(r["final_composite_success"] for r in rows),
                   **{k: counts[k] for k in ("maintained", "rescued", "lost", "neither")},
                   "final_valid": sum(r["final_valid"] for r in rows), "final_rejected": sum(r["final_rejected"] for r in rows),
                   "final_clashes": sum(r["final_hard_clash"] for r in rows),
                   "clashes_on_valid_finals": sum(r["final_valid"] * r["final_hard_clash"] for r in rows)}
        summaries.append(summary)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        tmp = Path(tmp)
        write_csv(tmp / "paired_steric_comparisons.csv", observations)
        write_csv(tmp / "steric_condition_summary.csv", summaries)
        for name in ("paired_steric_comparisons.csv", "steric_condition_summary.csv"):
            (tmp / name).replace(out / name)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    args = parser.parse_args()
    summaries = run(args.workspace)
    print(f"{sum(r['n_planned'] for r in summaries)} original winning candidates checked; "
          f"{sum(r['clashes_on_valid_finals'] for r in summaries)} clashes on returned poses.")

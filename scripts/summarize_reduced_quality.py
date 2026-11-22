"""Summarize completed paired runs; never tune or select poses by reference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = json.loads(args.comparison.read_text(encoding="utf-8"))
    assert len(rows) == 30 and all(r["status"] == "SUCCEEDED" for r in rows)
    before = {(r["complex_id"], r["job_id"]): r for r in json.loads(args.before.read_text(encoding="utf-8"))["runs"]}
    paired = {}
    operator_counts = {}
    unchanged_full = 0
    for row in rows:
        job = row["job"]
        work = args.comparison.parent / row["complex_id"] / job["job_id"]
        result = row["engine_result"]
        out = (work / result["engine_summary_path"]).parent
        expected_class = {"ag": "AGSearch", "pso": "PSOSearch", "hybrid": "ABCGAVGOSSearch"}[job["algorithm"]]
        events = [json.loads(line) for path in out.rglob("metrics.jsonl") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ops = Counter(e["engine_name"] for e in events if e.get("name") == "search.operator_completed")
        assert set(ops) == {expected_class}, (job, ops)
        assert row["pose_meta"]["final_hard_clash_gate"]["triggered"] is False
        assert row["pose_meta"]["docking_task"] == "local_refinement"
        assert row["fallback_executed"] is False
        operator_counts[row["complex_id"] + "/" + job["job_id"]] = dict(ops)
        key = (row["complex_id"], job["algorithm"], job["seed"])
        paired.setdefault(key, {})[job["protocol"]] = row
        if job["protocol"] == "full":
            assert row["pdb_sha256"] == before[(row["complex_id"], job["job_id"])]["pdb_sha256"], "FULL comparator changed"
            unchanged_full += 1
    pairs = []
    for key, arms in sorted(paired.items()):
        red, full = arms["reduced"], arms["full"]
        delta = red["dockq"] - full["dockq"]
        ratio = red["total_wall_seconds"] / full["total_wall_seconds"]
        pairs.append({"complex_id": key[0], "algorithm": key[1], "seed": key[2],
                      "dockq_reduced": red["dockq"], "dockq_full": full["dockq"], "delta_dockq": delta,
                      "quality_criterion_met": red["dockq"] >= .23 and delta >= -.05,
                      "reduced_total_seconds": red["total_wall_seconds"], "full_total_seconds": full["total_wall_seconds"],
                      "runtime_ratio": ratio, "reduced_faster": ratio < 1,
                      "reduced_objective_evaluations": red["actual_objective_evaluations"],
                      "full_objective_evaluations": full["actual_objective_evaluations"]})
    red_time = sum(p["reduced_total_seconds"] for p in pairs)
    full_time = sum(p["full_total_seconds"] for p in pairs)
    summary = {"runs": len(rows), "full_pdbs_identical_to_previous": unchanged_full,
               "quality_criterion": "DockQ_REDUCED >= 0.23 and DockQ_FULL - DockQ_REDUCED <= 0.05 (pre-existing project criterion)",
               "quality_pass_count": sum(p["quality_criterion_met"] for p in pairs),
               "reduced_faster_count": sum(p["reduced_faster"] for p in pairs),
               "dockq_mean_reduced": statistics.mean(p["dockq_reduced"] for p in pairs),
               "dockq_mean_full": statistics.mean(p["dockq_full"] for p in pairs),
               "reduced_total_seconds": red_time, "full_total_seconds": full_time,
               "aggregate_time_reduction_fraction": 1 - red_time / full_time,
               "median_runtime_ratio": statistics.median(p["runtime_ratio"] for p in pairs),
               "executed_operators": operator_counts, "pairs": pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"pairs", "executed_operators"}}, indent=2))


if __name__ == "__main__":
    main()

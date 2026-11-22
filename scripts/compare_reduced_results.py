"""Audit a repeated real batch against a preserved results ZIP (read-only inputs)."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compare(before_zip: Path, batch_dir: Path):
    before = {}
    with zipfile.ZipFile(before_zip) as archive:
        for name in archive.namelist():
            if not name.endswith(("/receipt.json", "/error.json")):
                continue
            record = json.loads(archive.read(name))
            job_id = record["job_id"] if "job_id" in record else record["job"]["job_id"]
            key = (record["complex_id"], job_id)
            if name.endswith("/error.json"):
                before[key] = {"status": record.get("engine_status", "FAILED")}
            else:
                result = record["engine_result"]
                pdb_name = name.removesuffix("receipt.json") + result["complex_pose_pdb_path"]
                before[key] = {"status": "SUCCEEDED", "pdb_sha256": digest(archive.read(pdb_name)),
                               "dockq": result["dockq"]}
    status = json.loads((batch_dir / "batch_status.json").read_text())
    assert status["state"] in {"COMPLETED", "COMPLETED_WITH_ERRORS"}, "Wait for all batch jobs to finish before final comparison"
    rows = []
    expected = {"ag": "AGSearch", "pso": "PSOSearch", "hybrid": "ABCGAVGOSSearch"}
    for key, task in sorted(status["tasks"].items()):
        directory = batch_dir / "targets" / task["complex_id"] / "jobs" / task["job_id"]
        row = {"complex_id": task["complex_id"], "job_id": task["job_id"],
               "before": before.get((task["complex_id"], task["job_id"]))}
        if task["state"] != "SUCCEEDED":
            error = json.loads((directory / "error.json").read_text())
            row.update(status=error.get("engine_status", "FAILED"),
                       failure_reason=error.get("failure_reason", error["message"]))
            rows.append(row)
            continue
        result = json.loads((directory / "receipt.json").read_text())["engine_result"]
        assert result["job_id"] == task["job_id"]
        summary = json.loads((directory / result["engine_summary_path"]).read_text())
        pose = json.loads((directory / result["pose_artifact_path"]).read_text())
        assert pose["engine_name"] == expected[result["algorithm"]]
        assert summary["search_space_mode"] == result["protocol"]
        gate = pose.get("meta", {}).get("final_hard_clash_gate", {})
        assert gate.get("triggered") is False, "A successful result must pass the clash gate"
        assert not summary.get("fallback_executed")
        assert not summary.get("fallback_to_full")
        events = [json.loads(line) for log in (directory / "engine").rglob("metrics.jsonl")
                  for line in log.read_text().splitlines() if line.strip()]
        engines = Counter(event["engine_name"] for event in events
                          if event.get("name") == "search.operator_completed")
        assert set(engines) == {expected[result["algorithm"]]}
        pdb_hash = digest((directory / result["complex_pose_pdb_path"]).read_bytes())
        wall = (datetime.fromisoformat(task["completed_at_utc"].replace("Z", "+00:00")) -
                datetime.fromisoformat(task["started_at_utc"].replace("Z", "+00:00"))).total_seconds()
        row.update(status="SUCCEEDED", algorithm=result["algorithm"], protocol=result["protocol"],
                   seed=result["seed"], generations=result["generations"], population=result["population"],
                   evaluation_budget=result["evaluation_budget"],
                   total_n_eval=summary.get("total_n_eval", summary.get("n_eval_total")),
                   executed_operators=dict(engines), pdb_sha256=pdb_hash,
                   full_pdb_unchanged=(pdb_hash == row["before"].get("pdb_sha256"))
                       if result["protocol"] == "full" and row["before"] else None,
                   dockq=result["dockq"], fnat=result["fnat"], irmsd=result["irmsd"], lrmsd=result["lrmsd"],
                   engine_reported_runtime_sec=result["runtime_sec"], wall_time_sec=wall,
                   minimum_distance=gate.get("min_distance"), gate_threshold=gate.get("threshold"),
                   anchor_mode=pose.get("meta", {}).get("search_anchor_mode"),
                   pose_generation=pose.get("generation"),
                   pose_path=str(directory / result["complex_pose_pdb_path"]))
        rows.append(row)
    assert not list(batch_dir.rglob("fallback_full"))
    return {"before_zip_sha256": digest(before_zip.read_bytes()), "batch_counts": status["counts"],
            "status_counts": dict(Counter(row["status"] for row in rows)), "runs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-zip", required=True, type=Path)
    parser.add_argument("--batch-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.before_zip, args.batch_dir.resolve())
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["status_counts"]))
    for row in result["runs"]:
        print(row["complex_id"], row["job_id"], row["status"], row.get("dockq"), row.get("pdb_sha256"))


if __name__ == "__main__":
    main()

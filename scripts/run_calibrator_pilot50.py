"""Run only the user-authorized 50-complex development pilot."""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
from docking_confidence.application.batch import run_batch
from docking_confidence.application.experimental_protocol import prepare_experimental_protocol
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import (
    Algorithm, BaseSearchConfig, DockingTask, FactorialExperimentConfig, ProtocolMode,
)
from docking_confidence.io.batch_discovery import discover_ppi4dock


def shard_report(report, count, index):
    """Balance input-size costs while keeping molecular aliases together."""
    groups = {}
    for target in report.targets:
        groups.setdefault(target.complex_fingerprint_sha256, []).append(target)
    costs = lambda members: sum(
        max(1, target.receptor_atom_count or 1) * max(1, target.ligand_atom_count or 1)
        for target in members
    )
    loads, assignments = [0] * count, [[] for _ in range(count)]
    for _, members in sorted(groups.items(), key=lambda item: (-costs(item[1]), item[0])):
        destination = min(range(count), key=lambda i: (loads[i], i))
        assignments[destination].extend(members)
        loads[destination] += costs(members)
    return replace(report, targets=tuple(sorted(assignments[index], key=lambda t: t.complex_id)))


def run_shards(args):
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    thread_settings = {key: "1" for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    )}
    environment.update(thread_settings)
    children, logs = [], []
    for index in range(args.shard_count):
        folder = root / f"shard_{index+1:02d}"
        handle = (root / f"shard_{index+1:02d}.log").open("a", encoding="utf-8")
        logs.append(handle)
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()),
             "--input-dir", args.input_dir, "--output-dir", str(folder),
             "--shard-count", str(args.shard_count), "--shard-index", str(index)],
            stdout=handle, stderr=subprocess.STDOUT, env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        children.append((process, folder))
    print(json.dumps({"phase": "parallel_pilot_started", "shards": args.shard_count,
                      "thread_settings": thread_settings, "output": str(root)}), flush=True)
    while True:
        workers = [{"pid": child.pid, "batch": str(folder), "exit_code": child.poll()}
                   for child, folder in children]
        (root / "pilot_workers.json").write_text(
            json.dumps({"thread_settings": thread_settings, "workers": workers}, indent=2),
            encoding="utf-8",
        )
        if all(item["exit_code"] is not None for item in workers):
            break
        time.sleep(5)
    for handle in logs:
        handle.close()
    failed = [item for item in workers if item["exit_code"] != 0]
    if failed:
        raise RuntimeError(f"pilot workers failed; inspect their persisted logs: {failed}")
    print(json.dumps({"phase": "all_pilot_shards_completed", "workers": workers}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--all-shards", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or args.shard_count > 8:
        parser.error("shard-count must be between 1 and 8")
    if args.all_shards:
        return run_shards(args)
    if args.shard_index is not None and not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be inside shard-count")
    engine = DockingPPLocalEngine()
    base = BaseSearchConfig(
        generations=100, population=100, top_pockets=1,
        docking_task=DockingTask.LOCAL_REFINEMENT,
        experiment=FactorialExperimentConfig(
            evaluation_budget=7600, reduced_search_scale=0.5,
            max_translation=5.0, max_rotation_degrees=25.0,
            execution_order_seed=20260925,
            pose_source=(
                "Exploratory calibrator-development pilot restricted to the 50 "
                "user-supplied complexes. Starting poses are the supplied partner "
                "models; their original preparation and relationship to the "
                "experimental complex have not been independently verified."
            ),
        ),
    )
    jobs = build_execution_plan(
        algorithms=[Algorithm.AG, Algorithm.PSO, Algorithm.HYBRID],
        protocol_mode=ProtocolMode.FACTORIAL, base_config=base, seeds=[1],
    )
    report = discover_ppi4dock(args.input_dir, validate_contents=True)
    if args.shard_index is not None:
        report = shard_report(report, args.shard_count, args.shard_index)
    prepared = prepare_experimental_protocol(
        args.input_dir, args.output_dir, jobs=jobs, engine=engine,
        analysis_role="exploratory", report=report,
    )
    print(json.dumps({
        "phase": "protocol_frozen", "protocol": prepared["path"],
        "target_count": len(prepared["protocol"]["plan"]["targets"]),
        "jobs_per_target": len(jobs), "budget_per_job": 7600,
    }), flush=True)
    result = run_batch(
        args.input_dir, args.output_dir, jobs=jobs, engine=engine,
        eligible_only=True, experimental_protocol=prepared["path"], report=report,
    )
    print(json.dumps({
        "phase": "completed", "planned": result.planned_jobs, "succeeded": result.succeeded,
        "failed": result.failed, "status_path": str(result.status_path),
    }), flush=True)


if __name__ == "__main__":
    main()

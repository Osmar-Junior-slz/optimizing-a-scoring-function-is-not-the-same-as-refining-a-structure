"""Run or resume the prepared factorial shards without changing the docking engine.

The scientific and per-shard protocols are prepared separately before starting.
Runs already completed and genuine no-pose outcomes are retained across restarts.
Transient and interrupted attempts keep their files; only incomplete shards are
started. The ordinary batch receipt, error, heartbeat and status files are the
record of each search.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docking_confidence.application.batch import _BatchFileLock, _atomic_write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def available_mib():
    """Available physical memory includes reclaimable caches on Windows."""
    if os.name != "nt":
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**20

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
            (field, ctypes.c_uint64) for field in (
                "total", "available", "page", "free_page", "virtual", "free_virtual", "free_extended"
            )
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return status.available / 2**20


def shard_paths(shard, root):
    paths = []
    for key in ("input_dir", "batch_dir", "experimental_protocol_path"):
        value = Path(shard[key])
        path = (value if value.is_absolute() else root / value).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"Path outside the registered study: {path}")
        paths.append(path)
    inputs, batch, protocol = paths
    if protocol != batch / "experimental_protocol.json":
        raise ValueError("Prepared protocol must belong to the named batch")
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != shard["protocol_file_sha256"]:
        raise ValueError(f"Experimental protocol differs from its commitment: {protocol}")
    return inputs, batch, protocol


def run_shard(shard, root):
    from docking_confidence.adapters.dockingpp_archive_engine import DockingPPReducedError
    from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
    from docking_confidence.application.batch import run_batch
    from docking_confidence.application.experimental_protocol import load_experimental_protocol
    from docking_confidence.domain.execution import Algorithm, DockingTask, ExecutionJob, ProtocolMode, ProtocolProfile

    class PreserveNoPose(DockingPPLocalEngine):
        def run(self, **kwargs):
            task = Path(kwargs["work_dir"]).parent
            err = task / "error.json"
            if err.is_file():
                recorded = read(err)
                archive = task / f"prior_attempt_{recorded['attempt']}_error.json"
                if not archive.exists():
                    shutil.copyfile(err, archive)
                if (recorded.get("engine_status") == "REDUCED_FAILED" and
                        recorded.get("failure_reason") == "no_valid_reduced_pose"):
                    raise DockingPPReducedError(recorded["engine_status"], recorded["failure_reason"])
            return super().run(**kwargs)

    inputs, batch, protocol = shard_paths(shard, root)
    settings = load_experimental_protocol(protocol)["plan"]
    jobs = []
    for saved in settings["jobs"]:
        profile = dict(saved["profile"])
        profile["mode"] = ProtocolMode(profile["mode"])
        profile["docking_task"] = DockingTask(profile["docking_task"])
        jobs.append(ExecutionJob(job_id=saved["job_id"], algorithm=Algorithm(saved["algorithm"]),
                                 protocol=ProtocolMode(saved["protocol"]), seed=saved["seed"],
                                 profile=ProtocolProfile(**profile)))
    result = run_batch(inputs, batch, jobs=jobs, engine=PreserveNoPose(timeout_seconds=600),
                       experimental_protocol=protocol)
    if result.state not in {"COMPLETED", "COMPLETED_WITH_ERRORS"}:
        raise RuntimeError(f"Shard did not finish: {result.state}")
    return {"state": result.state, "succeeded": result.succeeded, "failed": result.failed}


def completed(batch):
    path = batch / "batch_status.json"
    if not path.exists():
        return False
    status = read(path)
    if status["state"] not in {"COMPLETED", "COMPLETED_WITH_ERRORS"}:
        return False
    for task in status["tasks"].values():
        if task["state"] == "FAILED":
            error = read(batch / task["last_error"]["path"])
            if (error.get("engine_status"), error.get("failure_reason")) != (
                    "REDUCED_FAILED", "no_valid_reduced_pose"):
                return False
    return True


def supervise(plan, shards, max_workers):
    root = plan.parent
    logs = root / "worker_logs"
    logs.mkdir(exist_ok=True)
    pending = []
    for index, shard in enumerate(shards):
        _, batch, _ = shard_paths(shard, root)
        if not completed(batch):
            # A supervisor can stop while an existing nested process is finishing.
            with _BatchFileLock(batch / "batch.lock"):
                pass
            pending.append(index)
    active = {}
    failures = {}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    next_launch = 0.0
    while pending or active:
        for index, child in list(active.items()):
            rc = child.poll()
            if rc is not None:
                if rc:
                    failures[str(index + 1)] = rc
                del active[index]
        free = available_mib()
        if (pending and len(active) < max_workers and time.monotonic() >= next_launch
                and (not active or free >= 550)):
            index = pending.pop(0)
            command = [sys.executable, "-B", str(Path(__file__).resolve()),
                       "--plan", str(plan), "--worker-index", str(index)]
            with (logs / f"shard_{index + 1:02d}.log").open("ab", buffering=0) as log:
                active[index] = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            next_launch = time.monotonic() + 12
            print(f"Started shard {index+1:02d}, pid {active[index].pid}; available {free:.0f} MiB",
                  flush=True)
        _atomic_write_json(logs / "supervisor.json", {
            "timestamp": time.time(), "available_memory_mib": round(free),
            "active_shards": {str(i+1): p.pid for i, p in active.items()},
            "waiting_shards": [i+1 for i in pending], "worker_errors": failures,
        })
        if pending or active:
            time.sleep(5)
    if failures:
        raise RuntimeError(f"Workers need inspection or restart: {failures}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan = args.plan.resolve(strict=True)
    manifest = read(plan)
    for name in ("selection", "analysis_plan"):
        file = Path(manifest[name + "_path"])
        if hashlib.sha256(file.read_bytes()).hexdigest() != manifest[name + "_sha256"]:
            raise ValueError(f"Registered {name} has changed")
    shards = manifest["shards"]
    if not 1 <= args.max_workers <= 8:
        parser.error("--max-workers must be between 1 and 8")
    if args.worker_index is None:
        with _BatchFileLock(plan.parent / ".factorial_supervisor.lock"):
            supervise(plan, shards, args.max_workers)
    else:
        print(json.dumps(run_shard(shards[args.worker_index], plan.parent)), flush=True)


if __name__ == "__main__":
    main()

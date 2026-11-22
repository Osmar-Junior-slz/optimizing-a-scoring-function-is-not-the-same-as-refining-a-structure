"""Read-only reconstruction of the batch dashboard from durable files."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import os
from pathlib import Path
import re
import time

from docking_confidence.application.batch import (
    MANIFEST_FILENAME, load_batch_status, _load_json_object, _validate_manifest,
)
from docking_confidence.application.batch_telemetry import (
    HEARTBEAT_FILENAME, MAX_EVENTS, STALE_SECONDS, read_json, timestamp,
)
from docking_confidence.domain.execution import FIXED_EVALUATION_BUDGET

TERMINAL_STATES = {"COMPLETED", "COMPLETED_WITH_ERRORS", "CANCELLED", "INTERRUPTED"}


def plan_breakdown(manifest: dict) -> dict:
    plan = manifest.get("plan", {})
    targets, jobs = plan.get("targets", []), plan.get("jobs", [])
    algorithms = sorted({job["algorithm"] for job in jobs})
    protocols = sorted({job["protocol"] for job in jobs})
    conditions = sorted({f"{job['profile']['search_extent']}/{job['profile']['initialization']}" for job in jobs if job.get("profile", {}).get("initialization") is not None})
    seeds = sorted({job["seed"] for job in jobs})
    ids = [target["complex_id"] for target in targets]
    combinations = {(job["algorithm"], f"{job['profile']['search_extent']}/{job['profile']['initialization']}" if conditions else job["protocol"], job["seed"]) for job in jobs}
    arm_count = len(conditions) if conditions else len(protocols)
    product = len(targets) * len(algorithms) * arm_count * len(seeds)
    root = str(plan.get("input_root", ""))
    folder = root.replace("\\", "/").rstrip("/").split("/")[-1]
    match = re.fullmatch(r"(\d+)[ _-]*prote[ií]nas", folder, re.IGNORECASE)
    named_count = int(match[1]) if match else None
    return {
        "complexes": len(targets), "eligible_complexes": sum(t.get("eligible") is True for t in targets),
        "algorithms": algorithms, "protocols": protocols, "seeds": seeds, "conditions": conditions,
        "jobs_per_complex": len(jobs), "planned_jobs": len(targets) * len(jobs),
        "cartesian_jobs": product,
        "cartesian": len(combinations) == len(jobs) == len(algorithms) * arm_count * len(seeds),
        "duplicate_complex_ids": len(ids) - len(set(ids)),
        "unique_pdb_ids": len({value.split("_")[0].lower() for value in ids}),
        "excluded_complexes": len(plan.get("excluded_targets", [])),
        "input_root": root, "folder_label_count": named_count,
        "folder_count_difference": None if named_count is None else len(targets) - named_count,
        "complex_ids": ids,
    }


@lru_cache(maxsize=16)
def _manifest_snapshot(path: str, mtime_ns: int, size: int) -> dict:
    manifest = _load_json_object(Path(path))
    _validate_manifest(manifest)
    return manifest


def process_alive(pid, started_at_utc=None) -> bool | None:
    """Query process existence, never send a termination signal.

    Access denied is unknown. On Windows also reject a PID recycled after the
    recorded worker start. Checking creation time cannot prove docking advances.
    """
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
            handle = kernel.OpenProcess(0x1000, False, pid)
            if not handle:
                return False if ctypes.get_last_error() == 87 else None
            try:
                code = wintypes.DWORD()
                if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return None
                if code.value != 259:
                    return False
                expected = timestamp(started_at_utc)
                if expected is not None:
                    times = [wintypes.FILETIME() for _ in range(4)]
                    if kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                        created = ((times[0].dwHighDateTime << 32) + times[0].dwLowDateTime) / 1e7 - 11644473600
                        if created > expected + 2:
                            return False
                return True
            finally:
                kernel.CloseHandle(handle)
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (OSError, ValueError, OverflowError):
        return None


def worker_health(state: str, heartbeat: dict, *, now: float, alive: bool | None) -> dict:
    updated = timestamp(heartbeat.get("updated_at_utc"))
    age = max(0.0, now - updated) if updated is not None else None
    if state in TERMINAL_STATES:
        kind = "finished"
    elif alive is False:
        kind = "stopped"
    elif age is not None and age <= STALE_SECONDS:
        kind = "healthy"
    else:
        kind = "stale" if age is not None else "unknown"
    return {"kind": kind, "age_seconds": age, "process_alive": alive}


def task_duration(task: dict) -> float | None:
    value = task.get("duration_seconds")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value < float("inf"):
        return float(value)
    start, end = timestamp(task.get("started_at_utc")), timestamp(task.get("completed_at_utc"))
    return end - start if start is not None and end is not None and end > start else None


def estimate_remaining(tasks: list[dict], *, now: float) -> dict | None:
    """Estimate serial work from measured successes, marking sparse/fallback samples.

    Failures and calibrator skips are not representative docking durations. Before
    every condition has samples, the overall mean is an explicitly preliminary
    fallback. An overdue active job suspends the displayed prediction.
    """
    def group_key(task):
        profile = task.get("profile", {})
        return (task.get("algorithm"), task.get("protocol"),
                profile.get("search_extent", task.get("search_extent")),
                profile.get("initialization", task.get("initialization")))

    samples = defaultdict(list)
    remaining = [task for task in tasks if task["state"] in {"PENDING", "RUNNING"}]
    for task in tasks:
        duration = task_duration(task)
        if task["state"] == "SUCCEEDED" and duration is not None:
            samples[group_key(task)].append(duration)
    count = sum(len(group) for group in samples.values())
    if not remaining or not count:
        return None
    overall_mean = sum(sum(group) for group in samples.values()) / count
    seconds = 0.0
    preliminary = count < 5
    fallback_jobs = overdue_jobs = 0
    for task in remaining:
        group = samples[group_key(task)]
        preliminary = preliminary or len(group) < 3
        fallback_jobs += not group
        mean = sum(group) / len(group) if group else overall_mean
        started = timestamp(task.get("started_at_utc"))
        elapsed = max(0.0, now - started) if task["state"] == "RUNNING" and started is not None else 0
        overdue_jobs += task["state"] == "RUNNING" and elapsed >= mean
        seconds += max(0.0, mean - elapsed)
    return {"seconds": seconds, "sample_count": count, "preliminary": preliminary,
            "fallback_jobs": fallback_jobs, "overdue_jobs": overdue_jobs,
            "finish_at_utc": datetime.fromtimestamp(now + seconds, timezone.utc).isoformat()
            if not overdue_jobs else None}


def load_batch_monitor(output_root: str | Path, *, now: float | None = None, probe=process_alive) -> dict | None:
    """Only reads persisted state and OS process metadata. Never launches/resumes."""
    root = Path(output_root).expanduser().resolve(strict=False)
    status = load_batch_status(root)
    if status is None:
        return None
    now = time.time() if now is None else now
    path = root / MANIFEST_FILENAME
    stat = path.stat()
    manifest = _manifest_snapshot(str(path), stat.st_mtime_ns, stat.st_size)
    if status.get("batch_id") != manifest["batch_id"]:
        raise ValueError("Estado e manifesto pertencem a lotes diferentes")
    plan = plan_breakdown(manifest)
    jobs = {job["job_id"]: job for job in manifest["plan"]["jobs"]}
    tasks = [{**jobs.get(t["job_id"], {}), **t} for t in status.get("tasks", {}).values()]
    counts = Counter(task["state"] for task in tasks)
    processed = counts["SUCCEEDED"] + counts["FAILED"] + counts["SKIPPED_CALIBRATOR"]
    planned = plan["planned_jobs"]
    heartbeat = read_json(root / HEARTBEAT_FILENAME)
    if (heartbeat.get("batch_id") != status["batch_id"]
        or (status.get("worker_id") and heartbeat.get("worker_id") != status["worker_id"])):
        heartbeat = {}
    owner = read_json(root / "batch.lock")
    pid = heartbeat.get("worker_pid") or status.get("worker_pid") or owner.get("pid")
    alive = probe(pid, heartbeat.get("worker_started_at_utc") or owner.get("acquired_at_utc"))
    state = status["state"]
    if state == "RUNNING" and heartbeat.get("worker_state") in TERMINAL_STATES:
        state = heartbeat["worker_state"]
    health = worker_health(state, heartbeat, now=now, alive=alive)
    worker_started = timestamp(heartbeat.get("worker_started_at_utc") or status.get("created_at_utc"))
    worker_finished = timestamp(status.get("completed_at_utc") or heartbeat.get("updated_at_utc") or status.get("updated_at_utc"))
    elapsed_end = worker_finished if health["kind"] in {"finished", "stopped"} and worker_finished is not None else now
    active = []
    if state not in TERMINAL_STATES:
        for task in tasks:
            if task["state"] != "RUNNING":
                continue
            task = dict(task)
            if (heartbeat.get("current_job_id") == task["job_id"]
                and heartbeat.get("complex_id") == task["complex_id"]
                and heartbeat.get("started_at_utc") == task.get("started_at_utc")
                and heartbeat.get("attempt_count") == task.get("attempt_count")):
                task.update(heartbeat)
            started = timestamp(task.get("started_at_utc"))
            task["elapsed_seconds"] = max(0, now - started) if started else None
            fixed = task.get("profile", {}).get("budget_policy") == FIXED_EVALUATION_BUDGET
            task["progress_basis"] = "evaluations" if fixed else "generations"
            if fixed:
                current, total = task.get("evaluations_current"), task["profile"]["evaluation_budget"]
            else:
                current, total = task.get("generation_current"), task.get("generation_total")
            task["progress_fraction"] = (
                current / total if type(current) is int and type(total) is int
                and total > 0 and 0 <= current <= total else None
            )
            active.append(task)
    successes = [t for t in tasks if t["state"] == "SUCCEEDED" and timestamp(t.get("completed_at_utc")) is not None]
    last = max(successes, key=lambda t: timestamp(t["completed_at_utc"])) if successes else None
    events = heartbeat.get("recent_events", [])[-MAX_EVENTS:]
    if not events:
        events = [{"at_utc": task.get("completed_at_utc"), "message": f"{task['complex_id']} {task['job_id']} {task['state']}"}
                  for task in sorted(tasks, key=lambda t: t.get("completed_at_utc") or "")[-MAX_EVENTS:]
                  if task.get("completed_at_utc")]
    return {
        "state": state, "counts": {"planned": planned, "succeeded": counts["SUCCEEDED"],
            "failed": counts["FAILED"], "pending": counts["PENDING"],
            "skipped_calibrator": counts["SKIPPED_CALIBRATOR"],
            "running": counts["RUNNING"] if health["kind"] not in {"stopped", "finished"} else 0},
        "orphaned_running": counts["RUNNING"] if health["kind"] in {"stopped", "finished"} else 0,
        "processed": processed, "global_fraction": processed / planned if planned else 0.0,
        "plan": plan, "health": health, "worker_pid": pid, "heartbeat": heartbeat,
        "active_jobs": active, "last_completed": last, "events": events,
        "status_updated_at_utc": status.get("updated_at_utc"),
        "elapsed_seconds": max(0.0, elapsed_end - worker_started) if worker_started is not None else None,
        "completed_at_utc": status.get("completed_at_utc"),
        "failed_tasks": [t for t in tasks if t["state"] == "FAILED"][-20:],
        "confidence_decisions": [t["confidence_decision"] for t in tasks
                                 if t.get("protocol") == "full" and isinstance(t.get("confidence_decision"), dict)],
        "eta": estimate_remaining(tasks, now=now) if state == "RUNNING" and health["kind"] == "healthy" else None,
        "count_mismatch": len(tasks) != planned or status.get("counts", {}).get("planned") != planned,
    }

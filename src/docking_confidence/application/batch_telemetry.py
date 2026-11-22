"""Best-effort observation of a batch; never controls the scientific engine."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

from docking_confidence.application.batch import _atomic_write_json
from docking_confidence.domain.execution import FIXED_EVALUATION_BUDGET

HEARTBEAT_FILENAME = "batch_heartbeat.json"
HEARTBEAT_SECONDS = 5.0
STALE_SECONDS = 60.0
MAX_EVENTS = 40
MAX_METRIC_BYTES = 512 * 1024
_LOG = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def timestamp(value: Any) -> float | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (ValueError, TypeError, OverflowError):
        return None


def read_json(path: Path) -> dict:
    # Windows/cloud sync can briefly deny opening a file during atomic replace.
    for attempt in range(3):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            if attempt < 2:
                time.sleep(0.01)
    return {}


def nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0 or int(value) != value:
        return None
    return int(value)


class MetricsReader:
    """Incremental JSONL reader. n_eval is per generation, NOT cumulative.

    Counters are scoped to one metrics file/pocket. Root aggregate rewrites and
    repeated rows cannot double count. Partial trailing lines wait for completion.
    No guessed generation from step, elapsed time, population, or random numbers.
    """

    def __init__(self, *, fixed_evaluations: bool = False) -> None:
        self.files: dict[Path, dict] = {}
        self.fixed_evaluations = fixed_evaluations

    def read(self, output: Path) -> dict:
        paths = list(output.glob("metrics.jsonl")) + list(output.glob("*/metrics.jsonl"))
        latest: tuple[int, dict] | None = None
        for path in paths:
            stat = path.stat()
            state = self.files.setdefault(path, {"offset": 0, "evals": {}, "progress": {}})
            if stat.st_size < state["offset"]:
                state.update(offset=0, evals={}, progress={})
            with path.open("rb") as handle:
                handle.seek(state["offset"])
                data = handle.read(MAX_METRIC_BYTES)
            end = data.rfind(b"\n") + 1
            state["offset"] += end
            for line in data[:end].splitlines():
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                if not isinstance(row, dict):
                    continue
                generation = nonnegative_int(row.get("generation"))
                total = nonnegative_int(row.get("total_generations"))
                pocket = nonnegative_int(row.get("pocket_index"))
                if generation is None or (not self.fixed_evaluations and (total is None or total == 0 or generation > total)):
                    continue
                value = nonnegative_int(row.get("value"))
                if row.get("name") == "n_eval" and value is not None:
                    state["evals"][(pocket, generation)] = value
                state["progress"] = {
                    "generation_current": generation,
                    "generation_total": None if self.fixed_evaluations else total,
                    "pocket_index": pocket,
                    "metrics_path": str(path),
                    "metrics_updated_at_utc": datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc
                    ).isoformat(),
                }
            progress = dict(state["progress"])
            if progress:
                pocket = progress["pocket_index"]
                evaluations = [v for (p, _), v in state["evals"].items() if p == pocket]
                progress["evaluations_current"] = sum(evaluations) if evaluations else None
                progress["metrics_caught_up"] = state["offset"] >= stat.st_size
                # Leaf pocket logs are preferred if their mtime is the same.
                if latest is None or stat.st_mtime_ns >= latest[0]:
                    latest = (stat.st_mtime_ns, progress)
        return latest[1] if latest else {}


class WorkerHeartbeat:
    """Owned by run_batch under its existing OS lock, independent of Streamlit.

    One small atomic snapshot contains the heartbeat and a bounded event ring.
    All observation/I/O failures are isolated from engine execution and receipts.
    """

    def __init__(self, root: Path, batch_id: str, *, interval: float = HEARTBEAT_SECONDS):
        self.root = root
        self.interval = interval
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reader = MetricsReader()
        self._work_dir: Path | None = None
        self._old_runs: set[Path] = set()
        self._output: Path | None = None
        self._last_progress: tuple | None = None
        previous = read_json(root / HEARTBEAT_FILENAME)
        self.events = deque(
            previous.get("recent_events", []) if previous.get("batch_id") == batch_id else [],
            maxlen=MAX_EVENTS,
        )
        self.data: dict = {
            "schema_version": "batch_heartbeat_v1", "batch_id": batch_id,
            "worker_id": f"{os.getpid()}-{time.time_ns()}", "worker_pid": os.getpid(),
            "worker_started_at_utc": utc_now(), "worker_state": "RUNNING",
            "current_job_id": None, "complex_id": None, "algorithm": None,
            "protocol": None, "seed": None, "generation_current": None,
            "generation_total": None, "evaluations_current": None,
            "evaluations_total": None, "phase": "preparando lote",
        }

    def _event(self, kind: str, message: str) -> None:
        self.events.append({"at_utc": utc_now(), "kind": kind, "message": message})

    def _persist(self) -> None:
        self.data["updated_at_utc"] = utc_now()
        _atomic_write_json(self.root / HEARTBEAT_FILENAME, {
            **self.data, "recent_events": list(self.events),
        })

    def _safe(self, action, *args) -> None:
        try:
            with self._lock:
                action(*args)
        except Exception:
            _LOG.warning("Batch telemetry unavailable; docking continues", exc_info=True)

    def __enter__(self):
        self._safe(self._persist)
        self._thread = threading.Thread(target=self._loop, name="batch-heartbeat", daemon=True)
        self._safe(self._thread.start)
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.pulse()

    def pulse(self) -> None:
        self._safe(self._pulse)

    def _pulse(self) -> None:
        if self._work_dir is not None:
            if self._output is None:
                runs = set((self._work_dir / "dockingpp_runs").glob("*")) - self._old_runs
                if runs:
                    self._output = max(runs, key=lambda p: p.stat().st_mtime_ns) / "output"
            if self._output is not None:
                try:
                    self.data.update(self._reader.read(self._output))
                    self.data.pop("monitor_warning", None)
                except (OSError, ValueError) as exc:
                    # A temporary metrics read failure must not stop the pulse.
                    self.data["monitor_warning"] = f"Métricas temporariamente indisponíveis: {exc}"
        signature = tuple(self.data.get(key) for key in (
            "metrics_path", "pocket_index", "generation_current", "evaluations_current"
        ))
        label = " ".join([str(self.data.get("complex_id") or ""),
                          str(self.data.get("algorithm") or "").upper(),
                          str(self.data.get("protocol") or "").upper()])
        if signature != self._last_progress and self.data.get("generation_current") is not None:
            self.data["phase"] = "busca / geração observada"
            self.data["progress_updated_at_utc"] = utc_now()
            observed = f"{label} geração {self.data['generation_current']}"
            if self._reader.fixed_evaluations:
                current = self.data.get("evaluations_current")
                if current is not None:
                    observed += f" — {current}/{self.data['evaluations_total']} avaliações efetivas"
            else:
                observed += f"/{self.data['generation_total']}"
            self._event("generation", observed)
            self._last_progress = signature
        else:
            self._event("heartbeat", "heartbeat" + (f" — {label}" if label.strip() else ""))
        self._persist()

    def start_job(self, task: dict, job: dict, work_dir: Path) -> None:
        self._safe(self._start_job, task, job, work_dir)

    def _start_job(self, task: dict, job: dict, work_dir: Path) -> None:
        self._work_dir = work_dir
        self._old_runs = set((work_dir / "dockingpp_runs").glob("*"))
        self._output = None
        fixed_evaluations = job["profile"].get("budget_policy") == FIXED_EVALUATION_BUDGET
        self._reader = MetricsReader(fixed_evaluations=fixed_evaluations)
        self._last_progress = None
        self.data.update(
            current_job_id=job["job_id"], complex_id=task["complex_id"],
            algorithm=job["algorithm"], protocol=job["protocol"], seed=job["seed"],
            started_at_utc=task["started_at_utc"], attempt_count=task["attempt_count"],
            generation_current=None, generation_total=None if fixed_evaluations else job["profile"]["generations"],
            evaluations_current=None, evaluations_total=job["profile"]["evaluation_budget"],
            pocket_index=None, metrics_path=None, metrics_updated_at_utc=None,
            progress_updated_at_utc=None, metrics_caught_up=True,
            phase="preparação / aguardando métricas do motor",
        )
        self._event("job_started", f"{task['complex_id']} {job['algorithm']} {job['protocol'].upper()} iniciado")
        self._persist()

    def end_job(self, task: dict) -> None:
        self._safe(self._end_job, task)

    def _end_job(self, task: dict) -> None:
        self._pulse()
        self._event("job_completed", f"{task['complex_id']} {task['job_id']} {task['state']}")
        self.data.update(current_job_id=None, phase="entre jobs")
        self._work_dir = None
        self._persist()

    def finish(self, state: str) -> None:
        self._safe(self._finish, state)

    def _finish(self, state: str) -> None:
        self.data.update(worker_state=state, current_job_id=None, phase="worker finalizado")
        self._event("worker_finished", f"worker {state}")
        self._persist()

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=self.interval + 1)
        if exc_type is not None:
            self.finish("INTERRUPTED")

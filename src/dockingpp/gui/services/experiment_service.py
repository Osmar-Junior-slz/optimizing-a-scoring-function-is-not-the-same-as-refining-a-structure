"""Experiment folder management for protein-centric runs."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENTS_ROOT = Path("runs_experimentos")


def sanitize_protein_name(name: str) -> str:
    """Return a filesystem-safe protein/complex identifier."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "proteina_sem_nome"


def create_experiment_layout(
    protein_name: str,
    *,
    root: str | Path = DEFAULT_EXPERIMENTS_ROOT,
    now: datetime | None = None,
) -> dict[str, Path]:
    """Create the standard experiment folder layout."""

    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    protein_id = sanitize_protein_name(protein_name)
    root_path = Path(root)
    experiment_dir = root_path / protein_id / timestamp
    inputs_dir = experiment_dir / "inputs"
    results_dir = experiment_dir / "results"
    analysis_dir = experiment_dir / "analysis"
    for path in (inputs_dir, results_dir, analysis_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": root_path,
        "protein_dir": root_path / protein_id,
        "experiment_dir": experiment_dir,
        "inputs_dir": inputs_dir,
        "results_dir": results_dir,
        "analysis_dir": analysis_dir,
    }


def update_experiment_index(
    *,
    protein_name: str,
    experiment_dir: Path,
    mode: str,
    status: str,
    root: str | Path = DEFAULT_EXPERIMENTS_ROOT,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append/update a lightweight index entry for dashboard navigation."""

    root_path = Path(root)
    index_path = root_path / "index.json"
    root_path.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    except json.JSONDecodeError:
        payload = []
    if not isinstance(payload, list):
        payload = []

    protein_id = sanitize_protein_name(protein_name)
    entry = {
        "protein": protein_id,
        "run_id": experiment_dir.name,
        "path": str(experiment_dir).replace("\\", "/"),
        "mode": mode,
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        entry.update(extra)

    payload = [
        item
        for item in payload
        if not (
            isinstance(item, dict)
            and item.get("protein") == protein_id
            and item.get("run_id") == experiment_dir.name
        )
    ]
    payload.append(entry)
    index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

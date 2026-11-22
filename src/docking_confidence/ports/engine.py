from __future__ import annotations

from pathlib import Path
from typing import Protocol

from docking_confidence.domain.execution import DockingRunResult, ExecutionJob


class DockingEngine(Protocol):
    name: str
    scientific: bool

    def run(
        self,
        *,
        job: ExecutionJob,
        protein_path: Path,
        peptide_path: Path,
        work_dir: Path | None = None,
    ) -> DockingRunResult:
        ...

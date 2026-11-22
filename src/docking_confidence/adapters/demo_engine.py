from __future__ import annotations

import hashlib
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from docking_confidence.domain.execution import DockingRunResult, ExecutionJob, ProtocolMode
from docking_confidence.io.pdb import count_atoms, validate_pdb


class DemoDockingEngine:
    """Deterministic NON-SCIENTIFIC engine used only to test UI and orchestration.

    It never claims to perform molecular docking. A real docking engine must implement
    the DockingEngine port before scientific experiments are run.
    """

    name = "demo_non_scientific"
    scientific = False
    supports_artifact_work_dir = True

    @staticmethod
    def _rng(job: ExecutionJob, protein_atoms: int, peptide_atoms: int) -> np.random.Generator:
        raw = f"{job.job_id}|{protein_atoms}|{peptide_atoms}".encode("utf-8")
        digest = hashlib.sha256(raw).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**32 - 1)
        return np.random.default_rng(seed)

    def run(
        self,
        *,
        job: ExecutionJob,
        protein_path: Path,
        peptide_path: Path,
        work_dir: Path | None = None,
    ) -> DockingRunResult:
        _ = work_dir
        validate_pdb(protein_path)
        validate_pdb(peptide_path)
        started = perf_counter()
        protein_atoms = count_atoms(protein_path)
        peptide_atoms = count_atoms(peptide_path)
        rng = self._rng(job, protein_atoms, peptide_atoms)

        algo_bonus = {"ag": 0.00, "pso": 0.025, "hybrid": 0.04}[job.algorithm.value]
        size_term = min(0.08, math.log1p(peptide_atoms) / 100.0)
        full_quality = float(np.clip(0.42 + algo_bonus + size_term + rng.normal(0, 0.035), 0, 1))

        if job.protocol == ProtocolMode.REDUCED:
            quality_loss = float(np.clip(rng.normal(0.025, 0.025), -0.02, 0.12))
            dockq = float(np.clip(full_quality - quality_loss, 0, 1))
            scan_gap = float(rng.normal(0.35 - quality_loss, 0.08))
            viable = float(np.clip(rng.normal(0.72 - quality_loss, 0.08), 0, 1))
            dispersion = float(abs(rng.normal(0.30 + quality_loss, 0.08)))
            uncertainty = float(max(0, rng.poisson(1.5 + 10 * quality_loss)))
            energy = float(rng.normal(-4.5 + quality_loss, 0.45))
        else:
            dockq = full_quality
            scan_gap = viable = dispersion = uncertainty = energy = None

        # Simulated runtime is based on budget; wall time here is not meaningful.
        budget_ratio = job.profile.evaluation_budget / 10_100.0
        pocket_ratio = job.profile.top_pockets / 32.0
        runtime = float(max(1.0, 220.0 * budget_ratio * (0.92 + 0.08 * pocket_ratio) + rng.normal(0, 6)))
        _ = perf_counter() - started

        return DockingRunResult(
            job_id=job.job_id,
            algorithm=job.algorithm.value,
            protocol=job.protocol.value,
            seed=job.seed,
            generations=job.profile.generations,
            population=job.profile.population,
            top_pockets=job.profile.top_pockets,
            evaluation_budget=job.profile.evaluation_budget,
            runtime_sec=runtime,
            dockq=dockq,
            scan_gap_norm=scan_gap,
            viable_fraction_mean=viable,
            elite_dispersion=dispersion,
            uncertainty_count=uncertainty,
            energy_q20_norm=energy,
            engine=self.name,
            scientific=self.scientific,
        )

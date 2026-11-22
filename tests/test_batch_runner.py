from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import pytest

from docking_confidence.application.batch import (
    BatchError,
    BatchPlanConflictError,
    clear_batch_cancel_request,
    load_batch_status,
    request_batch_cancel,
    run_batch,
)
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import (
    Algorithm,
    BaseSearchConfig,
    DockingTask,
    DockingRunResult,
    ExecutionJob,
    ProtocolMode,
)
from docking_confidence.domain.mobile_partner import DEFAULT_MOBILE_PARTNER_DOMAIN


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class _Target:
    complex_id: str
    directory: Path
    receptor_chain: str
    ligand_chain: str
    receptor_path: Path
    ligand_path: Path
    reference_path: Path
    valid: bool
    validation_performed: bool
    errors: tuple[str, ...]
    receptor_sha256: str
    ligand_sha256: str
    reference_sha256: str
    receptor_sequence: str
    ligand_sequence: str
    reference_sequence: str
    complex_fingerprint_schema: str
    complex_fingerprint_sha256: str
    eligible: bool | None
    eligibility_reason: str


@dataclass(frozen=True)
class _Report:
    input_root: Path
    targets: tuple[_Target, ...]
    issues: tuple[object, ...] = ()
    validation_performed: bool = True
    mobile_partner_domain: object = DEFAULT_MOBILE_PARTNER_DOMAIN

    @property
    def valid_targets(self) -> tuple[_Target, ...]:
        return tuple(target for target in self.targets if target.valid)


class _RecordingEngine:
    name = "recording-demo"
    scientific = False

    def __init__(self) -> None:
        self.calls: list[tuple[ExecutionJob, Path, Path]] = []
        self.cancel_output: Path | None = None

    def run(
        self, *, job: ExecutionJob, protein_path: Path, peptide_path: Path
    ) -> DockingRunResult:
        self.calls.append((job, protein_path, peptide_path))
        if self.cancel_output is not None and len(self.calls) == 1:
            request_batch_cancel(self.cancel_output)
        return DockingRunResult(
            job_id=job.job_id,
            algorithm=job.algorithm.value,
            protocol=job.protocol.value,
            seed=job.seed,
            generations=job.profile.generations,
            population=job.profile.population,
            top_pockets=job.profile.top_pockets,
            evaluation_budget=job.profile.evaluation_budget,
            runtime_sec=1.0,
            dockq=0.5,
            actual_eval_count=job.profile.evaluation_budget if job.profile.budget_policy == "fixed_evaluations" else None,
            stop_reason="objective_evaluation_ceiling" if job.profile.budget_policy == "fixed_evaluations" else None,
            engine=self.name,
            scientific=self.scientific,
        )


def _make_report(root: Path, count: int = 2) -> _Report:
    targets = []
    for index in range(count):
        complex_id = f"1a{index:02d}_AB"
        directory = root / complex_id
        directory.mkdir(parents=True)
        receptor = directory / "A_model_st.pdb"
        ligand = directory / "B_model_st.pdb"
        reference = directory / f"{complex_id}_st.pdb"
        receptor.write_text(f"RECEPTOR-{index}\n", encoding="ascii")
        ligand.write_text(f"LIGAND-{index}\n", encoding="ascii")
        reference.write_text(f"REFERENCE-{index}\n", encoding="ascii")
        fingerprint = hashlib.sha256(f"R{index}|L{index}".encode()).hexdigest()
        targets.append(
            _Target(
                complex_id=complex_id,
                directory=directory,
                receptor_chain="A",
                ligand_chain="B",
                receptor_path=receptor,
                ligand_path=ligand,
                reference_path=reference,
                valid=True,
                validation_performed=True,
                errors=(),
                receptor_sha256=_sha(receptor),
                ligand_sha256=_sha(ligand),
                reference_sha256=_sha(reference),
                receptor_sequence=f"ACDEFGHIK{index}",
                ligand_sequence=f"PEPTIDE{index}",
                reference_sequence=f"ACDEFGHIK{index}|PEPTIDE{index}",
                complex_fingerprint_schema="canonical_sequences_v1",
                complex_fingerprint_sha256=fingerprint,
                eligible=True,
                eligibility_reason="eligible_peptide_length",
            )
        )
    return _Report(input_root=root, targets=tuple(targets))


def _jobs(*seeds: int) -> list[ExecutionJob]:
    return build_execution_plan(
        algorithms=[Algorithm.PSO],
        protocol_mode=ProtocolMode.REDUCED,
        base_config=BaseSearchConfig(generations=4, population=3, top_pockets=2),
        seeds=seeds,
    )


def test_batch_writes_one_audited_result_per_target_job_and_never_passes_reference(
    tmp_path: Path,
):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    report = _make_report(input_root, count=2)
    engine = _RecordingEngine()

    summary = run_batch(
        input_root,
        output_root,
        jobs=_jobs(7, 8),
        engine=engine,
        report=report,
    )

    assert summary.state == "COMPLETED"
    assert summary.planned_jobs == 4
    assert summary.succeeded == 4
    assert summary.failed == 0
    assert len(engine.calls) == 4
    manifest = json.loads(
        (output_root / "batch_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["plan"]["mobile_partner_domain"]["maximum_residues"] == 1200
    reference_paths = {target.reference_path.resolve() for target in report.targets}
    for _, receptor, ligand in engine.calls:
        assert receptor.resolve() not in reference_paths
        assert ligand.resolve() not in reference_paths

    for target in report.targets:
        for job in _jobs(7, 8):
            result_path = (
                output_root
                / "targets"
                / target.complex_id
                / "jobs"
                / job.job_id
                / "receipt.json"
            )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            assert payload["status"] == "SUCCEEDED"
            assert payload["target"]["reference_sha256"] == target.reference_sha256


def test_identical_resume_is_idempotent_and_skips_verified_results(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    report = _make_report(input_root, count=2)
    engine = _RecordingEngine()
    jobs = _jobs(3)

    first = run_batch(
        input_root, output_root, jobs=jobs, engine=engine, report=report
    )
    second = run_batch(
        input_root, output_root, jobs=jobs, engine=engine, report=report
    )

    assert first.attempted == 2
    assert second.attempted == 0
    assert second.skipped_existing == 2
    assert len(engine.calls) == 2
    assert first.batch_id == second.batch_id


def test_changed_plan_cannot_reuse_an_existing_output_directory(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    report = _make_report(input_root, count=1)
    engine = _RecordingEngine()
    run_batch(input_root, output_root, jobs=_jobs(1), engine=engine, report=report)

    with pytest.raises(BatchPlanConflictError, match="different"):
        run_batch(input_root, output_root, jobs=_jobs(2), engine=engine, report=report)


def test_changed_docking_task_cannot_reuse_local_refinement_results(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    report = _make_report(input_root, count=1)
    engine = _RecordingEngine()
    jobs = _jobs(1)
    summary = run_batch(input_root, output_root, jobs=jobs, engine=engine, report=report)
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert manifest["plan"]["jobs"][0]["profile"]["docking_task"] == "local_refinement"
    surface_jobs = [
        replace(job, profile=replace(job.profile, docking_task=DockingTask.SURFACE_DISCOVERY))
        for job in jobs
    ]
    with pytest.raises(BatchPlanConflictError, match="different"):
        run_batch(input_root, output_root, jobs=surface_jobs, engine=engine, report=report)
    assert len(engine.calls) == 1


def test_tampered_result_is_not_treated_as_a_completed_idempotent_job(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    report = _make_report(input_root, count=1)
    engine = _RecordingEngine()
    jobs = _jobs(4)
    run_batch(input_root, output_root, jobs=jobs, engine=engine, report=report)
    result_path = (
        output_root
        / "targets"
        / report.targets[0].complex_id
        / "jobs"
        / jobs[0].job_id
        / "receipt.json"
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["engine_result"]["dockq"] = 0.99
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    resumed = run_batch(
        input_root, output_root, jobs=jobs, engine=engine, report=report
    )

    assert resumed.attempted == 1
    assert resumed.skipped_existing == 0
    assert len(engine.calls) == 2


def test_cancel_marker_stops_between_jobs_and_explicit_resume_finishes(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    report = _make_report(input_root, count=1)
    engine = _RecordingEngine()
    engine.cancel_output = output_root
    jobs = _jobs(10, 11)

    cancelled = run_batch(
        input_root, output_root, jobs=jobs, engine=engine, report=report
    )
    assert cancelled.state == "CANCELLED"
    assert cancelled.succeeded == 1
    assert cancelled.pending == 1
    assert len(engine.calls) == 1

    clear_batch_cancel_request(output_root)
    engine.cancel_output = None
    resumed = run_batch(
        input_root, output_root, jobs=jobs, engine=engine, report=report
    )
    assert resumed.state == "COMPLETED"
    assert resumed.succeeded == 2
    assert resumed.skipped_existing == 1
    assert len(engine.calls) == 2
    status = load_batch_status(output_root)
    assert status is not None
    assert status["counts"]["succeeded"] == 2


def test_invalid_discovery_is_fail_closed_unless_skip_is_explicit(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    base = _make_report(input_root, count=1)

    @dataclass(frozen=True)
    class _Issue:
        code: str = "invalid_target"
        message: str = "bad target"
        relative_path: str = "bad"
        complex_id: str | None = None
        severity: str = "error"

    report = _Report(input_root=base.input_root, targets=base.targets, issues=(_Issue(),))
    with pytest.raises(BatchError, match="invalid inputs"):
        run_batch(
            input_root,
            output_root,
            jobs=_jobs(1),
            engine=_RecordingEngine(),
            report=report,
        )


def test_discovery_and_engine_must_share_the_same_mobile_partner_contract(
    tmp_path: Path,
):
    input_root = tmp_path / "input"
    input_root.mkdir()
    report = _make_report(input_root, count=1)

    class _MismatchedEngine(_RecordingEngine):
        def audit_descriptor(self) -> dict[str, object]:
            payload = DEFAULT_MOBILE_PARTNER_DOMAIN.as_dict()
            payload["maximum_residues"] = 30
            return {"mobile_partner_domain": payload}

    with pytest.raises(BatchError, match="domain contracts differ"):
        run_batch(
            input_root,
            tmp_path / "output",
            jobs=_jobs(1),
            engine=_MismatchedEngine(),
            report=report,
        )


def test_unrelated_nonempty_output_is_not_adopted_as_a_batch(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()
    (output_root / "unrelated.txt").write_text("keep", encoding="utf-8")
    report = _make_report(input_root, count=1)

    with pytest.raises(BatchError, match="non-empty"):
        run_batch(
            input_root,
            output_root,
            jobs=_jobs(1),
            engine=_RecordingEngine(),
            report=report,
        )

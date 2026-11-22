from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from contextlib import ExitStack, suppress
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import random
import traceback
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Any, Iterable, Mapping
import uuid

from docking_confidence.domain.execution import DockingRunResult, ExecutionJob, FIXED_EVALUATION_BUDGET
from docking_confidence.domain.mobile_partner import MobilePartnerResidueDomain
from docking_confidence.ports.engine import DockingEngine

if TYPE_CHECKING:
    from docking_confidence.io.batch_discovery import BatchDiscoveryReport, PPI4DockTarget


BATCH_SCHEMA_VERSION = "persistent_ppi4dock_batch_v3"
MANIFEST_FILENAME = "batch_manifest.json"
STATUS_FILENAME = "batch_status.json"
CANCEL_FILENAME = "CANCEL_REQUESTED"
LOCK_FILENAME = "batch.lock"
RECEIPT_FILENAME = "receipt.json"
CONFIDENCE_DECISION_FILENAME = "confidence_decision.json"

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# Six attempts, at most 1.55 seconds of waiting for a file lock to clear.
_ATOMIC_REPLACE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)


class BatchError(RuntimeError):
    """Base class for persistent batch failures."""


class BatchAlreadyRunningError(BatchError):
    """Raised when another process owns the output-directory batch lock."""


class BatchPlanConflictError(BatchError):
    """Raised when an output directory belongs to a different immutable plan."""


class BatchInputChangedError(BatchError):
    """Raised when an input no longer matches the discovery snapshot."""


@dataclass(frozen=True)
class BatchRunSummary:
    batch_id: str
    state: str
    planned_jobs: int
    succeeded: int
    failed: int
    pending: int
    attempted: int
    skipped_existing: int
    manifest_path: Path
    status_path: Path
    skipped_calibrator: int = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Publish complete JSON atomically, tolerating short Windows file locks."""
    data = _canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        for delay in (*_ATOMIC_REPLACE_RETRY_DELAYS, None):
            try:
                os.replace(temporary, path)
                return
            except OSError as exc:
                transient_access_error = (
                    isinstance(exc, PermissionError)
                    or getattr(exc, "winerror", None) in {5, 32, 33}
                )
                if not transient_access_error or delay is None:
                    raise
                sleep(delay)
    finally:
        # Cleanup must never obscure the original publication/write failure.
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BatchError(f"cannot read valid JSON from {path}") from exc
    if not isinstance(payload, dict):
        raise BatchError(f"expected a JSON object in {path}")
    return payload


def _validate_initial_output(destination: Path, experimental_protocol) -> None:
    if destination.exists() and not (destination / MANIFEST_FILENAME).is_file():
        allowed = {CANCEL_FILENAME, LOCK_FILENAME}
        if experimental_protocol is not None:
            allowed.add("experimental_protocol.json")
        unexpected = sorted(child.name for child in destination.iterdir() if child.name not in allowed)
        if unexpected:
            raise BatchError(f"output directory is non-empty and has no batch manifest: {unexpected}")


class _BatchFileLock:
    """Advisory one-byte lock released by the OS if the worker dies."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any | None = None

    def __enter__(self) -> "_BatchFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        handle = self.path.open("r+b")
        if self.path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise BatchAlreadyRunningError(
                f"another batch worker is using {self.path.parent}"
            ) from exc

        self._handle = handle
        owner = {
            "pid": os.getpid(),
            "acquired_at_utc": _utc_now(),
            "schema_version": BATCH_SCHEMA_VERSION,
        }
        handle.seek(0)
        handle.truncate()
        handle.write(_canonical_json_bytes(owner))
        handle.flush()
        os.fsync(handle.fileno())
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None


def _safe_component(value: object, *, field: str) -> str:
    text = str(value)
    if not _SAFE_COMPONENT.fullmatch(text):
        raise BatchError(f"unsafe {field}: {text!r}")
    return text


def _resolved_disjoint_roots(input_root: str | Path, output_root: str | Path) -> tuple[Path, Path]:
    source = Path(input_root).resolve(strict=True)
    destination = Path(output_root).resolve(strict=False)
    if not source.is_dir():
        raise BatchError(f"input root is not a directory: {source}")
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise BatchError("input and output roots must be separate, non-nested directories")
    return source, destination


def _relative_input_path(path: object, input_root: Path, *, field: str) -> tuple[Path, str]:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise BatchError(f"{field} is not a file: {resolved}")
    try:
        relative = resolved.relative_to(input_root)
    except ValueError as exc:
        raise BatchError(f"{field} is outside the input root: {resolved}") from exc
    return resolved, relative.as_posix()


def _required_sha256(target: object, field: str) -> str:
    value = getattr(target, field, None)
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise BatchError(f"target {getattr(target, 'complex_id', '?')} lacks valid {field}")
    return value


def _required_text(target: object, field: str) -> str:
    value = getattr(target, field, None)
    if not isinstance(value, str) or not value.strip():
        raise BatchError(f"target {getattr(target, 'complex_id', '?')} lacks {field}")
    return value


def _target_snapshot(target: "PPI4DockTarget", input_root: Path) -> dict[str, Any]:
    complex_id = _safe_component(target.complex_id, field="complex_id")
    receptor_path, receptor_relative = _relative_input_path(
        target.receptor_path, input_root, field="receptor_path"
    )
    ligand_path, ligand_relative = _relative_input_path(
        target.ligand_path, input_root, field="ligand_path"
    )
    reference_path, reference_relative = _relative_input_path(
        target.reference_path, input_root, field="reference_path"
    )
    if len({receptor_path, ligand_path, reference_path}) != 3:
        raise BatchError(f"target {complex_id} must use three distinct PDB files")

    snapshot = {
        "complex_id": complex_id,
        "receptor_chain": _required_text(target, "receptor_chain"),
        "ligand_chain": _required_text(target, "ligand_chain"),
        "receptor_path": receptor_relative,
        "ligand_path": ligand_relative,
        "reference_path": reference_relative,
        "receptor_sha256": _required_sha256(target, "receptor_sha256"),
        "ligand_sha256": _required_sha256(target, "ligand_sha256"),
        "reference_sha256": _required_sha256(target, "reference_sha256"),
        "receptor_sequence": _required_text(target, "receptor_sequence"),
        "ligand_sequence": _required_text(target, "ligand_sequence"),
        "reference_sequence": _required_text(target, "reference_sequence"),
        "complex_fingerprint_schema": _required_text(
            target, "complex_fingerprint_schema"
        ),
        "complex_fingerprint_sha256": _required_sha256(
            target, "complex_fingerprint_sha256"
        ),
        "eligible": getattr(target, "eligible", None),
        "eligibility_reason": str(getattr(target, "eligibility_reason", "")),
    }
    if snapshot["eligible"] not in (True, False):
        raise BatchError(f"target {complex_id} has no audited eligibility decision")
    return snapshot


def _job_snapshot(job: ExecutionJob) -> dict[str, Any]:
    job_id = _safe_component(job.job_id, field="job_id")
    if isinstance(job.seed, bool) or not isinstance(job.seed, int) or job.seed < 0:
        raise BatchError(f"job {job_id} has an invalid seed")
    snapshot = {
        "job_id": job_id,
        "algorithm": job.algorithm.value,
        "protocol": job.protocol.value,
        "seed": job.seed,
        "profile": {
            "mode": job.profile.mode.value,
            "docking_task": job.profile.docking_task.value,
            "generations": job.profile.generations,
            "population": job.profile.population,
            "top_pockets": job.profile.top_pockets,
            "evaluation_budget": job.profile.evaluation_budget,
            "generation_factor": job.profile.generation_factor,
            "pocket_factor": job.profile.pocket_factor,
        },
    }
    for field in ("search_extent", "initialization", "budget_policy", "search_scale", "max_translation", "max_rotation_degrees", "pose_source", "execution_order_seed", "execution_strategy"):
        value = getattr(job.profile, field, None)
        if value is not None:
            snapshot["profile"][field] = value
    return snapshot


def _issue_snapshot(issue: object) -> dict[str, Any]:
    return {
        "code": str(getattr(issue, "code", "unknown")),
        "message": str(getattr(issue, "message", "")),
        "relative_path": str(getattr(issue, "relative_path", "")),
        "complex_id": getattr(issue, "complex_id", None),
        "severity": str(getattr(issue, "severity", "error")),
    }


def _build_plan(
    *,
    report: "BatchDiscoveryReport",
    input_root: Path,
    jobs: tuple[ExecutionJob, ...],
    engine: DockingEngine,
    eligible_only: bool,
    skip_invalid: bool,
) -> tuple[dict[str, Any], tuple[object, ...]]:
    if Path(report.input_root).resolve(strict=True) != input_root:
        raise BatchError("discovery report belongs to a different input root")
    if getattr(report, "validation_performed", None) is not True:
        raise BatchError("batch execution requires content-validated discovery")
    mobile_partner_domain = getattr(report, "mobile_partner_domain", None)
    if not isinstance(mobile_partner_domain, MobilePartnerResidueDomain):
        raise BatchError(
            "discovery report has no valid mobile-partner domain contract"
        )
    mobile_partner_domain_payload = mobile_partner_domain.as_dict()
    error_issues = tuple(
        issue
        for issue in report.issues
        if str(getattr(issue, "severity", "error")) == "error"
    )
    if error_issues and not skip_invalid:
        raise BatchError(
            "discovery reported invalid inputs; correct them or explicitly set "
            "skip_invalid=True"
        )

    valid_targets = tuple(report.valid_targets)
    selected = tuple(
        target
        for target in valid_targets
        if not eligible_only or getattr(target, "eligible", None) is True
    )
    if not selected:
        qualifier = "eligible, valid" if eligible_only else "valid"
        raise BatchError(f"discovery found no {qualifier} targets")

    ordered_targets = tuple(sorted(selected, key=lambda item: str(item.complex_id)))
    target_snapshots = [_target_snapshot(target, input_root) for target in ordered_targets]
    target_ids = [item["complex_id"] for item in target_snapshots]
    if len(target_ids) != len(set(target_ids)):
        raise BatchError("discovery returned duplicate complex_id values")

    job_snapshots = [_job_snapshot(job) for job in jobs]
    job_ids = [item["job_id"] for item in job_snapshots]
    if len(job_ids) != len(set(job_ids)):
        raise BatchError("execution plan contains duplicate job_id values")

    engine_name = getattr(engine, "name", None)
    engine_scientific = getattr(engine, "scientific", None)
    if not isinstance(engine_name, str) or not engine_name.strip():
        raise BatchError("engine.name must be a non-empty string")
    if not isinstance(engine_scientific, bool):
        raise BatchError("engine.scientific must be boolean")
    descriptor_builder = getattr(engine, "audit_descriptor", None)
    if callable(descriptor_builder):
        engine_descriptor = descriptor_builder()
        if not isinstance(engine_descriptor, dict):
            raise BatchError("engine.audit_descriptor() must return a dictionary")
    else:
        engine_descriptor = {
            "adapter_class": type(engine).__qualname__,
            "adapter_module": type(engine).__module__,
        }
    _canonical_json_bytes(engine_descriptor)
    engine_domain = engine_descriptor.get("mobile_partner_domain")
    if engine_domain is not None and engine_domain != mobile_partner_domain_payload:
        raise BatchError(
            "discovery and engine mobile-partner domain contracts differ"
        )

    selected_ids = set(target_ids)
    excluded = []
    for target in report.targets:
        target_id = str(target.complex_id)
        if target_id not in selected_ids:
            excluded.append(
                {
                    "complex_id": target_id,
                    "valid": bool(getattr(target, "valid", False)),
                    "eligible": getattr(target, "eligible", None),
                    "errors": list(getattr(target, "errors", ())),
                    "eligibility_reason": str(
                        getattr(target, "eligibility_reason", "")
                    ),
                }
            )

    plan = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "input_root": str(input_root),
        "mobile_partner_domain": mobile_partner_domain_payload,
        "eligible_only": eligible_only,
        "skip_invalid": skip_invalid,
        "engine": {
            "name": engine_name,
            "scientific": engine_scientific,
            "descriptor": engine_descriptor,
        },
        "jobs": job_snapshots,
        "targets": target_snapshots,
        "excluded_targets": sorted(excluded, key=lambda item: item["complex_id"]),
        "discovery_issues": sorted(
            [_issue_snapshot(issue) for issue in report.issues],
            key=lambda item: (
                item["relative_path"],
                item["code"],
                item["message"],
            ),
        ),
    }
    from docking_confidence.domain.execution import OPERATIONAL_CASCADE
    operational_jobs = [item for item in job_snapshots if item["profile"].get("execution_strategy") == OPERATIONAL_CASCADE]
    if operational_jobs:
        from docking_confidence.application.operational import validate_operational_jobs
        validate_operational_jobs(job_snapshots)
        plan["execution_strategy"] = OPERATIONAL_CASCADE
    factorial_jobs = [item for item in job_snapshots if item["profile"].get("initialization") is not None
                      and item["profile"].get("execution_strategy") != OPERATIONAL_CASCADE]
    if factorial_jobs:
        from docking_confidence.application.experimental_protocol import validate_factorial_jobs
        validate_factorial_jobs(job_snapshots)
        rng = random.Random(factorial_jobs[0]["profile"]["execution_order_seed"])
        schedule = []
        for target in target_snapshots:
            block = [f"{target['complex_id']}::{job['job_id']}" for job in job_snapshots]
            rng.shuffle(block)
            schedule.extend(block)
        plan["experimental_design"] = "factorial_2x2"
        plan["execution_limits"] = {"timeout_seconds": getattr(engine, "timeout_seconds", None)}
        plan["execution_order"] = schedule
    return plan, ordered_targets


def _manifest_for_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    plan_sha256 = _payload_sha256(plan)
    core = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "batch_id": plan_sha256,
        "plan_sha256": plan_sha256,
        "created_at_utc": _utc_now(),
        "plan": dict(plan),
    }
    return {**core, "manifest_sha256": _payload_sha256(core)}


def _validate_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != BATCH_SCHEMA_VERSION:
        raise BatchPlanConflictError("existing batch manifest has an unsupported schema")
    recorded = payload.get("manifest_sha256")
    core = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if not isinstance(recorded, str) or recorded != _payload_sha256(core):
        raise BatchPlanConflictError("existing batch manifest failed its integrity hash")
    plan = payload.get("plan")
    if not isinstance(plan, dict) or payload.get("plan_sha256") != _payload_sha256(plan):
        raise BatchPlanConflictError("existing batch plan failed its integrity hash")
    if payload.get("batch_id") != payload.get("plan_sha256"):
        raise BatchPlanConflictError("existing batch_id is inconsistent")


def _load_or_create_manifest(path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    current_plan_sha256 = _payload_sha256(plan)
    if path.exists():
        existing = _load_json_object(path)
        _validate_manifest(existing)
        if existing["plan_sha256"] != current_plan_sha256:
            raise BatchPlanConflictError(
                "output directory is bound to a different input, engine, or execution plan"
            )
        return existing
    manifest = _manifest_for_plan(plan)
    _atomic_write_json(path, manifest)
    return manifest


def _task_key(complex_id: str, job_id: str) -> str:
    return f"{complex_id}::{job_id}"


def _task_sha256(
    *, plan_sha256: str, target: Mapping[str, Any], job: Mapping[str, Any]
) -> str:
    return _payload_sha256(
        {"plan_sha256": plan_sha256, "target": dict(target), "job": dict(job)}
    )


def _task_directory(output_root: Path, complex_id: str, job_id: str) -> Path:
    directory = output_root / "targets" / complex_id / "jobs" / job_id
    resolved = directory.resolve(strict=False)
    try:
        resolved.relative_to(output_root)
    except ValueError as exc:
        raise BatchError("resolved task output escaped the output root") from exc
    return resolved


def _valid_completed_result(path: Path, expected_task_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _load_json_object(path)
        record_sha256 = payload.pop("record_sha256")
    except (BatchError, KeyError):
        return False
    return (
        payload.get("schema_version") == BATCH_SCHEMA_VERSION
        and payload.get("status") == "SUCCEEDED"
        and payload.get("task_sha256") == expected_task_sha256
        and isinstance(record_sha256, str)
        and record_sha256 == _payload_sha256(payload)
    )


def _verify_target_inputs(target: Mapping[str, Any], input_root: Path) -> None:
    for role in ("receptor", "ligand", "reference"):
        path = (input_root / str(target[f"{role}_path"])).resolve(strict=True)
        try:
            path.relative_to(input_root)
        except ValueError as exc:
            raise BatchInputChangedError(
                f"{role} path for {target['complex_id']} escaped the input root"
            ) from exc
        observed = _file_sha256(path)
        expected = target[f"{role}_sha256"]
        if observed != expected:
            raise BatchInputChangedError(
                f"{role} PDB changed after discovery for {target['complex_id']}"
            )


def _result_payload(result: DockingRunResult) -> dict[str, Any]:
    if not isinstance(result, DockingRunResult) or not is_dataclass(result):
        raise BatchError("engine.run must return DockingRunResult")
    payload = asdict(result)
    _canonical_json_bytes(payload)
    return payload


def _validate_engine_result(
    result: DockingRunResult,
    *,
    job: ExecutionJob,
    engine: DockingEngine,
) -> None:
    expected = {
        "job_id": job.job_id,
        "algorithm": job.algorithm.value,
        "protocol": job.protocol.value,
        "seed": job.seed,
        "generations": job.profile.generations,
        "population": job.profile.population,
        "top_pockets": job.profile.top_pockets,
        "evaluation_budget": job.profile.evaluation_budget,
        "engine": engine.name,
        "scientific": engine.scientific,
    }
    for field, value in expected.items():
        if getattr(result, field) != value:
            raise BatchError(f"engine result has inconsistent {field}")
    for field, value in asdict(result).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise BatchError(f"engine result contains non-finite {field}")
    if result.runtime_sec < 0:
        raise BatchError("engine result runtime_sec must be non-negative")
    if result.dockq is not None and not 0.0 <= result.dockq <= 1.0:
        raise BatchError("engine result dockq must be in [0, 1]")
    if result.exit_code not in (None, 0):
        raise BatchError("a successful engine result must have exit_code 0")
    if job.profile.budget_policy == FIXED_EVALUATION_BUDGET and (
        type(result.actual_eval_count) is not int
        or result.actual_eval_count != job.profile.evaluation_budget
        or result.stop_reason != "objective_evaluation_ceiling"
    ):
        raise BatchError("A execução não completou o orçamento fixo de avaliações efetivas.")
    for field in (
        "pose_artifact_path",
        "complex_pose_pdb_path",
        "reference_metrics_path",
        "graph_features_path",
        "engine_summary_path",
        "engine_result_path",
        "resolved_config_path",
        "stdout_log_path",
        "stderr_log_path",
    ):
        value = getattr(result, field)
        if value is None:
            continue
        artifact = Path(value)
        if artifact.is_absolute() or ".." in artifact.parts:
            raise BatchError(f"engine result {field} must be a safe relative path")


_ENGINE_ARTIFACT_FIELDS = (
    "pose_artifact_path",
    "engine_summary_path",
    "engine_result_path",
    "resolved_config_path",
    "stdout_log_path",
    "stderr_log_path",
)


def _normalize_engine_artifact_paths(result: DockingRunResult) -> DockingRunResult:
    updates: dict[str, str] = {}
    for field in _ENGINE_ARTIFACT_FIELDS:
        value = getattr(result, field)
        if value is not None:
            updates[field] = (Path("engine") / value).as_posix()
    return replace(result, **updates) if updates else result


def _postprocess_reference_pose(
    *,
    result: DockingRunResult,
    target: object,
    job: ExecutionJob,
    task_directory: Path,
) -> DockingRunResult:
    if result.pose_artifact_path is None:
        return result
    pose_path = (task_directory / result.pose_artifact_path).resolve(strict=True)
    try:
        pose_path.relative_to(task_directory)
    except ValueError as exc:
        raise BatchError("pose artifact escaped its task directory") from exc

    from docking_confidence.evaluation.reference_dockq import (
        evaluate_reference_pose,
    )

    postprocess_dir = task_directory / "postprocess"
    evaluated = evaluate_reference_pose(
        receptor_model_pdb=Path(getattr(target, "receptor_path")),
        ligand_template_pdb=Path(getattr(target, "ligand_path")),
        reference_complex_pdb=Path(getattr(target, "reference_path")),
        receptor_chain=str(getattr(target, "receptor_chain")),
        ligand_chain=str(getattr(target, "ligand_chain")),
        pose_json=pose_path,
        output_dir=postprocess_dir,
    )
    updates: dict[str, Any] = {
        "dockq": evaluated.dockq,
        "fnat": evaluated.fnat,
        "irmsd": evaluated.irmsd,
        "lrmsd": evaluated.lrmsd,
        "complex_pose_pdb_path": evaluated.complex_pdb_path.relative_to(
            task_directory
        ).as_posix(),
        "reference_metrics_path": evaluated.metrics_path.relative_to(
            task_directory
        ).as_posix(),
    }

    if job.protocol.value == "reduced":
        from docking_confidence.domain.structure import (
            CanonicalComplexFingerprint,
            RunBinding,
        )
        from docking_confidence.features.contact_graph import (
            ContactGraphConfig,
            describe_interface_contact_graph,
        )
        from docking_confidence.io.pdb import read_complex_pose

        complex_fingerprint = CanonicalComplexFingerprint.from_sequences(
            getattr(target, "receptor_sequence"),
            getattr(target, "ligand_sequence"),
        )
        graph_config = ContactGraphConfig()
        pose = read_complex_pose(
            evaluated.complex_pdb_path,
            receptor_chains=[str(getattr(target, "receptor_chain"))],
            ligand_chains=[str(getattr(target, "ligand_chain"))],
            run_binding=RunBinding(
                complex_id=str(getattr(target, "complex_id")),
                run_id=job.job_id,
                algorithm=job.algorithm.value,
                seed=job.seed,
                protocol="reduced",
                engine=result.engine,
            ),
            complex_fingerprint=complex_fingerprint,
            include_hetero=graph_config.include_hetero,
            exclude_waters=graph_config.exclude_waters,
            heavy_atoms_only=graph_config.heavy_atoms_only,
            model_id=graph_config.model_id,
            pose_id=f"{job.job_id}-best-pose",
            pose_rank=1,
            selection_method="dockingpp_best_computed_pose",
            # ``PoseProvenance.source`` is deliberately a small, stable enum.
            # The more specific DockingPP origin remains in selection_method and
            # in the reference-evaluation audit record.
            pose_source="docking_engine",
        )
        graph_path = postprocess_dir / "contact_graph.json"
        _atomic_write_json(
            graph_path, describe_interface_contact_graph(pose, graph_config)
        )
        updates["graph_features_path"] = graph_path.relative_to(
            task_directory
        ).as_posix()
    return replace(result, **updates)


def _initial_status(
    *, manifest: Mapping[str, Any], previous: Mapping[str, Any] | None
) -> dict[str, Any]:
    plan = manifest["plan"]
    old_tasks = previous.get("tasks", {}) if isinstance(previous, Mapping) else {}
    if not isinstance(old_tasks, Mapping):
        old_tasks = {}
    tasks: dict[str, dict[str, Any]] = {}
    for target in plan["targets"]:
        for job in plan["jobs"]:
            key = _task_key(target["complex_id"], job["job_id"])
            old = old_tasks.get(key, {})
            attempt_count = old.get("attempt_count", 0) if isinstance(old, Mapping) else 0
            if isinstance(attempt_count, bool) or not isinstance(attempt_count, int):
                attempt_count = 0
            tasks[key] = {
                "complex_id": target["complex_id"],
                "job_id": job["job_id"],
                "algorithm": job["algorithm"],
                "protocol": job["protocol"],
                "seed": job["seed"],
                "state": "PENDING",
                "attempt_count": max(0, attempt_count),
                "started_at_utc": old.get("started_at_utc") if isinstance(old, Mapping) else None,
                "completed_at_utc": old.get("completed_at_utc") if isinstance(old, Mapping) else None,
                "last_error": old.get("last_error") if isinstance(old, Mapping) else None,
                "duration_seconds": old.get("duration_seconds") if isinstance(old, Mapping) else None,
            }
            if job["profile"].get("initialization") is not None:
                tasks[key].update(search_extent=job["profile"]["search_extent"], initialization=job["profile"]["initialization"])
    created_at = (
        previous.get("created_at_utc")
        if isinstance(previous, Mapping)
        else _utc_now()
    )
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "batch_id": manifest["batch_id"],
        "plan_sha256": manifest["plan_sha256"],
        "state": "RUNNING",
        "worker_pid": os.getpid(),
        "created_at_utc": created_at,
        "updated_at_utc": _utc_now(),
        "completed_at_utc": None,
        "tasks": tasks,
        "counts": {},
    }


def _update_counts(status: dict[str, Any]) -> None:
    states = [task["state"] for task in status["tasks"].values()]
    status["counts"] = {
        "planned": len(states),
        "succeeded": states.count("SUCCEEDED"),
        "failed": states.count("FAILED"),
        "pending": states.count("PENDING"),
        "running": states.count("RUNNING"),
        "skipped_calibrator": states.count("SKIPPED_CALIBRATOR"),
    }
    status["updated_at_utc"] = _utc_now()


def _write_status(path: Path, status: dict[str, Any]) -> None:
    _update_counts(status)
    _atomic_write_json(path, status)


def _summary(
    *,
    manifest: Mapping[str, Any],
    status: Mapping[str, Any],
    attempted: int,
    skipped_existing: int,
    manifest_path: Path,
    status_path: Path,
) -> BatchRunSummary:
    counts = status["counts"]
    return BatchRunSummary(
        batch_id=str(manifest["batch_id"]),
        state=str(status["state"]),
        planned_jobs=int(counts["planned"]),
        succeeded=int(counts["succeeded"]),
        failed=int(counts["failed"]),
        pending=int(counts["pending"] + counts["running"]),
        attempted=attempted,
        skipped_existing=skipped_existing,
        manifest_path=manifest_path,
        status_path=status_path,
        skipped_calibrator=int(counts.get("skipped_calibrator", 0)),
    )


def _calibrated_job_pairs(
    jobs: tuple[ExecutionJob, ...],
) -> tuple[tuple[ExecutionJob, ...], dict[str, str]]:
    """Keep each algorithm/seed's REDUCED before its conditional FULL."""
    pairs: dict[tuple[str, int], dict[str, ExecutionJob]] = {}
    for job in jobs:
        protocol = job.protocol.value
        if protocol not in {"reduced", "full"}:
            raise BatchError("calibrated execution requires FULL/REDUCED pairs")
        pair = pairs.setdefault((job.algorithm.value, job.seed), {})
        if protocol in pair:
            raise BatchError("calibrated execution has a duplicate algorithm/seed/protocol")
        pair[protocol] = job
    ordered: list[ExecutionJob] = []
    reduced_for_full: dict[str, str] = {}
    for pair in pairs.values():
        if set(pair) != {"reduced", "full"}:
            raise BatchError("calibrated execution requires a FULL and REDUCED for every algorithm/seed")
        ordered.extend((pair["reduced"], pair["full"]))
        reduced_for_full[pair["full"].job_id] = pair["reduced"].job_id
    return tuple(ordered), reduced_for_full


def _confidence_decision(
    *,
    gate: Any,
    descriptor: Mapping[str, Any],
    destination: Path,
    source: Path,
    manifest: Mapping[str, Any],
    target: Mapping[str, Any],
    reduced_job: Mapping[str, Any],
    full_job: Mapping[str, Any],
    reduced_status: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute from verified evidence; persisted skip decisions are never inputs."""
    evaluation_started = perf_counter()
    preparation_runtime_sec = None
    directory = _task_directory(destination, target["complex_id"], reduced_job["job_id"])
    receipt_path = directory / RECEIPT_FILENAME
    reduced_hash = _task_sha256(
        plan_sha256=manifest["plan_sha256"], target=target, job=reduced_job
    )
    fallback = {
        "action": "RUN_FULL", "reason": "reduced_failed",
        "probability_lower": None, "probability_upper": None, "probability": None,
        "threshold": None, "artifact_sha256": descriptor.get("artifact_sha256"),
        "decision_capable": False,
    }
    decision = dict(fallback)
    receipt_sha256 = None
    receipt_record_sha256 = None
    try:
        _verify_target_inputs(target, source)
        if reduced_status["state"] == "SUCCEEDED":
            if not _valid_completed_result(receipt_path, reduced_hash):
                decision["reason"] = "reduced_receipt_missing_or_invalid"
            else:
                receipt = _load_json_object(receipt_path)
                preparation_runtime_sec = receipt.get("confidence_preparation_runtime_sec")
                for name, expected in (
                    ("batch_id", manifest["batch_id"]),
                    ("plan_sha256", manifest["plan_sha256"]),
                    ("complex_id", target["complex_id"]),
                    ("target", dict(target)), ("job", dict(reduced_job)),
                ):
                    if receipt.get(name) != expected:
                        raise BatchError(f"REDUCED receipt disagrees with the immutable plan: {name}")
                receipt_sha256 = _file_sha256(receipt_path)
                receipt_record_sha256 = receipt["record_sha256"]
                if _file_sha256(directory / "confidence" / "evidence.json") != receipt.get("confidence_evidence_sha256"):
                    raise BatchError("REDUCED confidence evidence disagrees with its receipt")
                evaluated = gate.evaluate(task_directory=directory, receipt=receipt)
                if not isinstance(evaluated, dict) or evaluated.get("action") not in {"ACCEPT_REDUCED", "RUN_FULL"}:
                    raise BatchError("calibrator returned an invalid decision")
                _canonical_json_bytes(evaluated)
                expected_artifact = descriptor.get("artifact_sha256")
                if expected_artifact is not None and evaluated.get("artifact_sha256") != expected_artifact:
                    raise BatchError("calibrator decision belongs to a different artifact")
                if evaluated["action"] == "ACCEPT_REDUCED" and evaluated.get("decision_capable") is not True:
                    raise BatchError("calibrator cannot authorize automatic acceptance")
                if not isinstance(evaluated.get("reason"), str) or not evaluated["reason"]:
                    raise BatchError("calibrator decision has no audit reason")
                # Reject evidence changed during evaluation as well as on resume.
                if _file_sha256(receipt_path) != receipt_sha256:
                    raise BatchError("REDUCED receipt changed during calibration")
                _verify_target_inputs(target, source)
                decision = {**fallback, **evaluated}
    except Exception as exc:
        decision = {
            **fallback, "reason": "calibrator_evaluation_failed",
            "error_type": type(exc).__name__, "error_message": str(exc),
        }
    evaluation_runtime_sec = perf_counter() - evaluation_started
    measured_preparation = (isinstance(preparation_runtime_sec, (int, float))
                            and not isinstance(preparation_runtime_sec, bool)
                            and math.isfinite(preparation_runtime_sec) and preparation_runtime_sec >= 0)
    core = {
        **decision,
        "schema_version": "calibrated_batch_decision_v1",
        "batch_id": manifest["batch_id"], "plan_sha256": manifest["plan_sha256"],
        "complex_id": target["complex_id"],
        "reduced_job_id": reduced_job["job_id"], "full_job_id": full_job["job_id"],
        "reduced_task_sha256": reduced_hash,
        "full_task_sha256": _task_sha256(
            plan_sha256=manifest["plan_sha256"], target=target, job=full_job
        ),
        "reduced_receipt_sha256": receipt_sha256,
        "reduced_receipt_record_sha256": receipt_record_sha256,
        "confidence_descriptor_sha256": _payload_sha256(descriptor),
        "evaluated_at_utc": _utc_now(),
        "confidence_evaluation_runtime_sec": evaluation_runtime_sec,
        "confidence_overhead_sec": (preparation_runtime_sec + evaluation_runtime_sec
                                    if measured_preparation else None),
        "confidence_timing_scope": "feature_preparation_and_decision_before_audit_commit; excludes_engine_and_reference_scoring",
    }
    record = {**core, "decision_sha256": _payload_sha256(core)}
    decision_path = directory / CONFIDENCE_DECISION_FILENAME
    try:
        _atomic_write_json(decision_path, record)
    except OSError as exc:
        # A missing durable acceptance must never suppress the FULL fallback.
        record.update(
            action="RUN_FULL", reason="confidence_decision_write_failed",
            decision_capable=False, error_type=type(exc).__name__, error_message=str(exc),
        )
        record["decision_sha256"] = None
    return {
        **{name: record.get(name) for name in (
            "complex_id", "reduced_job_id", "full_job_id", "action", "reason",
            "probability_lower", "probability_upper", "probability", "threshold",
            "artifact_sha256", "decision_capable", "decision_sha256",
            "error_type", "error_message",
        )},
        "decision_path": decision_path.relative_to(destination).as_posix(),
    }


def request_batch_cancel(output_root: str | Path) -> Path:
    """Request cooperative cancellation between legacy engine calls."""

    root = Path(output_root).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / CANCEL_FILENAME
    _atomic_write_json(
        marker,
        {
            "schema_version": BATCH_SCHEMA_VERSION,
            "requested_at_utc": _utc_now(),
        },
    )
    return marker


def clear_batch_cancel_request(output_root: str | Path) -> None:
    """Explicitly acknowledge a cancellation before resuming a batch."""

    (Path(output_root).resolve(strict=False) / CANCEL_FILENAME).unlink(missing_ok=True)


def load_batch_status(output_root: str | Path) -> dict[str, Any] | None:
    path = Path(output_root).resolve(strict=False) / STATUS_FILENAME
    return _load_json_object(path) if path.is_file() else None


def run_batch(
    input_root: str | Path,
    output_root: str | Path,
    *,
    jobs: Iterable[ExecutionJob],
    engine: DockingEngine,
    report: "BatchDiscoveryReport | None" = None,
    eligible_only: bool = True,
    skip_invalid: bool = False,
    confidence_artifact: str | Path | None = None,
    experimental_protocol: str | Path | None = None,
    collect_operational_pairs: bool = False,
    shadow_confidence: bool = False,
) -> BatchRunSummary:
    """Run or resume an immutable, auditable PPI4DOCK batch.

    A successful target/job result is skipped only when its record integrity hash and
    immutable task hash are valid. Cancellation is cooperative because the legacy
    ``DockingEngine`` port has no in-flight cancellation operation. The experimental
    reference PDB is fingerprinted for provenance but is never passed to the engine.
    """

    source, destination = _resolved_disjoint_roots(input_root, output_root)
    if not isinstance(eligible_only, bool) or not isinstance(skip_invalid, bool):
        raise BatchError("eligible_only and skip_invalid must be boolean")
    _validate_initial_output(destination, experimental_protocol)
    destination.mkdir(parents=True, exist_ok=True)
    job_tuple = tuple(jobs)
    if not job_tuple:
        raise BatchError("execution plan is empty")
    from docking_confidence.domain.execution import OPERATIONAL_CASCADE
    operational = any(job.profile.execution_strategy == OPERATIONAL_CASCADE for job in job_tuple)
    if not isinstance(collect_operational_pairs, bool):
        raise BatchError("collect_operational_pairs must be boolean")
    if not isinstance(shadow_confidence, bool) or (shadow_confidence and (
            collect_operational_pairs or confidence_artifact is None)):
        raise BatchError("confidence validation requires an explicit frozen artifact and cannot use unscored pair collection")
    if collect_operational_pairs and (not operational or confidence_artifact is not None):
        raise BatchError("article pair collection requires operational pairs and no inference artifact")
    if operational:
        from docking_confidence.application.operational import validate_operational_jobs
        validate_operational_jobs([_job_snapshot(job) for job in job_tuple])
    factorial = any(job.profile.initialization is not None and job.profile.execution_strategy != OPERATIONAL_CASCADE for job in job_tuple)
    if factorial and confidence_artifact is not None:
        raise BatchError("factorial experiments must execute all four conditions; calibrator skipping is unavailable")
    if factorial != (experimental_protocol is not None):
        raise BatchError("factorial jobs require a prepared experimental protocol; legacy jobs do not accept one")
    gate = None
    confidence_descriptor: dict[str, Any] | None = None
    reduced_for_full: dict[str, str] = {}
    if confidence_artifact is not None or operational:
        from docking_confidence.application.calibrated_batch import CalibratedBatchGate

        job_tuple, reduced_for_full = _calibrated_job_pairs(job_tuple)
        if operational:
            from docking_confidence.application.operational import FullFallbackGate, operational_gate
            gate = (FullFallbackGate("article_paired_data_collection") if collect_operational_pairs
                    else operational_gate(confidence_artifact, job_tuple, engine))
        else:
            gate = CalibratedBatchGate(confidence_artifact)
            gate.validate_jobs(job_tuple, engine)
        if shadow_confidence:
            from docking_confidence.application.calibrated_batch import ValidationConfidenceGate
            if not isinstance(gate, CalibratedBatchGate) or gate.protocol_issues:
                raise BatchError("article confidence validation requires a matching calibrated execution contract")
            gate = ValidationConfidenceGate(gate)
        confidence_descriptor = gate.audit_descriptor()
        if not isinstance(confidence_descriptor, dict):
            raise BatchError("calibrator audit descriptor must be a dictionary")
        _canonical_json_bytes(confidence_descriptor)

    if report is None:
        from docking_confidence.io.batch_discovery import discover_ppi4dock

        report = discover_ppi4dock(source, validate_contents=True)

    plan, targets = _build_plan(
        report=report,
        input_root=source,
        jobs=job_tuple,
        engine=engine,
        eligible_only=eligible_only,
        skip_invalid=skip_invalid,
    )
    if confidence_descriptor is not None:
        plan["confidence_gate"] = confidence_descriptor
    if experimental_protocol is not None:
        from docking_confidence.application.experimental_protocol import bind_experimental_protocol
        plan["experimental_protocol"] = bind_experimental_protocol(experimental_protocol, plan)
    target_by_id = {str(target.complex_id): target for target in targets}
    job_by_id = {job.job_id: job for job in job_tuple}
    job_snapshots = {item["job_id"]: item for item in plan["jobs"]}

    manifest_path = destination / MANIFEST_FILENAME
    status_path = destination / STATUS_FILENAME
    cancel_path = destination / CANCEL_FILENAME

    from docking_confidence.application.batch_telemetry import WorkerHeartbeat

    with _BatchFileLock(destination / LOCK_FILENAME), ExitStack() as observers:
        _validate_initial_output(destination, experimental_protocol)
        manifest = _load_or_create_manifest(manifest_path, plan)
        previous = load_batch_status(destination)
        if previous is not None and previous.get("plan_sha256") != manifest["plan_sha256"]:
            raise BatchPlanConflictError("batch status belongs to a different plan")
        status = _initial_status(manifest=manifest, previous=previous)
        heartbeat = observers.enter_context(WorkerHeartbeat(destination, manifest["batch_id"]))
        status["worker_id"] = heartbeat.data["worker_id"]

        task_specs: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
        skipped_existing = 0
        for target_snapshot in plan["targets"]:
            inputs_match_snapshot = True
            try:
                _verify_target_inputs(target_snapshot, source)
            except (OSError, BatchInputChangedError):
                inputs_match_snapshot = False
            for job_snapshot in plan["jobs"]:
                key = _task_key(target_snapshot["complex_id"], job_snapshot["job_id"])
                task_hash = _task_sha256(
                    plan_sha256=manifest["plan_sha256"],
                    target=target_snapshot,
                    job=job_snapshot,
                )
                result_path = (
                    _task_directory(
                        destination,
                        target_snapshot["complex_id"],
                        job_snapshot["job_id"],
                    )
                    / RECEIPT_FILENAME
                )
                if inputs_match_snapshot and _valid_completed_result(result_path, task_hash):
                    status["tasks"][key]["state"] = "SUCCEEDED"
                    skipped_existing += 1
                    if gate is not None and job_snapshot["protocol"] == "full":
                        task_specs.append((key, target_snapshot, job_snapshot))
                else:
                    task_specs.append((key, target_snapshot, job_snapshot))
        if plan.get("execution_order"):
            positions = {key: index for index, key in enumerate(plan["execution_order"])}
            task_specs.sort(key=lambda item: positions[item[0]])
        _write_status(status_path, status)

        attempted = 0
        cancelled = False
        for key, target_snapshot, job_snapshot in task_specs:
            if cancel_path.is_file():
                cancelled = True
                break

            task_status = status["tasks"][key]
            if gate is not None and job_snapshot["protocol"] == "full":
                reduced_id = reduced_for_full[job_snapshot["job_id"]]
                reduced_key = _task_key(target_snapshot["complex_id"], reduced_id)
                decision = _confidence_decision(
                    gate=gate, descriptor=confidence_descriptor, destination=destination,
                    source=source, manifest=manifest, target=target_snapshot,
                    reduced_job=job_snapshots[reduced_id], full_job=job_snapshot,
                    reduced_status=status["tasks"][reduced_key],
                )
                task_status["confidence_decision"] = decision
                status["tasks"][reduced_key]["confidence_decision"] = decision
                if cancel_path.is_file():
                    _write_status(status_path, status)
                    cancelled = True
                    break
                if task_status["state"] == "SUCCEEDED":
                    _write_status(status_path, status)
                    continue
                if decision["action"] == "ACCEPT_REDUCED":
                    task_status.update(
                        state="SKIPPED_CALIBRATOR", completed_at_utc=_utc_now(),
                        started_at_utc=None, last_error=None, duration_seconds=None,
                    )
                    _write_status(status_path, status)
                    continue

            attempted += 1
            task_status.update(
                {
                    "state": "RUNNING",
                    "attempt_count": int(task_status["attempt_count"]) + 1,
                    "started_at_utc": _utc_now(),
                    "completed_at_utc": None,
                    "last_error": None,
                    "duration_seconds": None,
                }
            )
            _write_status(status_path, status)

            complex_id = str(target_snapshot["complex_id"])
            job_id = str(job_snapshot["job_id"])
            task_directory = _task_directory(destination, complex_id, job_id)
            result_path = task_directory / RECEIPT_FILENAME
            error_path = task_directory / "error.json"
            engine_work_dir = task_directory / "engine"
            task_hash = _task_sha256(
                plan_sha256=manifest["plan_sha256"],
                target=target_snapshot,
                job=job_snapshot,
            )
            job_started = perf_counter()
            heartbeat.start_job(task_status, dict(job_snapshot), engine_work_dir)
            try:
                _verify_target_inputs(target_snapshot, source)
                target = target_by_id[complex_id]
                job = job_by_id[job_id]
                # The reference_path is deliberately absent from the search-engine call.
                run_arguments = {
                    "job": job,
                    "protein_path": Path(target.receptor_path).resolve(strict=True),
                    "peptide_path": Path(target.ligand_path).resolve(strict=True),
                }
                if bool(getattr(engine, "supports_artifact_work_dir", False)):
                    engine_work_dir.mkdir(parents=True, exist_ok=True)
                    result = engine.run(**run_arguments, work_dir=engine_work_dir)
                else:
                    result = engine.run(**run_arguments)
                _validate_engine_result(result, job=job, engine=engine)
                if (
                    bool(getattr(engine, "requires_reference_postprocess", False))
                    and result.pose_artifact_path is None
                ):
                    raise BatchError(
                        "scientific engine did not persist the required best pose"
                    )
                result = _normalize_engine_artifact_paths(result)
                if gate is not None and job.protocol.value == "reduced":
                    confidence_preparation_started = perf_counter()
                    gate.prepare_evidence(
                        task_directory=task_directory, target=target, job=job, result=result
                    )
                    confidence_evidence_sha256 = _file_sha256(
                        task_directory / "confidence" / "evidence.json"
                    )
                    confidence_preparation_runtime_sec = perf_counter() - confidence_preparation_started
                result = _postprocess_reference_pose(
                    result=result,
                    target=target,
                    job=job,
                    task_directory=task_directory,
                )
                _verify_target_inputs(target_snapshot, source)
                _validate_engine_result(result, job=job, engine=engine)
                result_core = {
                    "schema_version": BATCH_SCHEMA_VERSION,
                    "status": "SUCCEEDED",
                    "batch_id": manifest["batch_id"],
                    "plan_sha256": manifest["plan_sha256"],
                    "task_sha256": task_hash,
                    "complex_id": complex_id,
                    "target": dict(target_snapshot),
                    "job": dict(job_snapshots[job_id]),
                    "engine_work_dir": str(
                        engine_work_dir.relative_to(destination).as_posix()
                    ),
                    "engine_result": _result_payload(result),
                    "completed_at_utc": _utc_now(),
                }
                if gate is not None and job.protocol.value == "reduced":
                    result_core["confidence_evidence_sha256"] = confidence_evidence_sha256
                    result_core["confidence_preparation_runtime_sec"] = confidence_preparation_runtime_sec
                result_record = {
                    **result_core,
                    "record_sha256": _payload_sha256(result_core),
                }
                _atomic_write_json(result_path, result_record)
                task_status.update(
                    {
                        "state": "SUCCEEDED",
                        "completed_at_utc": result_core["completed_at_utc"],
                        "last_error": None,
                    }
                )
            except Exception as exc:
                error_record = {
                    "schema_version": BATCH_SCHEMA_VERSION,
                    "status": "FAILED",
                    "batch_id": manifest["batch_id"],
                    "plan_sha256": manifest["plan_sha256"],
                    "task_sha256": task_hash,
                    "complex_id": complex_id,
                    "job_id": job_id,
                    "attempt": task_status["attempt_count"],
                    "error_type": type(exc).__name__,
                    "engine_status": getattr(exc, "status", None),
                    "failure_reason": getattr(exc, "reason", None),
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "failed_at_utc": _utc_now(),
                }
                _atomic_write_json(error_path, error_record)
                task_status.update(
                    {
                        "state": "FAILED",
                        "completed_at_utc": error_record["failed_at_utc"],
                        "last_error": {
                            "type": error_record["error_type"],
                            "message": error_record["message"],
                            "path": str(error_path.relative_to(destination).as_posix()),
                        },
                    }
                )
            task_status["duration_seconds"] = perf_counter() - job_started
            _write_status(status_path, status)
            heartbeat.end_job(task_status)

        _update_counts(status)
        if cancelled:
            status["state"] = "CANCELLED"
        elif status["counts"]["failed"]:
            status["state"] = "COMPLETED_WITH_ERRORS"
        else:
            status["state"] = "COMPLETED"
        status["completed_at_utc"] = _utc_now()
        _write_status(status_path, status)
        heartbeat.finish(status["state"])
        return _summary(
            manifest=manifest,
            status=status,
            attempted=attempted,
            skipped_existing=skipped_existing,
            manifest_path=manifest_path,
            status_path=status_path,
        )

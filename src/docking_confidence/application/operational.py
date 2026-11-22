"""Operational budgets and conservative routing, separate from scientific controls."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

from docking_confidence.domain.execution import (
    BaseSearchConfig, DockingTask, FIXED_EVALUATION_BUDGET, OPERATIONAL_CASCADE,
    OperationalSearchConfig,
)


def load_operational_config(path: str | Path | None = None) -> BaseSearchConfig:
    source = Path(path) if path is not None else Path(__file__).resolve().parents[3] / "configs/operational_cascade.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    expected = {"schema_version", "population", "generations", "full_evaluation_budget",
                "reduced_evaluation_fraction", "reduced_search_scale", "max_translation",
                "max_rotation_degrees", "initialization", "pose_source", "early_stopping"}
    if not isinstance(data, dict) or set(data) != expected or data["schema_version"] != OPERATIONAL_CASCADE:
        raise ValueError("invalid operational configuration schema or fields")
    if data["reduced_evaluation_fraction"] != 0.75:
        raise ValueError("operational REDUCED must use exactly 75% of FULL evaluations")
    if data["early_stopping"] != "disabled_pending_independent_validation":
        raise ValueError("operational early stopping has not been independently validated")
    for field in ("population", "generations"):
        if isinstance(data[field], bool) or not isinstance(data[field], int) or data[field] < 1:
            raise ValueError(f"operational {field} must be a positive integer")
    controls = OperationalSearchConfig(**{key: data[key] for key in (
        "full_evaluation_budget", "reduced_search_scale", "max_translation",
        "max_rotation_degrees", "initialization", "pose_source")})
    config = BaseSearchConfig(generations=data["generations"], population=data["population"],
                              top_pockets=1, docking_task=DockingTask.LOCAL_REFINEMENT, operational=controls)
    config.validate()
    return config


def validate_operational_jobs(jobs: list[dict]) -> None:
    """Reject mixed regimes or tampered unequal-budget recipes before any run."""
    groups = defaultdict(dict)
    for job in jobs:
        profile = job["profile"]
        mode = job["protocol"]
        if (profile.get("execution_strategy") != OPERATIONAL_CASCADE
                or profile.get("budget_policy") != FIXED_EVALUATION_BUDGET
                or profile.get("docking_task") != "local_refinement"
                or profile.get("top_pockets") != 1
                or mode not in {"reduced", "full"}
                or profile.get("mode") != mode):
            raise ValueError("operational cascade requires unmixed fixed-budget local FULL/REDUCED pairs")
        for field in ("population", "generations", "evaluation_budget"):
            if isinstance(profile.get(field), bool) or not isinstance(profile.get(field), int) or profile[field] < 1:
                raise ValueError(f"invalid operational {field}")
        if profile["evaluation_budget"] < profile["population"]:
            raise ValueError("operational budget must cover the initial population")
        key = (job["algorithm"], job["seed"])
        if mode in groups[key]:
            raise ValueError("duplicate operational arm")
        groups[key][mode] = profile
    if not groups:
        raise ValueError("operational plan is empty")
    for pair in groups.values():
        if set(pair) != {"reduced", "full"}:
            raise ValueError("operational cascade requires a FULL and REDUCED per algorithm/seed")
        reduced, full = pair["reduced"], pair["full"]
        controls = OperationalSearchConfig(
            full_evaluation_budget=full["evaluation_budget"],
            reduced_search_scale=reduced["search_scale"],
            max_translation=full["max_translation"], max_rotation_degrees=full["max_rotation_degrees"],
            initialization=full["initialization"], pose_source=full["pose_source"])
        controls.validate(full["population"])
        if reduced["evaluation_budget"] != controls.reduced_evaluation_budget:
            raise ValueError("operational REDUCED evaluations must be exactly 75% of FULL")
        if (full.get("search_extent") != "expanded" or full.get("search_scale") != 1.0
                or reduced.get("search_extent") != "reduced"):
            raise ValueError("invalid operational search extents")
        for field in ("population", "generations", "initialization", "max_translation", "max_rotation_degrees", "pose_source"):
            if reduced.get(field) != full.get(field):
                raise ValueError(f"operational pair differs in {field}")


class FullFallbackGate:
    """Unavailable/incompatible calibration can never suppress the FULL fallback."""

    def __init__(self, reason: str, *, artifact_path: Path | None = None, issues=()):
        self.reason = reason
        self.descriptor = {
            "schema_version": "operational_full_fallback_v1", "decision_capable": False,
            "artifact_path": str(artifact_path) if artifact_path is not None else None,
            "artifact_sha256": None, "reason": reason, "execution_protocol_issues": list(issues),
        }
        if artifact_path is not None and artifact_path.is_file():
            from docking_confidence.application.batch import _file_sha256
            self.descriptor["artifact_sha256"] = _file_sha256(artifact_path)

    def audit_descriptor(self) -> dict:
        return dict(self.descriptor)

    def prepare_evidence(self, *, task_directory: Path, **kwargs) -> None:
        from docking_confidence.application.batch import _atomic_write_json
        _atomic_write_json(task_directory / "confidence/evidence.json", {
            "schema_version": "operational_calibrator_unavailable_v1", **self.descriptor,
        })

    def evaluate(self, **kwargs) -> dict:
        return {"action": "RUN_FULL", "reason": self.reason, "decision_capable": False,
                "artifact_sha256": self.descriptor["artifact_sha256"],
                "execution_protocol_issues": self.descriptor["execution_protocol_issues"]}


def operational_gate(artifact_path, jobs, engine):
    from docking_confidence.application.calibrated_batch import CalibratedBatchGate, configured_calibrator_path
    path = Path(artifact_path).expanduser().resolve() if artifact_path is not None else configured_calibrator_path()
    if path is None or not path.is_file():
        return FullFallbackGate("operational_calibrator_not_installed", artifact_path=path)
    try:
        gate = CalibratedBatchGate(path)
        gate.validate_jobs(jobs, engine)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        return FullFallbackGate("operational_calibrator_invalid", artifact_path=path, issues=[str(exc)])
    if gate.protocol_issues:
        return FullFallbackGate("operational_calibrator_requires_retraining", artifact_path=path,
                                issues=gate.protocol_issues)
    return gate


def validate_training_regime(config: dict, plan: dict) -> None:
    """Keep old calibrator recipes from silently absorbing a new budget regime."""
    contract = config.get("execution_contract")
    operational = plan.get("execution_strategy") == OPERATIONAL_CASCADE
    if not operational and contract is None:
        return
    if not operational or not isinstance(contract, dict) or contract.get("schema_version") != OPERATIONAL_CASCADE:
        raise ValueError("operational training requires a separate matching execution_contract configuration")
    validate_operational_jobs(plan["jobs"])
    for job in plan["jobs"]:
        p = job["profile"]
        for field in ("population", "initialization", "max_translation", "max_rotation_degrees"):
            if p.get(field) != contract.get(field):
                raise ValueError(f"training execution_contract mismatch: {field}")
        budget_key = "reduced_evaluation_budget" if job["protocol"] == "reduced" else "full_evaluation_budget"
        if p["evaluation_budget"] != contract.get(budget_key):
            raise ValueError(f"training execution_contract mismatch: {budget_key}")
        if job["protocol"] == "reduced" and p["search_scale"] != contract.get("reduced_search_scale"):
            raise ValueError("training execution_contract mismatch: reduced_search_scale")

"""Connect verified REDUCED-only inference to the persistent docking worker."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import pandas as pd
import yaml

from docking_confidence.application.batch import (
    _atomic_write_json, _file_sha256, _load_json_object, _payload_sha256,
    _validate_manifest,
)
from docking_confidence.application.predict import load_prediction_session
from docking_confidence.application.reduced_evidence import _artifact_path, read_reduced_graph
from docking_confidence.domain.structure import CanonicalComplexFingerprint, RunBinding
from docking_confidence.features.contact_graph import ContactGraphConfig, describe_interface_contact_graph
from docking_confidence.io.pdb import read_complex_pose
from docking_confidence.io.config import LOCAL_REFINEMENT_CONTEXT_SCHEMA_VERSION
from docking_confidence.features.execution_context import local_execution_context


def configured_calibrator_path() -> Path | None:
    """Resolve the explicitly installed model; never select a run by modification time."""
    root = Path(__file__).resolve().parents[3]
    binding_path = root / "configs" / "calibrator.yaml"
    if not binding_path.is_file():
        return None
    binding = yaml.safe_load(binding_path.read_text(encoding="utf-8"))
    if not isinstance(binding, dict) or binding.get("schema_version") != "docking_calibrator_binding_v1":
        raise ValueError("invalid calibrator binding configuration")
    value = binding.get("artifact_path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("calibrator binding requires artifact_path")
    if binding.get("purpose") == "generalized_quality_prediction":
        # The installed 2025 margin regressor is not a legacy accept/FULL gate.
        # It is scored by application.generalized_calibrator and predict CLI.
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


class ValidationConfidenceGate:
    """Record the frozen predictor while retaining FULL ground truth for the article."""

    def __init__(self, predictor):
        self.predictor = predictor

    def audit_descriptor(self):
        return {**self.predictor.audit_descriptor(),
                "validation_mode": "score_reduced_and_always_execute_full_v1"}

    def prepare_evidence(self, **kwargs):
        return self.predictor.prepare_evidence(**kwargs)

    def evaluate(self, **kwargs):
        prediction = self.predictor.evaluate(**kwargs)
        return {**prediction, "predictor_action": prediction["action"],
                "predictor_reason": prediction["reason"], "action": "RUN_FULL",
                "reason": "article_validation_full_required", "validation_mode": True}


class CalibratedBatchGate:
    """Freeze and verify once; each REDUCED run still validates its own evidence."""

    def __init__(self, artifact_path: str | Path):
        self.artifact_path = Path(artifact_path).expanduser().resolve(strict=True)
        self.session = load_prediction_session(self.artifact_path, require_bundle=True)
        self.artifact = self.session.artifact
        self.config = self.artifact["resolved_config"]
        if not self.artifact.get("docking_task"):
            raise ValueError("calibrated docking requires an artifact with an explicit docking task")
        self.contract, self.contract_issue = self._execution_contract()
        self.protocol_issues = ["execution_plan_not_checked"]

    def _execution_contract(self) -> tuple[dict | None, str | None]:
        # This preparation links the trained CSV to its immutable source batch.
        # A model without that evidence can be scored, but cannot skip FULL here.
        try:
            preparation_path = (
                Path(self.artifact["training_dataset_snapshot"]["path"]).parent
                / "dataset_build_report.json"
            )
            preparation = _load_json_object(preparation_path)
            if (preparation.get("valid") is not True
                    or preparation.get("dataset_sha256") != self.artifact["training_dataset_snapshot"]["sha256"]
                    or preparation.get("config_sha256") != self.artifact["training_config_source"]["sha256"]):
                raise ValueError("training preparation does not match the model")
            if preparation.get("source_kind") == "multiple_hash_verified_training_batches_v1":
                plans = []
                batch_hashes = []
                frames = []
                seen = set()
                sources = preparation.get("source_preparations", [])
                digests = preparation.get("source_preparation_hashes", {})
                if not sources or len(sources) != len(set(sources)):
                    raise ValueError("merged training preparation requires unique sources")
                for source in sources:
                    source_path = Path(source)
                    if _file_sha256(source_path) != digests.get(source):
                        raise ValueError("training source preparation changed")
                    item = _load_json_object(source_path)
                    if item.get("valid") is not True or item.get("config_sha256") != preparation["config_sha256"]:
                        raise ValueError("training source configuration mismatch")
                    csv_path = Path(item["dataset_path"])
                    if _file_sha256(csv_path) != item["dataset_sha256"]:
                        raise ValueError("training source dataset changed")
                    frame = pd.read_csv(BytesIO(csv_path.read_bytes()))
                    groups = set(frame["complex_fingerprint_sha256"])
                    if groups & seen:
                        raise ValueError("molecular groups overlap between training batches")
                    seen.update(groups)
                    frames.append(frame)
                    manifest = _load_json_object(Path(item["batch_dir"]) / "batch_manifest.json")
                    _validate_manifest(manifest)
                    if manifest["manifest_sha256"] != item["batch_manifest_sha256"]:
                        raise ValueError("training source batch changed")
                    plans.append(manifest["plan"])
                    batch_hashes.append(manifest["manifest_sha256"])
                if batch_hashes != preparation.get("source_batch_manifest_sha256"):
                    raise ValueError("merged training batch bindings changed")
                merged = pd.concat(frames, ignore_index=True).sort_values(
                    ["complex_fingerprint_sha256", "algorithm", "seed"], kind="stable")
                import hashlib
                if hashlib.sha256(merged.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest() != preparation["dataset_sha256"]:
                    raise ValueError("merged training rows do not match source datasets")
                first = plans[0]
                for plan in plans[1:]:
                    if any(plan[key] != first[key] for key in ("engine", "jobs", "mobile_partner_domain")):
                        raise ValueError("training batches use different execution contracts")
                return {"source_batch_manifest_sha256": batch_hashes,
                        "engine": first["engine"], "jobs": first["jobs"],
                        "mobile_partner_domain": first["mobile_partner_domain"]}, None
            manifest_path = Path(preparation["batch_dir"]) / "batch_manifest.json"
            manifest = _load_json_object(manifest_path)
            _validate_manifest(manifest)
            if manifest["manifest_sha256"] != preparation.get("batch_manifest_sha256"):
                raise ValueError("training batch manifest does not match its preparation")
            plan = manifest["plan"]
            return {
                "batch_manifest_sha256": manifest["manifest_sha256"],
                "engine": plan["engine"], "jobs": plan["jobs"],
                "mobile_partner_domain": plan["mobile_partner_domain"],
            }, None
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            return None, f"training_execution_contract_unavailable: {exc}"

    def audit_descriptor(self) -> dict:
        return {
            "schema_version": "verified_calibrated_docking_v1",
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": self.session.artifact_sha256,
            "audit_manifest_sha256": self.session.manifest_sha256,
            "config_sha256": self.artifact["training_config_source"]["sha256"],
            "docking_task": self.artifact["docking_task"],
            "threshold": self.artifact["policy"].get("threshold"),
            "decision_capable": self.artifact["decision_capable"],
            "authorization_gates": deepcopy(self.artifact["authorization"]["gates"]),
            "execution_contract_sha256": _payload_sha256(self.contract) if self.contract else None,
            "execution_contract_issue": self.contract_issue,
            "execution_jobs": deepcopy(self.contract["jobs"]) if self.contract else [],
            "execution_protocol_issues": list(self.protocol_issues),
            "feature_source": "reference_free_reduced_pose",
        }

    def validate_jobs(self, jobs, engine) -> None:
        snapshots = []
        pairs = {}
        for job in jobs:
            if job.profile.docking_task.value != self.artifact["docking_task"]:
                raise ValueError("docking task is incompatible with the selected calibrator")
            if job.protocol.value not in {"full", "reduced"}:
                raise ValueError("calibrated docking requires FULL/REDUCED job pairs")
            key = (job.algorithm.value, job.seed)
            modes = pairs.setdefault(key, [])
            modes.append(job.protocol.value)
            profile = {key: value for key, value in asdict(job.profile).items() if value is not None}
            profile["mode"] = job.profile.mode.value
            profile["docking_task"] = job.profile.docking_task.value
            snapshots.append({
                "algorithm": job.algorithm.value, "protocol": job.protocol.value,
                "seed": job.seed, "profile": profile,
            })
        if not pairs or any(sorted(modes) != ["full", "reduced"] for modes in pairs.values()):
            raise ValueError("calibrated docking needs exactly one FULL and REDUCED per algorithm/seed")
        self.protocol_issues = []
        if self.contract is None:
            self.protocol_issues.append("training_execution_contract_unavailable")
            return
        expected = [{key: job[key] for key in ("algorithm", "protocol", "seed", "profile")}
                    for job in self.contract["jobs"]]
        for job in expected:
            job["profile"] = {key: value for key, value in job["profile"].items() if value is not None}
        if any(job["profile"].get("execution_strategy") == "operational_cascade_v1" for job in snapshots):
            for mode in ("reduced", "full"):
                observed = {job["profile"]["evaluation_budget"] for job in snapshots if job["protocol"] == mode}
                trained = {job["profile"]["evaluation_budget"] for job in expected if job["protocol"] == mode}
                if observed != trained:
                    self.protocol_issues.append(f"{mode}_evaluation_budget_differs_from_training")
        order = lambda item: (item["algorithm"], item["seed"], item["protocol"])
        if sorted(snapshots, key=order) != sorted(expected, key=order):
            self.protocol_issues.append("execution_jobs_differ_from_training")
        descriptor = engine.audit_descriptor() if callable(getattr(engine, "audit_descriptor", None)) else {}
        observed_engine = {"name": engine.name, "scientific": engine.scientific, "descriptor": descriptor}
        if observed_engine != self.contract["engine"]:
            self.protocol_issues.append("execution_engine_differs_from_training")

    def prepare_evidence(self, *, task_directory: Path, target, job, result) -> None:
        """Create features before reference scoring, exclusively from the computed pose."""
        from docking_confidence.evaluation.reference_dockq import prepare_reduced_pose

        directory = Path(task_directory).resolve()
        pose_json = _artifact_path(directory, result.pose_artifact_path)
        evidence_dir = directory / "confidence"
        evidence_dir.mkdir(exist_ok=True)
        raw_pose = prepare_reduced_pose(
            receptor_model_pdb=Path(target.receptor_path),
            ligand_template_pdb=Path(target.ligand_path),
            receptor_chain=str(target.receptor_chain), ligand_chain=str(target.ligand_chain),
            pose_json=pose_json, output_dir=evidence_dir,
        )
        graph_cfg = ContactGraphConfig.from_mapping(self.config["contact_graph"])
        pose = read_complex_pose(
            raw_pose, receptor_chains=[target.receptor_chain], ligand_chains=[target.ligand_chain],
            run_binding=RunBinding(complex_id=target.complex_id, run_id=job.job_id,
                algorithm=job.algorithm.value, seed=job.seed, protocol="reduced", engine=result.engine),
            complex_fingerprint=CanonicalComplexFingerprint.from_sequences(
                target.receptor_sequence, target.ligand_sequence,
                expected_sha256=target.complex_fingerprint_sha256,
                schema=target.complex_fingerprint_schema),
            include_hetero=graph_cfg.include_hetero, exclude_waters=graph_cfg.exclude_waters,
            heavy_atoms_only=graph_cfg.heavy_atoms_only, model_id=graph_cfg.model_id,
            pose_id=f"{job.job_id}-best-pose", pose_rank=1,
            selection_method="dockingpp_best_computed_pose", pose_source="docking_engine",
        )
        graph_path = evidence_dir / "contact_graph.json"
        _atomic_write_json(graph_path, describe_interface_contact_graph(pose, graph_cfg))
        evidence = {
            "schema_version": "reference_free_reduced_evidence_v1",
            "complex_id": target.complex_id, "run_id": job.job_id,
            "complex_fingerprint_sha256": target.complex_fingerprint_sha256,
            "artifact_sha256": self.session.artifact_sha256,
            "engine_pose_json": pose_json.relative_to(directory).as_posix(),
            "engine_pose_sha256": _file_sha256(pose_json),
            "pose_path": raw_pose.relative_to(directory).as_posix(),
            "pose_sha256": _file_sha256(raw_pose),
            "graph_path": graph_path.relative_to(directory).as_posix(),
            "graph_sha256": _file_sha256(graph_path),
        }
        if self.artifact.get("feature_schema") == LOCAL_REFINEMENT_CONTEXT_SCHEMA_VERSION:
            context, trace = local_execution_context(
                algorithm=job.algorithm.value, receptor_model_pdb=target.receptor_path,
                ligand_template_pdb=target.ligand_path, pose_json=pose_json,
                expected_receptor_sha256=target.receptor_sha256,
                expected_ligand_sha256=target.ligand_sha256,
            )
            evidence.update(execution_context=context, execution_context_evidence=trace)
        _atomic_write_json(evidence_dir / "evidence.json", evidence)

    def evaluate(self, *, task_directory: Path, receipt: dict) -> dict:
        directory = Path(task_directory).resolve()
        target, job, result = receipt["target"], receipt["job"], receipt["engine_result"]
        if job["protocol"] != "reduced" or result["protocol"] != "reduced":
            raise ValueError("calibrator accepts only REDUCED results")
        if receipt.get("record_sha256") != _payload_sha256({
            key: value for key, value in receipt.items() if key != "record_sha256"
        }):
            raise ValueError("REDUCED receipt integrity mismatch")
        evidence_path = directory / "confidence" / "evidence.json"
        if _file_sha256(evidence_path) != receipt.get("confidence_evidence_sha256"):
            raise ValueError("REDUCED inference evidence disagrees with its immutable receipt")
        evidence = _load_json_object(evidence_path)
        for key, expected in {
            "schema_version": "reference_free_reduced_evidence_v1",
            "complex_id": target["complex_id"], "run_id": job["job_id"],
            "complex_fingerprint_sha256": target["complex_fingerprint_sha256"],
            "artifact_sha256": self.session.artifact_sha256,
            "engine_pose_json": result["pose_artifact_path"],
        }.items():
            if evidence.get(key) != expected:
                raise ValueError(f"REDUCED inference evidence disagrees with receipt: {key}")
        for path_key, hash_key in (("engine_pose_json", "engine_pose_sha256"),
                                  ("pose_path", "pose_sha256"), ("graph_path", "graph_sha256")):
            if _file_sha256(_artifact_path(directory, evidence[path_key])) != evidence[hash_key]:
                raise ValueError(f"REDUCED inference evidence hash mismatch: {path_key}")
        # Reuse the same pose/graph validation as dataset preparation, but supply
        # the reference-free evidence created before any DockQ calculation.
        graph_receipt = deepcopy(receipt)
        graph_receipt["engine_result"].update(
            complex_pose_pdb_path=evidence["pose_path"], graph_features_path=evidence["graph_path"])
        graph = read_reduced_graph(directory, graph_receipt, self.config)
        row = {key: target[key] for key in (
            "complex_id", "receptor_sequence", "ligand_sequence",
            "complex_fingerprint_schema", "complex_fingerprint_sha256")}
        row.update(
            run_id=job["job_id"], algorithm=job["algorithm"], seed=job["seed"],
            engine_reduced=result["engine"], scientific_reduced=result["scientific"],
            docking_task=self.artifact["docking_task"], feature_schema=self.artifact["feature_schema"],
            config=self.artifact["training_config_source"]["path"],
            config_sha256=self.artifact["training_config_source"]["sha256"],
        )
        row.update({key: result.get(key) for key in self.artifact["feature_columns"]
                    if not key.startswith("graph_")})
        row.update(graph)
        if self.artifact.get("feature_schema") == LOCAL_REFINEMENT_CONTEXT_SCHEMA_VERSION:
            context = evidence.get("execution_context")
            if not isinstance(context, dict):
                raise ValueError("REDUCED inference evidence lacks input-pose context")
            expected_hashes = {
                "input_pose_retention_receptor_sha256": target["receptor_sha256"],
                "input_pose_retention_ligand_sha256": target["ligand_sha256"],
                "input_pose_retention_pose_sha256": evidence["engine_pose_sha256"],
            }
            if any(context.get(key) != value for key, value in expected_hashes.items()):
                raise ValueError("input-pose context disagrees with immutable REDUCED evidence")
            row.update(context)
        csv_bytes = pd.DataFrame([row]).to_csv(index=False).encode("utf-8")
        scored = self.session.predict_bytes(csv_bytes)
        prediction = scored.iloc[0]
        decision = {
            "action": str(prediction["action"]), "reason": str(prediction["reason"]),
            "probability_lower": float(prediction["probability_lower"]),
            "probability_upper": float(prediction["probability_upper"]),
            "probability": float(prediction["probability"]),
            "threshold": self.artifact["policy"].get("threshold"),
            "artifact_sha256": self.session.artifact_sha256,
            "decision_capable": self.artifact["decision_capable"],
            "in_applicability_domain": bool(prediction["in_applicability_domain"]),
            "unseen_complex": bool(prediction["unseen_complex"]),
            "trusted_reduced_input_provenance": bool(prediction["trusted_reduced_input_provenance"]),
            "execution_protocol_issues": list(self.protocol_issues),
            "feature_source": "reference_free_reduced_pose",
        }
        if self.protocol_issues:
            decision["predictor_reason"] = decision["reason"]
            decision.update(action="RUN_FULL", reason=self.protocol_issues[0])
        # Persist exactly the REDUCED-only input and actual operational decision.
        (directory / "confidence" / "inference_input.csv").write_bytes(csv_bytes)
        scored["predictor_action"] = scored["action"]
        scored["predictor_reason"] = scored["reason"]
        scored["action"], scored["reason"] = decision["action"], decision["reason"]
        scored.to_csv(directory / "confidence" / "prediction.csv", index=False)
        return decision

"""Opt-in standard protein-protein scoring using official DockQ, with provenance.

Historical evaluation is unchanged. No numerical substitute is used when the
official evaluator is absent; an isolated dependency target is supported.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

METRIC_ID = "official_dockq_v2_protein_protein"
METRIC_SCHEMA = "structural_quality_metric_v1"
RESULT_SCHEMA = "official_dockq_reference_evaluation_v1"


class OfficialDockQUnavailableError(RuntimeError):
    """The explicitly requested standard evaluator cannot be executed."""


def canonical_sha256(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def quality_metric_descriptor(version: str = "2.1.3") -> dict:
    if not isinstance(version, str) or not version.startswith("2."):
        raise ValueError("Official DockQ v2 implementation_version is required")
    return {"schema_version": METRIC_SCHEMA, "method": METRIC_ID, "implementation": "DockQ",
            "implementation_version": version, "chain_mapping": "explicit_receptor_ligand",
            "capri_peptide": False, "contact_cutoff_angstrom": 5, "interface_cutoff_angstrom": 10}


def quality_metric_sha256(descriptor: dict) -> str:
    if not isinstance(descriptor, dict) or canonical_sha256(descriptor) != canonical_sha256(quality_metric_descriptor(descriptor.get("implementation_version"))):
        raise ValueError("Metric descriptor differs from explicit standard protein-protein DockQ")
    return canonical_sha256(descriptor)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chain(value, role):
    if not isinstance(value, str) or len(value) != 1 or value.isspace() or not value.isascii():
        raise ValueError(f"{role} must be one explicit nonblank ASCII chain identifier")
    return value


def _protein_chains(path: Path) -> set[str]:
    chains = {line[21:22] for line in path.read_text(encoding="utf-8").splitlines()
              if line.startswith("ATOM  ") and len(line) >= 22}
    if not chains:
        raise ValueError(f"No protein ATOM records found: {path}")
    return chains


def _invoke_backend(request: dict, *, python_executable=None, dependency_path=None, timeout_seconds=180) -> dict:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    executable = Path(python_executable or sys.executable).resolve(strict=True)
    worker = Path(__file__).with_name("official_dockq_runner.py")
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    if dependency_path is not None:
        dependency = Path(dependency_path).resolve(strict=True)
        if not dependency.is_dir():
            raise ValueError("dependency_path must be a directory")
        env["PYTHONPATH"] = str(dependency)
    args = [str(executable), "-B", "-X", "utf8", str(worker)]
    if request.get("preflight"):
        args.append("--preflight")
    completed = subprocess.run(args, input=None if request.get("preflight") else json.dumps(request, allow_nan=False),
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=timeout_seconds,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        result = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise OfficialDockQUnavailableError("Official DockQ worker returned invalid JSON: " + completed.stderr[-1200:]) from exc
    if completed.returncode != 0 or result.get("available") is not True:
        raise OfficialDockQUnavailableError(f"Official DockQ unavailable: {result.get('error_type')}: {result.get('error')}")
    return result


def official_dockq_preflight(*, python_executable=None, dependency_path=None, timeout_seconds=30) -> dict:
    try:
        result = _invoke_backend({"preflight": True}, python_executable=python_executable,
                                 dependency_path=dependency_path, timeout_seconds=timeout_seconds)
        descriptor = quality_metric_descriptor(result["implementation_version"])
        return {**result, "quality_metric": descriptor, "quality_metric_sha256": quality_metric_sha256(descriptor)}
    except (OfficialDockQUnavailableError, OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "quality_metric_id": METRIC_ID, "error": str(exc),
                "installation_hint": "Install DockQ==2.1.3 in an isolated runtime; do not downgrade the application's NumPy."}


def evaluate_standard_complex(model_pdb, reference_pdb, *, receptor_chain, ligand_chain,
                              output_dir, reference_receptor_chain=None, reference_ligand_chain=None,
                              python_executable=None, dependency_path=None, expected_metric_sha256=None,
                              expected_model_sha256=None, expected_reference_sha256=None, expected_version="2.1.3",
                              timeout_seconds=180, source_binding=None) -> dict:
    """Score one fixed two-chain mapping and write a new immutable result JSON."""
    model = Path(model_pdb).resolve(strict=True)
    native = Path(reference_pdb).resolve(strict=True)
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Standard evaluation output must be new or empty")
    rec, lig = _chain(receptor_chain, "receptor_chain"), _chain(ligand_chain, "ligand_chain")
    native_rec = _chain(reference_receptor_chain or rec, "reference_receptor_chain")
    native_lig = _chain(reference_ligand_chain or lig, "reference_ligand_chain")
    if rec == lig or native_rec == native_lig:
        raise ValueError("Receptor and ligand chains must be different")
    if _protein_chains(model) != {rec, lig} or _protein_chains(native) != {native_rec, native_lig}:
        raise ValueError("Model and reference must contain exactly the declared two protein chains")
    model_hash, native_hash = _sha(model), _sha(native)
    if expected_model_sha256 is not None and model_hash != expected_model_sha256:
        raise ValueError("Saved model differs from its expected SHA-256")
    if expected_reference_sha256 is not None and native_hash != expected_reference_sha256:
        raise ValueError("Reference differs from its expected SHA-256")
    requested_metric = quality_metric_descriptor(expected_version)
    if expected_metric_sha256 is not None and quality_metric_sha256(requested_metric) != expected_metric_sha256:
        raise ValueError("Requested metric differs from frozen quality metric SHA-256")
    chain_map = {native_rec: rec, native_lig: lig}
    response = _invoke_backend({"model_pdb": str(model), "reference_pdb": str(native),
        "model_sha256": model_hash, "reference_sha256": native_hash,
        "native_to_model_chain_map": chain_map, "expected_version": expected_version},
        python_executable=python_executable, dependency_path=dependency_path, timeout_seconds=timeout_seconds)
    descriptor = quality_metric_descriptor(response["implementation_version"])
    if descriptor != requested_metric:
        raise ValueError("Actual evaluator differs from frozen metric")
    if _sha(model) != model_hash or _sha(native) != native_hash:
        raise ValueError("Input PDB changed during structural evaluation")
    values = response["metrics"]
    metrics = {"dockq": float(values["DockQ"]), "fnat": float(values["fnat"]),
               "irmsd": float(values["iRMSD"]), "lrmsd": float(values["LRMSD"])}
    if not all(math.isfinite(v) for v in metrics.values()) or not 0 <= metrics["dockq"] <= 1 or not 0 <= metrics["fnat"] <= 1 or min(metrics["irmsd"], metrics["lrmsd"]) < 0:
        raise ValueError("Official evaluator returned invalid metrics")
    score = metrics["dockq"]
    category = "high" if score >= .80 else "medium" if score >= .49 else "acceptable" if score >= .23 else "incorrect"
    record = {"schema_version": RESULT_SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_used_for_search": False, "quality_metric_id": METRIC_ID,
        "quality_metric": descriptor, "quality_metric_sha256": quality_metric_sha256(descriptor),
        "inputs": {"model_pdb": str(model), "model_sha256": model_hash,
                   "reference_pdb": str(native), "reference_sha256": native_hash},
        "native_to_model_chain_map": chain_map, "metrics": {**metrics, "dockq_category": category},
        "implementation": {k:response[k] for k in ("implementation", "implementation_version", "implementation_sha256", "constants_sha256", "python_executable")},
        "official_interface_result": response.get("official_interface_result"),
        "scope": "post-search standard quality evaluation; not a decision or calibration guarantee"}
    if source_binding is not None:
        if not isinstance(source_binding, dict):
            raise ValueError("source_binding must be a JSON object")
        # Take an independent JSON snapshot before it enters the integrity hash.
        record["source_binding"] = json.loads(json.dumps(source_binding, allow_nan=False))
    record["record_sha256"] = canonical_sha256(record)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "standard_dockq.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    return {**record, "metrics_path": str(path), **metrics}


def evaluate_reference_pose_standard(*, receptor_model_pdb, ligand_template_pdb,
        reference_complex_pdb, receptor_chain, ligand_chain, pose_json, output_dir, **runtime):
    """Assemble the computed pose reference-free, then run official DockQ."""
    from docking_confidence.evaluation.reference_dockq import prepare_reduced_pose
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Standard pose evaluation output must be new or empty")
    preflight = official_dockq_preflight(**{k:v for k,v in runtime.items() if k in {"python_executable", "dependency_path"}})
    if not preflight["available"]:
        raise OfficialDockQUnavailableError(preflight["error"])
    model = prepare_reduced_pose(receptor_model_pdb=receptor_model_pdb, ligand_template_pdb=ligand_template_pdb,
        receptor_chain=receptor_chain, ligand_chain=ligand_chain, pose_json=pose_json, output_dir=output/"computed_pose")
    return evaluate_standard_complex(model, reference_complex_pdb, receptor_chain=receptor_chain,
        ligand_chain=ligand_chain, output_dir=output/"quality", **runtime)


def rescore_receipt_standard(receipt_path, reference_pdb, *, output_dir, **runtime):
    """Rescore an unchanged saved receipt into a separate standard result."""
    path = Path(receipt_path).resolve(strict=True)
    receipt = json.loads(path.read_bytes())
    digest = receipt.get("record_sha256")
    if not digest or canonical_sha256({k:v for k,v in receipt.items() if k != "record_sha256"}) != digest:
        raise ValueError("Saved receipt failed its integrity SHA-256")
    target, engine = receipt["target"], receipt["engine_result"]
    relative = Path(engine["complex_pose_pdb_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Saved complex pose path must remain within its receipt directory")
    model = (path.parent/relative).resolve(strict=True)
    model.relative_to(path.parent)
    result = evaluate_standard_complex(model, reference_pdb, receptor_chain=target["receptor_chain"],
        ligand_chain=target["ligand_chain"], output_dir=output_dir,
        expected_reference_sha256=target["reference_sha256"], **runtime)
    return {**result, "source_receipt": str(path), "source_receipt_record_sha256": digest,
            "complex_id": receipt["complex_id"], "run_id": receipt["job"]["job_id"]}

"""One frozen article study: factorial controls, live VA validation, joint analysis.

The docking algorithms and historical protocols are reused unchanged. FULL is
always retained as ground truth; probability fitting never uses study outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import os
import sys
import traceback
from importlib.metadata import version

import pandas as pd
from sklearn.isotonic import IsotonicRegression

from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
from docking_confidence.application.batch import (
    _BatchFileLock, _atomic_write_json, _build_plan, _file_sha256,
    _load_json_object, _payload_sha256, _resolved_disjoint_roots, _utc_now,
    clear_batch_cancel_request, load_batch_status, request_batch_cancel, run_batch,
)
from docking_confidence.application.calibrated_batch import CalibratedBatchGate, configured_calibrator_path
from docking_confidence.application.experimental_protocol import prepare_experimental_protocol, _plan_differences
from docking_confidence.application.operational import load_operational_config
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, FactorialExperimentConfig, ProtocolMode
from docking_confidence.io.batch_discovery import discover_ppi4dock
from docking_confidence.risk.scientific import load_risk_protocol, risk_coverage_envelope
from docking_confidence.validation.metrics import sigmoid_raw_score

PROTOCOL = "article_protocol.json"
STATUS = "article_status.json"
BLOCKS = ("controle_orientador", "avaliacao_calibrador")
SCHEMA = "integrated_article_study_v2"
ROOT = Path(__file__).resolve().parents[3]
THREAD_LIMITS = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def study_software():
    files = ("application/article_study.py", "application/article_reporting.py", "application/batch.py",
             "application/calibrated_batch.py", "application/factorial_reporting.py", "risk/scientific.py")
    return {"source_sha256": {name: _file_sha256(ROOT / "src/docking_confidence" / name) for name in files},
            "risk_protocol_sha256": _file_sha256(ROOT / "configs/article_risk_protocol.yaml"),
            "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "matplotlib")}}


def select_independent_targets(report, known_fingerprints):
    """Freeze one molecular representative before reading any study outcomes."""
    if report.has_errors or not report.eligible_targets:
        raise ValueError("Valide uma pasta com complexos elegíveis e sem erros de entrada.")
    canonical = {}
    aliases = []
    overlap = []
    for target in sorted(report.eligible_targets, key=lambda target: target.complex_id):
        fingerprint = target.complex_fingerprint_sha256
        if fingerprint in known_fingerprints:
            overlap.append(target.complex_id)
        if fingerprint in canonical:
            aliases.append({"selected": canonical[fingerprint].complex_id, "excluded": target.complex_id})
        else:
            canonical[fingerprint] = target
    if overlap:
        raise ValueError("O teste contém proteínas usadas no desenvolvimento do calibrador: " + ", ".join(overlap))
    return replace(report, targets=tuple(canonical.values()), issues=()), aliases


def development_fingerprints(gate):
    known = set(gate.artifact["known_development_complex_fingerprints"])
    # Include training inputs whose REDUCED failed, even though they are absent
    # from the fitted CSV. The gate has already verified these source bindings.
    preparation = _load_json_object(Path(gate.artifact["training_dataset_snapshot"]["path"]).parent / "dataset_build_report.json")
    sources = preparation.get("source_preparations", [])
    preparations = [_load_json_object(Path(path)) for path in sources] if sources else [preparation]
    for item in preparations:
        manifest = _load_json_object(Path(item["batch_dir"]) / "batch_manifest.json")
        known.update(target["complex_fingerprint_sha256"] for target in manifest["plan"]["targets"])
    return known


def study_jobs(pose_source):
    base = load_operational_config()
    operational = build_execution_plan(algorithms=list(Algorithm), protocol_mode=ProtocolMode.OPERATIONAL,
                                       base_config=base, seeds=[1])
    controls = base.operational
    factorial = build_execution_plan(algorithms=list(Algorithm), protocol_mode=ProtocolMode.FACTORIAL,
        base_config=BaseSearchConfig(population=base.population, top_pockets=1,
            experiment=FactorialExperimentConfig(controls.full_evaluation_budget,
                controls.reduced_search_scale, controls.max_translation,
                controls.max_rotation_degrees, pose_source)), seeds=[1])
    return {BLOCKS[0]: factorial, BLOCKS[1]: operational}


def freeze_comparators(gate):
    """Freeze probability models and exact risk curves; no unapproved cutoff."""
    groups = gate.artifact["calibration"]["score_groups"]
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        [group["score"] for group in groups],
        [group["positive_count"] / group["weight"] for group in groups],
        sample_weight=[group["weight"] for group in groups])
    risk = pd.read_csv(gate.artifact_path.parent / "risk_calibration_predictions.csv", float_precision="round_trip")
    raw = risk["base_raw_score"].to_numpy()
    risk_protocol = load_risk_protocol()
    scores = {"original": sigmoid_raw_score(raw), "isotonic": iso.predict(raw),
              "venn_abers": risk["probability_lower"].to_numpy(dtype=float)}
    profile = risk_coverage_envelope(risk, scores, risk["in_applicability_domain"].to_numpy(dtype=bool),
                                    risk_protocol, stage="risk_calibration")
    policies = {method: {"supported": False, "threshold": None,
        "method": risk_protocol["method"], "reason": "no_approved_risk_tolerance; curves_only",
        "decision_score": ("venn_abers_lower_multiprobability" if method == "venn_abers" else method + "_probability")}
        for method in scores}
    return {"isotonic_x": iso.X_thresholds_.tolist(), "isotonic_y": iso.y_thresholds_.tolist(),
            "policies": policies, "risk_protocol": risk_protocol, "validation_risk_profile": profile,
            "probability_training_target": gate.artifact["preservation_rule"],
            "target_scope": "probabilities retain the original experimental target; structural risk uses the separately declared published-DockQ-class endpoint; scores are not relabelled as probabilities of the new endpoint",
            "scope": "frozen_before_study; original_calibration_role_only; historical_risk_role_reanalysis_exploratory"}


def load_article_protocol(output_dir):
    payload = _load_json_object(Path(output_dir) / PROTOCOL)
    core = {key: value for key, value in payload.items() if key != "protocol_sha256"}
    if payload.get("schema_version") != SCHEMA or payload.get("protocol_sha256") != _payload_sha256(core):
        raise ValueError("O protocolo integrado foi alterado ou está inválido.")
    return payload


def _calibrator_incompatibility_message(gate, engine):
    message = "O calibrador é incompatível com o estudo: " + ", ".join(gate.protocol_issues)
    if "execution_engine_differs_from_training" not in gate.protocol_issues or gate.contract is None:
        return message
    current = {"name": engine.name, "scientific": engine.scientific, "descriptor": engine.audit_descriptor()}
    differences = _plan_differences({"engine": gate.contract["engine"]}, {"engine": current})
    runtime_fields = {"engine.descriptor.python_executable", "engine.descriptor.python_runtime_version"}
    labels = {"engine.descriptor.python_executable": "Executável Python",
              "engine.descriptor.python_runtime_version": "Versão do Python"}
    fields = {item["field"] for item in differences}
    if fields and fields <= runtime_fields:
        message += ("\n\nA interface está usando um ambiente Python diferente do registrado no treinamento. "
                    "Feche o servidor com Ctrl+C e, na pasta deste projeto, execute `.\\iniciar_sistema.cmd`. "
                    "Esse comando abre a interface com a .venv do projeto. Depois, valide novamente o estudo.")
    else:
        message += "\n\nO motor ou sua configuração difere do treinamento. Confira as diferenças antes de iniciar."
    for item in differences:
        message += (f"\n\n{labels.get(item['field'], item['field'])}: "
                    f"treinamento `{item['saved']}`; atual `{item['current']}`.")
    return message


def prepare_article_study(input_dir, output_dir, *, pose_source, report=None, artifact_path=None, engine=None):
    if not isinstance(pose_source, str) or not pose_source.strip():
        raise ValueError("Descreva a origem da pose fornecida e sua relação com a referência.")
    source, destination = _resolved_disjoint_roots(input_dir, output_dir)
    path = artifact_path or configured_calibrator_path()
    if path is None:
        raise ValueError("Instale o calibrador antes de preparar o artigo.")
    gate = CalibratedBatchGate(path)
    jobs = study_jobs(pose_source.strip())
    engine = engine or DockingPPLocalEngine(timeout_seconds=600)
    gate.validate_jobs(jobs[BLOCKS[1]], engine)
    if gate.protocol_issues:
        raise ValueError(_calibrator_incompatibility_message(gate, engine))
    report = report or discover_ppi4dock(source, validate_contents=True)
    if Path(report.input_root).resolve() != source:
        raise ValueError("Valide novamente a pasta de entrada selecionada.")
    selected, aliases = select_independent_targets(report, development_fingerprints(gate))
    plans = {name: _build_plan(report=selected, input_root=source, jobs=tuple(block_jobs),
        engine=engine, eligible_only=True, skip_invalid=False)[0] for name, block_jobs in jobs.items()}
    config_path = ROOT / "configs/operational_cascade.yaml"
    core = {"schema_version": SCHEMA, "input_dir": str(source), "output_dir": str(destination),
        "pose_source": pose_source.strip(), "artifact_path": str(gate.artifact_path),
        "artifact_sha256": gate.session.artifact_sha256, "model_manifest_sha256": gate.session.manifest_sha256,
        "operational_config_sha256": _file_sha256(config_path), "block_plans": plans,
        "selection": {"input_count": len(report.targets), "selected_count": len(selected.targets), "aliases": aliases,
            "excluded_ineligible": [{"complex_id": target.complex_id, "reason": target.eligibility_reason,
                                      "mobile_residue_count": target.ligand_residue_count}
                                     for target in report.targets if target.eligible is not True],
            "rule": "first complex_id per fingerprint before outcomes; reject all known development inputs"},
        "comparators": freeze_comparators(gate), "automatic_acceptance_authorized": False,
        "historical_artifact_decision_capable_audit_only": gate.artifact["decision_capable"],
        "worker_thread_limits": THREAD_LIMITS,
        "software": study_software(),
        "analysis": {"scope": "independent_complex_evaluation; exploratory_unless_external_prospective_evidence",
            "full_policy": "always execute FULL; log frozen confidence on REDUCED-only features",
            "thresholds": "no approved risk tolerance; predeclared risk-coverage curves; no test tuning or retraining",
            "inference_scope": "5700 reduced uniform / 7600 full uniform only; no extrapolation to factorial arms",
            "cost_scope": "measured_validation_cost_and_separate_counterfactual_deployment_estimate",
            "bootstrap_repetitions": 500, "bootstrap_seed": 20261010}}
    destination.mkdir(parents=True, exist_ok=True)
    with _BatchFileLock(destination / ".article_prepare.lock"):
        protocol_path = destination / PROTOCOL
        if protocol_path.exists():
            existing = load_article_protocol(destination)
            if {key: existing[key] for key in core} != core:
                raise ValueError("O estudo salvo usa outros dados, modelo ou parâmetros. Escolha uma nova pasta de saída.")
            protocol = existing
        else:
            if any(path.name != ".article_prepare.lock" for path in destination.iterdir()):
                raise ValueError("A saída deve ser uma pasta nova ou um estudo integrado já preparado.")
            protocol = {**core, "created_at_utc": _utc_now()}
            protocol["protocol_sha256"] = _payload_sha256(protocol)
            _atomic_write_json(protocol_path, protocol)
        prepare_experimental_protocol(source, destination / BLOCKS[0], jobs=jobs[BLOCKS[0]],
                                      engine=engine, report=selected, analysis_role="exploratory")
    return protocol


def request_article_cancel(output_dir):
    root = Path(output_dir).resolve()
    load_article_protocol(root)
    _atomic_write_json(root / "article_cancel.json", {"requested_at_utc": _utc_now()})
    for name in BLOCKS:
        if (root / name / "batch_status.json").exists():
            request_batch_cancel(root / name)


def _acknowledge_cancel(root):
    (root / "article_cancel.json").unlink(missing_ok=True)
    for name in BLOCKS:
        clear_batch_cancel_request(root / name)


def run_article_study(output_dir, *, resume=False, engine=None):
    root = Path(output_dir).resolve(strict=True)
    with _BatchFileLock(root / ".article_worker.lock"):
        protocol = load_article_protocol(root)
        started = _utc_now()
        def status(state, block=None, **details):
            _atomic_write_json(root / STATUS, {"state": state, "block": block, "updated_at_utc": _utc_now(),
                "started_at_utc": started, "worker_pid": os.getpid(), "protocol_sha256": protocol["protocol_sha256"], **details})
        try:
            if resume:
                _acknowledge_cancel(root)
            status("VALIDATING")
            engine = engine or DockingPPLocalEngine(timeout_seconds=600)
            if _file_sha256(Path(protocol["artifact_path"])) != protocol["artifact_sha256"]:
                raise ValueError("O modelo mudou depois de preparar o estudo.")
            if _file_sha256(ROOT / "configs/operational_cascade.yaml") != protocol["operational_config_sha256"]:
                raise ValueError("O orçamento mudou depois de preparar o estudo.")
            # Rechecks input hashes, molecular overlap, frozen comparator fit and engine.
            prepare_article_study(protocol["input_dir"], root, pose_source=protocol["pose_source"],
                                  artifact_path=protocol["artifact_path"], engine=engine)
            jobs = study_jobs(protocol["pose_source"])
            discovered = discover_ppi4dock(protocol["input_dir"], validate_contents=True)
            selected_ids = {target["complex_id"] for target in protocol["block_plans"][BLOCKS[0]]["targets"]}
            selected = replace(discovered, targets=tuple(target for target in discovered.targets if target.complex_id in selected_ids), issues=())
            for name in BLOCKS:
                if (root / "article_cancel.json").exists():
                    status("CANCELLED", name)
                    return
                previous = load_batch_status(root / name)
                if previous and previous["state"] in {"COMPLETED", "COMPLETED_WITH_ERRORS"}:
                    continue
                status("RUNNING", name)
                summary = run_batch(protocol["input_dir"], root / name, jobs=jobs[name], engine=engine,
                    report=selected, experimental_protocol=(root / name / "experimental_protocol.json") if name == BLOCKS[0] else None,
                    confidence_artifact=protocol["artifact_path"] if name == BLOCKS[1] else None,
                    shadow_confidence=name == BLOCKS[1])
                if summary.state == "CANCELLED":
                    status("CANCELLED", name)
                    return
            status("ANALYZING")
            from docking_confidence.application.article_reporting import write_article_report
            result = write_article_report(root)
            status("COMPLETED", report=result)
            return result
        except Exception as exc:
            status("FAILED", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            raise


def launch_article_study(output_dir, *, resume=False):
    from docking_confidence.application.batch_launcher import _launch_batch_process
    from docking_confidence.application.batch_monitor import process_alive
    root = Path(output_dir).resolve(strict=True)
    load_article_protocol(root)
    with _BatchFileLock(root / ".article_launch.lock"):
        info_path = root / "article_worker.json"
        if info_path.exists():
            existing = _load_json_object(info_path)
            if process_alive(existing["pid"], existing.get("started_at_utc")) is not False:
                raise ValueError("O estudo já está em execução.")
        # Pin this checkout even when the UI was opened from an unrelated
        # working directory or the environment contains a foreign PYTHONPATH.
        bootstrap = (f"import os,sys,runpy; os.environ.update({THREAD_LIMITS!r}); "
                     f"sys.path.insert(0,{str(ROOT / 'src')!r}); "
                     "runpy.run_module('docking_confidence.application.article_study',run_name='__main__')")
        command = [sys.executable, "-I", "-c", bootstrap, "--output-dir", str(root)]
        if resume:
            # Acknowledge the old request before spawning. A new cancellation
            # during worker imports/preflight must not be erased by the child.
            _acknowledge_cancel(root)
        worker = _launch_batch_process(tuple(command), log_kind="article")
        _atomic_write_json(info_path, {"pid": worker.pid, "log_path": str(worker.log_path), "started_at_utc": _utc_now()})
        return worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_article_study(args.output_dir, resume=args.resume)


if __name__ == "__main__":
    main()

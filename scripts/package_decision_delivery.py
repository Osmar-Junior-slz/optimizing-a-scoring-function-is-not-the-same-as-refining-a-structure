"""Package a completed, audited decision study without running tests or experiments.

No original file is edited. Installed environments, raw docking payloads, other
models and files outside the explicit project allowlist are not distributed.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docking_confidence.application.decision_validation import (
    EVALUATION_SCHEMA, EXPOSURE_SCHEMA, _checked_policy, _read_record, contract_sha256,
)
from docking_confidence.application.generalized_calibrator import load_calibrator
from docking_confidence.application.standard_dataset import verify_standard_quality_records


OLD_MODEL_NAME = "calibrador_gva_2025_hibrido_15_20261117"
NEW_MODEL_NAME = "calibrador_gva_2025_hibrido_15_standard_20261119"
OLD_MODEL_SHA256 = "2b2da73c64590e2c146abac5d94518591236e3d3494a2be8b3ef213e4290787a"
IDENTITY_AUDIT_SHA256 = "085aba600c4d9ea66b017bf41fcf343a520d7ef476796e4dbcf4cf734bade113"
STANDARD_FIT_SHA256 = "8320f34f5119b894f9eacaef636e7331be9a3ded7f5d0fed4c53101bd521fbb2"
STANDARD_FIT_RELATIVE_SOURCE = "../../output/standard_fit_20261119/paired_dataset.csv"
EXCLUDED_PARTS = {".git", ".venv", ".local", "__pycache__", ".pytest_cache", "node_modules"}
EVIDENCE_SUFFIXES = {".json", ".csv", ".yaml", ".yml", ".md", ".txt"}
ARTICLE_SUFFIXES = {".pdf", ".tex", ".bib", ".sty", ".cls", ".bst", ".png", ".jpg", ".jpeg",
    ".svg", ".eps", ".csv", ".tsv", ".json", ".md", ".txt"}
MODEL_CONTENT = {"artifact.json", "resolved_config.json", "validation_report.json", "calibration_examples.csv",
    "training_assignments.csv", "training_cross_validation_predictions.csv", "inputs/dataset.csv"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path):
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def inside_file(path, root):
    """Reject traversal, symlinks/junctions and paths beyond an allowed root."""
    root = Path(root).resolve(strict=True)
    path = Path(path).absolute()
    lexical = path.relative_to(root)
    if ".." in lexical.parts:
        raise ValueError("Source path contains parent traversal")
    current = root
    for component in lexical.parts:
        current = current / component
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ValueError(f"Links/junctions cannot be packaged: {current}")
    resolved = path.resolve(strict=True)
    resolved.relative_to(root)
    if not resolved.is_file():
        raise ValueError(f"Expected a source file: {resolved}")
    return resolved


@dataclass(frozen=True)
class SourceFile:
    source: Path
    destination: str
    category: str
    sha256: str
    size_bytes: int
    snapshot: bytes | None = None
    captured_at_utc: str | None = None


def add_file(plan, source, destination, category, allowed_root=None, *, snapshot=False):
    allowed_root = ROOT if allowed_root is None else allowed_root
    path = inside_file(source, allowed_root)
    relative = PurePosixPath(destination)
    if relative.is_absolute() or ".." in relative.parts or "\\" in destination:
        raise ValueError("Package destination must be a safe relative POSIX path")
    name = relative.as_posix()
    if any(entry.destination.casefold() == name.casefold() for entry in plan):
        raise ValueError(f"Duplicate package path: {name}")
    if snapshot:
        content = path.read_bytes()
        if not isinstance(json.loads(content), dict):
            raise ValueError("Live metadata snapshot must be a complete JSON object")
        plan.append(SourceFile(path, name, category, hashlib.sha256(content).hexdigest(), len(content),
            content, datetime.now(timezone.utc).isoformat()))
    else:
        plan.append(SourceFile(path, name, category, sha256_file(path), path.stat().st_size))


def add_tree(plan, source, destination, category, suffixes, allowed_root=None):
    allowed_root = ROOT if allowed_root is None else allowed_root
    source = Path(source).resolve(strict=True)
    source.relative_to(Path(allowed_root).resolve(strict=True))
    for path in sorted(source.rglob("*")):
        parts = path.relative_to(source).parts
        if (any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in parts)
                or not path.is_file() or path.suffix.lower() not in suffixes):
            continue
        add_file(plan, path, (PurePosixPath(destination) / path.relative_to(source).as_posix()).as_posix(),
            category, allowed_root)


def verified_model(model_dir):
    model_dir = Path(model_dir).resolve(strict=True)
    artifact = inside_file(model_dir / "artifact.json", ROOT / "models")
    if model_dir.name == OLD_MODEL_NAME and sha256_file(artifact) != OLD_MODEL_SHA256:
        raise ValueError("Historical model differs from the fixed original artifact SHA-256")
    loaded = load_calibrator(artifact)
    manifest_path = model_dir / "audit_manifest.json"
    manifest = read_object(manifest_path)
    paths = [manifest_path]
    for entry, is_source in [(entry, False) for entry in manifest["bundle_files"]] + [(entry, True) for entry in manifest["source_inputs"]]:
        relative = Path(entry["path"])
        if (is_source and model_dir.name == NEW_MODEL_NAME
                and relative.as_posix() == STANDARD_FIT_RELATIVE_SOURCE
                and entry.get("sha256") == STANDARD_FIT_SHA256):
            expected = (ROOT / "output/standard_fit_20261119/paired_dataset.csv").resolve(strict=True)
            if (model_dir / relative).resolve(strict=True) != expected:
                raise ValueError("New model source does not resolve to the exact declared standard fitting dataset")
            path = inside_file(expected, ROOT / "output/standard_fit_20261119")
        else:
            if relative.is_absolute() or ".." in relative.parts or relative.as_posix() not in MODEL_CONTENT:
                raise ValueError("Model bundle must contain only model-relative paths or its exact frozen standard source")
            path = inside_file(model_dir / relative, model_dir)
        if path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Model bundle/source hash mismatch: {path}")
        paths.append(path)
    allowed_datasets = {model_dir / "inputs/dataset.csv"}
    if model_dir.name == NEW_MODEL_NAME:
        allowed_datasets.add((ROOT / "output/standard_fit_20261119/paired_dataset.csv").resolve())
    if not allowed_datasets.intersection(paths):
        raise ValueError("Frozen model input dataset is missing from its verified bundle")
    return loaded, paths


def model_bundle_members(model_dir, files):
    """Preserve relative source resolution without parent-traversal ZIP names."""
    model_dir = Path(model_dir).resolve(strict=True)
    project_layout = any(not path.is_relative_to(model_dir) for path in files)
    members = []
    for path in files:
        project_relative = path.relative_to(ROOT).as_posix()
        archive_path = project_relative if project_layout else path.relative_to(model_dir).as_posix()
        if PurePosixPath(archive_path).is_absolute() or ".." in PurePosixPath(archive_path).parts:
            raise ValueError("Model ZIP member must stay inside its normalized archive root")
        members.append({"source_path": path, "package_path": f"sistema/{project_relative}", "archive_path": archive_path})
    return members


def stage_dataset(stage, summary, metric):
    import pandas as pd
    path = inside_file(summary["standard_dataset_path"], stage)
    if sha256_file(path) != summary["standard_dataset_sha256"]:
        raise ValueError("Stage standard dataset changed after its summary")
    frame = pd.read_csv(path, float_precision="round_trip")
    if not frame.empty:
        verify_standard_quality_records(frame, path.parent, metric)
    return path


def checked_exposure_ledger():
    root = ROOT / "output/decision_exposure_ledger"
    records = []
    for path in sorted(root.glob("*.json")) if root.exists() else []:
        path = inside_file(path, root)
        record = _read_record(path, EXPOSURE_SCHEMA)
        identities = [record.get("protocol_sha256"), record.get("dataset_sha256")]
        groups = record.get("groups")
        valid_hash = lambda value: isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
        if (record.get("role") not in {"validation", "test"} or not all(map(valid_hash, identities))
                or not isinstance(groups, list) or not groups or len(groups) != len(set(groups))
                or not all(map(valid_hash, groups))):
            raise ValueError("Global decision exposure ledger has an invalid identity/group record")
        identity = {key: record[key] for key in ("protocol_sha256", "role", "dataset_sha256")}
        if path.stem != contract_sha256(identity):
            raise ValueError("Global exposure ledger filename differs from its committed identity")
        records.append({"path": path, "record": record})
    return records


def check_article(article, qa_path):
    article = Path(article).resolve(strict=True)
    article.relative_to(ROOT)
    qa_path = inside_file(qa_path or article / "qa_final.json", article)
    qa = read_object(qa_path)
    if qa.get("status") != "passed" and qa.get("valid") is not True:
        raise ValueError("Final article QA must explicitly pass")
    pdf_value = qa.get("pdf_path")
    if pdf_value:
        candidate = Path(pdf_value)
        pdf = inside_file(candidate if candidate.is_absolute() else article / candidate, article)
    else:
        pdfs = sorted(article.glob("*.pdf"))
        if len(pdfs) != 1:
            raise ValueError("Article QA must identify its final PDF when multiple/no root PDFs exist")
        pdf = inside_file(pdfs[0], article)
    if pdf.suffix.lower() != ".pdf" or sha256_file(pdf) != qa.get("pdf_sha256"):
        raise ValueError("Final PDF SHA-256 does not match the passed QA")
    if not any(article.rglob("*.tex")) or not any(article.rglob("*.bib")):
        raise ValueError("Final article requires its TEX and BIB sources")
    return {"article_dir": str(article), "qa_path": str(qa_path), "qa_sha256": sha256_file(qa_path),
        "pdf_path": str(pdf), "pdf_sha256": sha256_file(pdf),
        "scientific_status": qa.get("scientific_status"), "decision_capable": qa.get("decision_capable")}


def require_prediction_only_binding():
    binding = yaml.safe_load((ROOT / "configs/calibrator.yaml").read_text(encoding="utf-8"))
    if not isinstance(binding, dict) or binding.get("purpose") != "generalized_quality_prediction":
        raise ValueError("Pending final delivery requires the explicit prediction-only binding purpose")
    path = Path(binding.get("artifact_path", ""))
    artifact = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if artifact != (ROOT / "models" / NEW_MODEL_NAME / "artifact.json").resolve():
        raise ValueError("Prediction-only binding must reference the designated new standard model")


def audit_evidence(args):
    validation = Path(args.validation_stage).resolve(strict=True)
    validation.relative_to(ROOT / "output")
    summary = read_object(validation / "stage_summary.json")
    if summary.get("role") != "validation":
        raise ValueError("Validation stage summary has the wrong role")
    policy_path = inside_file(validation / "decision/decision_policy.json", validation)
    policy, protocol, _ = _checked_policy(policy_path)
    ledger = checked_exposure_ledger()
    expected_model = ROOT / "models" / NEW_MODEL_NAME / "artifact.json"
    if Path(protocol["model"]["artifact_path"]).resolve() != expected_model.resolve():
        raise ValueError("Protocol is not bound to the designated new standard model")
    expected_protocol = ROOT / "output/decision_study_20261119/protocol/decision_protocol.json"
    if Path(policy["protocol_path"]).resolve() != expected_protocol.resolve():
        raise ValueError("Policy is not bound to the designated frozen study protocol")
    if (protocol.get("analysis_role") != "prospective" or protocol.get("prospective_context_issues")
            or summary["status"] != policy["status"] or summary["protocol_sha256"] != protocol["record_sha256"]):
        raise ValueError("Validation summary/prospective policy context is inconsistent")
    development = _read_record(validation / "decision/validation_report.json", "gva_decision_development_report_v1")
    if any(development.get(key) != value for key, value in policy["validation_report"].items()):
        raise ValueError("Validation report differs from the frozen selected policy")
    for key in ("selected", "baseline", "difference", "criteria", "decision_capable"):
        if summary.get(key) != development.get(key):
            raise ValueError(f"Validation stage summary differs from its report: {key}")
    metric = protocol["model"]["source_config"]["quality_metric"]
    data = stage_dataset(validation, summary, metric)
    if sha256_file(data) != policy["validation_source"]["sha256"]:
        raise ValueError("Policy validation source differs from the completed stage")
    final, pending_final, final_report = None, False, None
    if policy["status"] == "selected":
        candidate = Path(args.test_stage).resolve() if args.test_stage else None
        if candidate is not None:
            candidate.relative_to(ROOT / "output")
        complete = candidate is not None and (candidate / "stage_summary.json").is_file()
        if not complete and getattr(args, "allow_pending_final", False):
            default_final = ROOT / "output/decision_test_stage_20261119"
            if candidate is None and (default_final / "stage_summary.json").exists():
                raise ValueError("The final stage already has a summary; supply --test-stage instead of declaring it pending")
            if (policy_path.parent / "final_evaluation_claim.json").exists():
                raise ValueError("Final evaluation has already been consumed; supply its completed stage")
            require_prediction_only_binding()
            pending_final, test_status, decision_capable = True, "final_test_pending", False
        else:
            if not complete:
                raise ValueError("A selected policy requires its completed frozen final-test stage before delivery")
            final = candidate.resolve(strict=True)
            final.relative_to(ROOT / "output")
            final_summary = read_object(final / "stage_summary.json")
            report_path = inside_file(final / "decision/final_evaluation.json", final)
            report = _read_record(report_path, EVALUATION_SCHEMA)
            final_report = report
            claim = read_object(policy_path.parent / "final_evaluation_claim.json")
            if (final_summary.get("role") != "test" or report["policy_sha256"] != policy["record_sha256"]
                    or report["protocol_sha256"] != protocol["record_sha256"]
                    or claim.get("report_sha256") != report["record_sha256"]
                    or Path(claim.get("report_path", "")).resolve() != report_path
                    or final_summary["status"] != report["status"]
                    or final_summary["protocol_sha256"] != protocol["record_sha256"]
                    or report["status"] not in {"passed", "failed"} or not report.get("prospective_eligible")):
                raise ValueError("Final stage is not the linked frozen prospective test")
            for key in ("selected", "baseline", "difference", "criteria", "decision_capable"):
                if final_summary.get(key) != report.get(key):
                    raise ValueError(f"Final stage summary differs from its report: {key}")
            final_data = stage_dataset(final, final_summary, metric)
            if sha256_file(final_data) != report["source"]["sha256"]:
                raise ValueError("Final test source differs from its completed stage")
            test_status, decision_capable = report["status"], report["decision_capable"]
    elif policy["status"] == "no_useful_rule":
        if args.test_stage:
            raise ValueError("No useful rule: omit --test-stage and retain the final test reservation")
        if (ROOT / "runs/decision_test_20261119/study_manifest.json").exists():
            raise ValueError("Final collection already started; its test cannot be described as reserved and unexamined")
        final_groups = set(protocol["roles"]["test_groups"])
        if any((item["record"]["role"] == "test" and item["record"]["protocol_sha256"] == protocol["record_sha256"])
                or bool(final_groups & set(item["record"]["groups"])) for item in ledger):
            raise ValueError("Exposure ledger records the final test groups as opened; the test is not reserved")
        test_status, decision_capable = "reserved_not_collected_no_useful_rule", False
    else:
        raise ValueError("Validation stage must have selected a rule or reported no_useful_rule")
    return {"validation": validation, "test": final, "protocol": protocol,
        "policy": policy, "test_status": test_status, "decision_capable": bool(decision_capable),
        "pending_final": pending_final, "exposure_ledger": ledger,
        "final_report": final_report,
        "delivery_status": "system_ready_final_test_pending" if pending_final else "completed_evidence_delivery"}


def check_standard_fit(root, metric):
    import pandas as pd
    report = read_object(root / "conversion_manifest.json")
    core = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != contract_sha256(core) or not report.get("valid"):
        raise ValueError("Standard fitting conversion report is incomplete or changed")
    data = inside_file(report["dataset_path"], root)
    if sha256_file(data) != report["dataset_sha256"] or report.get("quality_metric") != metric:
        raise ValueError("Standard fitting data/metric differs from its conversion report")
    frame = pd.read_csv(data, float_precision="round_trip")
    if len(frame) != 143:
        raise ValueError("Delivery requires the complete 143-pair standard fitting dataset")
    verified = verify_standard_quality_records(frame, root, metric)
    if verified["records"] != 286:
        raise ValueError("Delivery requires all 286 verified official fitting quality records")


def check_historical_sources(historical, old_article):
    provenance = read_object(historical / "fontes_sha256.json")
    if provenance.get("script_sha256") != sha256_file(historical / "avaliar.py"):
        raise ValueError("Historical replay script differs from its saved provenance hash")
    result = read_object(historical / "resultados.json")
    if (result.get("scope") != "exploratory_out_of_protocol_component_replay"
            or result.get("prospective_validation") is not False or result.get("operationally_valid") is not False):
        raise ValueError("Historical evidence must retain its exploratory out-of-protocol scope")
    rows = result.get("records", [])
    predicted = [row for row in rows if row.get("prediction") is not None]
    separate = [row for row in predicted if row.get("fit_membership") == "not_in_fit"]
    if (len(rows), len(predicted), len(separate), sum(row.get("observed_preserved") is True for row in separate)) != (300, 285, 142, 130):
        raise ValueError("Historical source no longer matches the declared 300/285/142/130 evidence")
    if sha256_file(old_article / "dados/resultados.json") != sha256_file(historical / "resultados.json"):
        raise ValueError("Historical article data snapshot differs from the verified saved replay")


def add_collection(plan, study, label, protocol_sha256, role, stage):
    root = Path(study).resolve(strict=True)
    root.relative_to(ROOT / "runs")
    manifest = read_object(root / "study_manifest.json")
    expected = manifest.get("record_sha256")
    if expected != contract_sha256({key: value for key, value in manifest.items() if key != "record_sha256"}):
        raise ValueError("Collection manifest integrity failed")
    if manifest.get("protocol_sha256") != protocol_sha256 or manifest.get("role") != role:
        raise ValueError("Collection metadata differs from the delivered protocol/role")
    commitment = read_object(Path(stage) / "stage_commitment.json")
    if sha256_file(root / "study_manifest.json") != commitment["study_manifest_sha256"]:
        raise ValueError("Collection manifest changed after stage preparation")
    for source in commitment["source_batches"]:
        batch = Path(source["batch_dir"]).resolve(strict=True)
        batch.relative_to(root)
        if (sha256_file(batch / "batch_manifest.json") != source["manifest_sha256"]
                or sha256_file(batch / "batch_status.json") != source["status_sha256"]):
            raise ValueError("Collection shard metadata changed after stage preparation")
    destination = f"evidencias/coletas/{label}"
    for name in ("study_manifest.json", "collection_summary.json"):
        add_file(plan, root / name, f"{destination}/{name}", "collection_metadata")
    for index, ids in enumerate(manifest["partitions"]):
        if ids:
            for name in ("batch_manifest.json", "batch_status.json"):
                relative = f"shard_{index + 1:02d}/batch/{name}"
                add_file(plan, root / relative, f"{destination}/{relative}", "collection_metadata")


def add_pending_collection(plan, context):
    """Capture live metadata only; never inspect a final CSV or task receipt."""
    root = ROOT / "runs/decision_test_20261119"
    manifest_path = root / "study_manifest.json"
    if not manifest_path.exists():
        return
    manifest = _read_record(manifest_path, "frozen_decision_study_run_v1")
    if (manifest.get("role") != "test" or manifest.get("protocol_sha256") != context["protocol"]["record_sha256"]
            or manifest.get("stage_record_sha256") != context["policy"]["record_sha256"]):
        raise ValueError("Pending collection belongs to another protocol/policy")
    destination = "evidencias/coletas/decision_test_20261119"
    add_file(plan, manifest_path, f"{destination}/study_manifest.json", "pending_collection_manifest")
    for name in ("collection_summary.json", "workers.json"):
        if (root / name).is_file():
            add_file(plan, root / name, f"{destination}/{name}", "pending_collection_snapshot", snapshot=True)
    for index, ids in enumerate(manifest["partitions"]):
        if ids:
            for name in ("batch_manifest.json", "batch_status.json"):
                relative = f"shard_{index + 1:02d}/batch/{name}"
                if (root / relative).is_file():
                    add_file(plan, root / relative, f"{destination}/{relative}", "pending_collection_snapshot", snapshot=True)


def check_pending_article(context, article):
    if context["pending_final"] and (article.get("scientific_status") != "final_test_pending"
            or article.get("decision_capable") is not False):
        raise ValueError("Pending final delivery requires QA scientific_status=final_test_pending and decision_capable=false")
    if context.get("test") is not None and (article.get("scientific_status") != context["test_status"]
            or article.get("decision_capable") is not context["decision_capable"]):
        raise ValueError("Final article QA scientific status and decision_capable must match the completed final report")


def add_quality_reports(plan, protocol):
    root = ROOT / "output/quality_delivery_20261119"
    audit = _read_record(root / "selection_identity_audit.json", "selection_unordered_identity_audit_v1")
    if (audit["record_sha256"] != IDENTITY_AUDIT_SHA256 or audit.get("valid") is not True
            or audit.get("unordered_prior_pair_overlap_count") != 0 or audit.get("prior_complex_id_overlap_count") != 0
            or audit.get("selected_components_verified_from_PDBs") != 200
            or audit["selection_sha256"] != sha256_file(ROOT / "output/decision_study_20261119/selection.json")
            or audit["registry_sha256"] != sha256_file(ROOT / "configs/decision_data_registry.json")):
        raise ValueError("Selection identity audit differs from its frozen passed evidence")
    report = read_object(root / "standard_model_build_report.json")
    if (report.get("report_sha256") != contract_sha256({k: v for k, v in report.items() if k != "report_sha256"})
            or report.get("artifact_sha256") != protocol["model"]["artifact_sha256"]
            or report.get("old_model_sha256_unchanged") != OLD_MODEL_SHA256):
        raise ValueError("Standard model build report is changed or refers to another model")
    comparison = _read_record(root / "fit_metric_comparison.json", "historical_fit_quality_metric_comparison_v1")
    if (comparison.get("row_count") != 143 or comparison.get("same_15_raw_model_descriptors") is not True
            or comparison.get("same_reduced_structural_pose_sha256") is not True
            or comparison.get("fresh_50_50_outcomes_accessed") is not False
            or comparison.get("models_or_collection_modified") is not False):
        raise ValueError("Fitting metric comparison no longer preserves the declared retrospective scope")
    for name in ("selection_identity_audit.json", "audit_selection_identity.py", "standard_model_build_report.json",
                 "fit_source_matches.json", "fit_metric_comparison.json", "compare_fit_quality_metrics.py"):
        add_file(plan, root / name, f"evidencias/quality_delivery_20261119/{name}", "quality_delivery_audit")
    for path in sorted(root.glob("regression*.xml")):
        add_file(plan, path, f"evidencias/quality_delivery_20261119/{path.name}", "previous_test_report")


def add_reduced_only_verification(plan, protocol):
    """Copy the exact saved fitting replay; never rescore or open fresh outcomes."""
    root = ROOT / "output/quality_delivery_20261119"
    exact = root / "red_only_verification_exact"
    report = read_object(exact / "verification.json")
    if (report.get("status") != "passed" or report.get("scope") != "software replay on fitting descriptors; not validation performance"
            or report.get("model_sha256") != protocol["model"]["artifact_sha256"]
            or report.get("reference_or_full_outcomes_supplied") is not False
            or type(report.get("maximum_score_difference")) not in {int, float}
            or report["maximum_score_difference"] != 0 or report.get("rows") != 143 or report.get("complexes") != 49):
        raise ValueError("REDUCED-only software verification does not match the exact saved fitting replay")
    source = ROOT / "output/standard_fit_20261119/paired_dataset.csv"
    if (report.get("source_dataset_sha256") != sha256_file(source)
            or report.get("red_only_input_sha256") != sha256_file(exact / "entrada_reduced.csv")
            or report.get("predictions_sha256") != sha256_file(exact / "prediction_reduced_only.csv")):
        raise ValueError("REDUCED-only verification input/prediction/source hashes changed")
    with (exact / "entrada_reduced.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    if (columns != report.get("input_columns") or len(rows) != 143
            or len({row["complex_fingerprint_sha256"] for row in rows}) != 49
            or any("full" in name.lower() or "dockq" in name.lower()
                   or name.startswith(("quality_", "historical_", "probability", "prediction")) for name in columns)):
        raise ValueError("Exact REDUCED-only example contains unexpected columns or identities")
    for name in ("verification.json", "entrada_reduced.csv", "prediction_reduced_only.csv"):
        add_file(plan, exact / name, f"evidencias/verificacao_reduced/{name}", "reduced_only_software_verification")
    add_file(plan, root / "verify_standard_red_only.py", "evidencias/verificacao_reduced/verify_standard_red_only.py",
        "reduced_only_software_verification")


def build_plan(args):
    output = Path(args.out_dir).resolve()
    output.relative_to(ROOT / "output")
    if output.exists():
        raise FileExistsError("Delivery output must not exist; previous delivery files are never overwritten")
    context = audit_evidence(args)
    article = check_article(args.article_dir, args.article_qa)
    check_pending_article(context, article)
    context["article_main_pdf"] = "artigo/" + Path(article["pdf_path"]).relative_to(Path(article["article_dir"])).as_posix()
    context["article_qa"] = "artigo/" + Path(article["qa_path"]).relative_to(Path(article["article_dir"])).as_posix()
    source_roots = [context["validation"], context["test"], Path(article["article_dir"]),
        ROOT / "output/standard_fit_20261119", ROOT / "output/decision_study_20261119", ROOT / "models",
        ROOT / "output/decision_test_stage_20261119", Path(args.test_stage).resolve() if args.test_stage else None]
    if any(path and (output == path or path in output.parents or output in path.parents) for path in source_roots):
        raise ValueError("Delivery output must be separate from all scientific source directories")
    plan = []
    for name in ("README.md", "iniciar_sistema.cmd", "iniciar_sistema.py", "pyproject.toml",
                 "requirements-baseline.txt", "requirements-dockq-standard.txt"):
        add_file(plan, ROOT / name, f"sistema/{name}", "system")
    for directory, suffixes in (("src", {".py", ".yaml", ".yml", ".json"}),
            ("configs", {".yaml", ".yml", ".json"}), ("scripts", {".py"}), ("tests", {".py"})):
        add_tree(plan, ROOT / directory, f"sistema/{directory}", "system", suffixes)
    add_file(plan, ROOT / "docs/RUN_FROZEN_DECISION_STUDY.md", "sistema/docs/RUN_FROZEN_DECISION_STUDY.md", "system")
    for item in context["exposure_ledger"]:
        path = item["path"]
        add_file(plan, path, f"sistema/output/decision_exposure_ledger/{path.name}", "global_exposure_ledger")
    models = {}
    for name in (OLD_MODEL_NAME, NEW_MODEL_NAME):
        model = ROOT / "models" / name
        loaded, files = verified_model(model)
        if name == NEW_MODEL_NAME and sha256_file(model / "artifact.json") != context["protocol"]["model"]["artifact_sha256"]:
            raise ValueError("New model differs from the frozen protocol")
        members = model_bundle_members(model, files)
        for member in members:
            add_file(plan, member["source_path"], member["package_path"], "frozen_model")
        models[name] = {"artifact_sha256": sha256_file(model / "artifact.json"),
            "verified": loaded["verified"], "bundle_layout": "project_relative" if any(not p.is_relative_to(model) for p in files) else "model_relative",
            "bundle_paths": [member["archive_path"] for member in members],
            "bundle_members": [{key: member[key] for key in ("package_path", "archive_path")} for member in members]}
    selection = ROOT / "output/decision_study_20261119"
    add_tree(plan, selection, "evidencias/decision_study_20261119", "frozen_protocol", {".json"})
    registry = context["protocol"].get("exposure_registry")
    if registry and sha256_file(ROOT / "configs/decision_data_registry.json") != registry["sha256"]:
        raise ValueError("Exposure registry differs from the original frozen protocol snapshot")
    add_file(plan, ROOT / "configs/decision_data_registry.json", "evidencias/decision_data_registry.json", "exposure_registry")
    standard_fit = ROOT / "output/standard_fit_20261119"
    check_standard_fit(standard_fit, context["protocol"]["model"]["source_config"]["quality_metric"])
    add_tree(plan, standard_fit, "evidencias/standard_fit_20261119", "standard_fit", EVIDENCE_SUFFIXES)
    add_quality_reports(plan, context["protocol"])
    add_reduced_only_verification(plan, context["protocol"])
    historical = ROOT / "output/avaliacao_hibrido_100_20261119"
    old_article = ROOT / "output/artigo_completo_20261119"
    check_historical_sources(historical, old_article)
    for name in ("resultados.json", "RELATORIO.md", "fontes_sha256.json", "avaliar.py"):
        add_file(plan, historical / name, f"evidencias/historico_100/{name}", "historical_100_replay")
    add_tree(plan, old_article / "dados", "evidencias/historico_100/dados", "historical_article_data",
        EVIDENCE_SUFFIXES | {".tex", ".tsv"})
    add_file(plan, old_article / "conferir_modelo.py", "evidencias/historico_100/conferir_modelo_original.py", "historical_100_replay")
    for role in ("validation", "test"):
        stage = context[role]
        if stage:
            add_tree(plan, stage, f"evidencias/{stage.name}", f"{role}_stage", EVIDENCE_SUFFIXES)
            add_collection(plan, ROOT / "runs" / f"decision_{role}_20261119", f"decision_{role}_20261119",
                context["protocol"]["record_sha256"], role, stage)
    if context["pending_final"]:
        add_pending_collection(plan, context)
    add_tree(plan, article["article_dir"], "artigo", "article", ARTICLE_SUFFIXES)
    helper_count = 0
    for name in ("gerar_resultados_decisao.py", "diagnostico_calibracao_validacao.py", "gerar_figuras.py"):
        source = Path(article["article_dir"]) / name
        if source.is_file():
            add_file(plan, source, f"artigo/{name}", "article_reproduction_helper")
            helper_count += 1
    if not helper_count:
        raise ValueError("Article delivery requires a recognized reproduction helper")
    return output, plan, context, article, models


def guide_text(context):
    criterion_text = ("ainda não avaliados" if context["pending_final"]
        or context["test_status"] == "reserved_not_collected_no_useful_rule"
        else ("atendidos" if context["decision_capable"] else "não atendidos"))
    pending_notice = ("**ESTUDO FINAL PENDENTE. NENHUMA REGRA ESTÁ AUTORIZADA.**\n"
        "O sistema está pronto para uso e a validação selecionou uma regra, mas o\n"
        "teste final ainda não foi concluído. Este pacote não contém resultados,\n"
        "CSV de rótulos ou relatório final fabricado desse teste. Os metadados de\n"
        "coleta em curso, quando presentes, são snapshots identificados por hash.\n\n"
        if context["pending_final"] else "")
    final_details = ""
    final_report = context.get("final_report")
    if final_report is not None:
        selected = final_report["selected"]
        criteria = context["protocol"]["criteria"]
        percent = lambda value: f"{100 * value:.2f}%".replace(".", ",") if value is not None else "indisponível"
        labels = {"preservation_met": "Preservação entre aceitos", "coverage_met": "Cobertura das comparações planejadas",
            "minimum_complexes_met": "Número mínimo de complexos", "selective_rule": "Seleção de um subconjunto próprio",
            "empirical_baseline_improvement": "Ganho de preservação sobre o comparador", "runtime_met": "Melhoria de tempo"}
        unmet = [labels.get(key, key) for key, value in final_report["criteria"].items() if value is False]
        final_details = (f"Preservação entre aceitos: **{percent(selected['preservation'])}** "
            f"({selected['preserved_runs']}/{selected['accepted_runs']}); requisito estritamente superior a "
            f"{percent(criteria['minimum_preservation'])}.\n"
            f"Cobertura: **{percent(selected['coverage'])}** "
            f"({selected['accepted_runs']}/{selected['planned_runs']}); mínimo predefinido de "
            f"{percent(criteria['minimum_coverage'])}.\n"
            f"Critérios efetivamente não atendidos: {', '.join(unmet) if unmet else 'nenhum'}.\n\n")
    return f"""# Entrega do estudo de decisão

{pending_notice}Estado da validação: {context['policy']['status']}.
Estado do teste: {context['test_status']}.
{final_details}
Critérios empíricos do teste final: {criterion_text}.

`sistema/` contém código, configuração, testes e apenas os dois modelos congelados
declarados. `evidencias/` contém dados oficiais, sidecars de avaliação, protocolos,
relatórios e metadados de coleta. `artigo/` contém o PDF com QA final e suas fontes.
O manuscrito principal desta entrega é `{context.get('article_main_pdf', 'artigo/')}`;
sua revisão de qualidade é `{context.get('article_qa', 'artigo/qa_final.json')}`.
Os materiais históricos em `evidencias/historico_100/` sustentam os resultados
retrospectivos e não substituem este manuscrito acadêmico.
Os testes não foram executados pelo empacotador; os resultados de verificação do
projeto, quando presentes, são os registros previamente produzidos no projeto.
O ledger global de exposição foi preservado em `sistema/output/decision_exposure_ledger/`
quando existente, para conservar o histórico de papéis já examinados.

Crie uma nova `.venv` em `sistema/` e siga o README para instalar o projeto e as
dependências. Ambientes instalados e instaladores não fazem parte desta entrega.
O runtime oficial DockQ deve ser instalado separadamente conforme
`requirements-dockq-standard.txt`; não foi copiado o diretório `.local`.

É possível carregar os bundles e repetir as previsões GVA sobre os CSVs oficiais
incluídos usando `docking_confidence.application.generalized_calibrator.predict_calibrator`,
informando explicitamente os novos caminhos do artifact.json e do CSV. Esse replay
não constitui uma avaliação nova nem permite escolher novamente o limiar no teste.
Há um exemplo REDUCED sem DockQ, FULL ou rótulos em `evidencias/verificacao_reduced/`.
Depois de instalar o projeto, execute este comando a partir da pasta `sistema/`,
escolhendo um nome de saída ainda inexistente:

```powershell
./.venv/Scripts/python.exe -m docking_confidence predict --artifact models/{NEW_MODEL_NAME}/artifact.json --input ../evidencias/verificacao_reduced/entrada_reduced.csv --out replay_reduced_novo.csv
```

O relatório desse exemplo verifica um contrato de software nas 143 linhas de
descritores do ajuste: ponto, limites e escore bruto foram idênticos (diferença máxima registrada
de zero). Ele não mede desempenho independente. Os textos decimais da entrada foram
preservados; a tentativa anterior com arredondamento de leitura/escrita não integra
esta entrega.
O histórico de 100 complexos está em `evidencias/historico_100/`: resultados.json
preserva os 300 casos planejados e os 285 registros com previsões e descritores.
Os 142 casos fora do ajuste e suas 130 preservações pertencem à análise histórica.
Os scripts históricos incluídos são cópias de proveniência e ainda pressupõem o
layout/caminhos originais; sua adaptação deve conservar os hashes e o caráter
exploratório. Os snapshots de tabelas, configurações e fontes ficam em `dados/`.

Protocolos, políticas, recibos e alguns relatórios conservam caminhos absolutos da
execução original. Os originais não foram reescritos: não se garante que todos os
comandos de decisão funcionem apenas ao mover o ZIP. A relocalização exige mapear
os caminhos e revalidar seus hashes; não se deve alterar silenciosamente os registros
congelados. O modelo histórico inclui `inputs/dataset.csv`. O modelo novo conserva
seu manifesto original e a fonte em `sistema/output/standard_fit_20261119/paired_dataset.csv`.
Seu ZIP reproduz a estrutura `models/` e `output/`, preservando essa resolução relativa
sem usar membros de arquivo com `../`.

As poses e payloads brutos de docking permanecem na instalação original e não foram
copiados para este pacote. Recalcular DockQ exige essas poses e as referências
originais com os SHA256 registrados. Os sidecars oficiais e CSVs incluídos permitem
auditar as medições registradas, mas não substituem esses arquivos para recomputação.

Não há garantia populacional de segurança nem alegação de economia operacional
medida do GVA. O overhead do fallback não mede a inferência do modelo atual.
Se não houve regra útil, o teste permanece reservado e não coletado.

`delivery_manifest.json` relaciona SHA256 e tamanho de todos os arquivos de conteúdo.
O hash do próprio manifesto e de cada membro do ZIP é verificado e registrado em
`package_verification.json`, fora do ZIP, evitando uma referência circular de hash.
"""


def verify_zip(archive, expected):
    verified = {}
    with zipfile.ZipFile(archive, "r") as handle:
        names = handle.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("Archive members differ from the audited package inventory")
        for name in sorted(names):
            digest = hashlib.sha256()
            with handle.open(name) as member:
                for block in iter(lambda: member.read(1024 * 1024), b""):
                    digest.update(block)
            actual = digest.hexdigest()
            if actual != expected[name]:
                raise ValueError(f"Archive member SHA256 mismatch: {name}")
            verified[name] = actual
    return verified


def copy_source(item, payload, entries):
    target = payload.joinpath(*PurePosixPath(item.destination).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    if item.snapshot is not None:
        target.write_bytes(item.snapshot)
    else:
        shutil.copyfile(item.source, target)
    if ((item.snapshot is None and sha256_file(item.source) != item.sha256)
            or sha256_file(target) != item.sha256):
        raise ValueError(f"Source or copied file changed during packaging: {item.source}")
    entries.append({"path": item.destination, "sha256": item.sha256,
        "size_bytes": item.size_bytes, "category": item.category,
        "live_metadata_snapshot": item.snapshot is not None, "captured_at_utc": item.captured_at_utc})


def ledger_identity(records):
    return {(str(item["path"]), item["record"]["record_sha256"]) for item in records}


def synchronize_exposure_ledger(plan, payload, entries, context):
    """Include appended immutable events, while rejecting removal or mutation."""
    previous = ledger_identity(context["exposure_ledger"])
    current = checked_exposure_ledger()
    identities = ledger_identity(current)
    if not previous.issubset(identities):
        raise ValueError("Global exposure ledger entries were removed or changed during packaging")
    for event in current:
        if (str(event["path"]), event["record"]["record_sha256"]) not in previous:
            add_file(plan, event["path"],
                f"sistema/output/decision_exposure_ledger/{event['path'].name}", "global_exposure_ledger")
            copy_source(plan[-1], payload, entries)
    context["exposure_ledger"] = current
    return identities


def package_delivery(args):
    output, plan, context, article, models = build_plan(args)
    output.mkdir(parents=True, exist_ok=False)
    payload = output / "conteudo"
    payload.mkdir()
    entries = []
    for item in sorted(plan, key=lambda entry: entry.destination):
        copy_source(item, payload, entries)
    for name, model in models.items():
        relative = f"sistema/models/{name}/bundle_reproduzivel.zip"
        target = payload / relative
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for member in model["bundle_members"]:
                archive.write(payload / member["package_path"], member["archive_path"])
        entries.append({"path": relative, "sha256": sha256_file(target), "size_bytes": target.stat().st_size,
            "category": "generated_verified_model_bundle"})
    guide = payload / "LEIA_ME_ENTREGA.md"
    guide.write_text(guide_text(context), encoding="utf-8")
    entries.append({"path": guide.name, "sha256": sha256_file(guide), "size_bytes": guide.stat().st_size,
        "category": "delivery_guide"})
    readme = payload / "README.md"
    readme.write_text(guide_text(context), encoding="utf-8")
    entries.append({"path": readme.name, "sha256": sha256_file(readme), "size_bytes": readme.stat().st_size,
        "category": "delivery_guide"})
    manifest_path = payload / "delivery_manifest.json"
    archive_path = output / "entrega_decisao.zip"
    initial_ledger_count = len(context["exposure_ledger"])
    for attempt in range(3):
        packaged_ledger = synchronize_exposure_ledger(plan, payload, entries, context)
        manifest = {"schema_version": "frozen_decision_delivery_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_status": context["policy"]["status"], "test_status": context["test_status"],
        "status": context["delivery_status"], "scientific_status": context["test_status"],
        "decision_capable": context["decision_capable"], "automatic_installation": False,
        "tests_executed_by_packager": False, "original_model_expected_sha256": OLD_MODEL_SHA256,
        "models": models, "article_qa": article, "files": sorted(entries, key=lambda entry: entry["path"]),
        "article_main_pdf": context["article_main_pdf"], "article_qa_path": context["article_qa"],
        "hash_inventory_excludes": ["delivery_manifest.json; its hash is recorded in external package_verification.json"],
        "portability": "prediction replay available with relocated explicit paths; frozen protocols retain original absolute paths",
        "raw_docking_payloads": "kept outside delivery; required for DockQ recomputation"}
        manifest["record_sha256"] = contract_sha256(manifest)
        # Only these intermediates in this newly created output may be rewritten
        # when an immutable ledger event was appended during archive creation.
        with manifest_path.open("x" if attempt == 0 else "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
        expected = {entry["path"]: entry["sha256"] for entry in entries}
        expected[manifest_path.name] = sha256_file(manifest_path)
        with zipfile.ZipFile(archive_path, "x" if attempt == 0 else "w",
                compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative in sorted(expected):
                archive.write(payload / relative, relative)
        verified = verify_zip(archive_path, expected)
        current_ledger = ledger_identity(checked_exposure_ledger())
        if not packaged_ledger.issubset(current_ledger):
            raise ValueError("Global exposure ledger entries were removed or changed during archive verification")
        if current_ledger == packaged_ledger:
            ledger_verified_at_utc = datetime.now(timezone.utc).isoformat()
            break
    else:
        raise ValueError("Global exposure ledger did not stabilize after three archive attempts; delivery is not verified")
    if sha256_file(ROOT / "models" / OLD_MODEL_NAME / "artifact.json") != OLD_MODEL_SHA256:
        raise ValueError("Original historical model changed during packaging")
    for item in plan:
        if item.snapshot is None and sha256_file(item.source) != item.sha256:
            raise ValueError(f"Original source changed before final archive verification: {item.source}")
    verification = {"schema_version": "frozen_delivery_archive_verification_v1", "valid": True,
        "status": context["delivery_status"], "decision_capable": context["decision_capable"],
        "archive_path": str(archive_path), "archive_sha256": sha256_file(archive_path),
        "archive_size_bytes": archive_path.stat().st_size, "verified_members": len(verified),
        "manifest_sha256": expected[manifest_path.name], "member_sha256": verified,
        "immutable_source_files_unchanged": True,
        "exposure_ledger_scope": "all validated events present at archive verification; scientific reports remain separately frozen",
        "exposure_ledger_verified_at_utc": ledger_verified_at_utc,
        "exposure_ledger_records": len(packaged_ledger),
        "exposure_events_added_during_packaging": len(packaged_ledger) - initial_ledger_count,
        "live_metadata_snapshots": sum(item.snapshot is not None for item in plan),
        "original_model_sha256": OLD_MODEL_SHA256,
        "tests_executed": False}
    write_new(output / "package_verification.json", verification)
    return {"archive_path": str(archive_path), "archive_sha256": verification["archive_sha256"],
        "verified_members": len(verified), "status": context["delivery_status"], "validation_status": context["policy"]["status"],
        "test_status": context["test_status"], "verification_path": str(output / "package_verification.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "output/entrega_decisao_20261119"))
    parser.add_argument("--article-dir", default=str(ROOT / "output/artigo_revisado_decisao_20261119"))
    parser.add_argument("--validation-stage", default=str(ROOT / "output/decision_validation_stage_20261119"))
    parser.add_argument("--test-stage")
    parser.add_argument("--allow-pending-final", action="store_true", help="Explicitly package a prediction-only system with final_test_pending QA, without opening final test CSVs")
    parser.add_argument("--article-qa", help="Passed final QA JSON with pdf_sha256; defaults to article-dir/qa_final.json")
    args = parser.parse_args(argv)
    result = package_delivery(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

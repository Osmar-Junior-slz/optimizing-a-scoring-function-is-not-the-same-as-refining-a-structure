"""Delivery guards use temporary fixtures; no real delivery is produced here."""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import zipfile

import pytest


_SPEC = importlib.util.spec_from_file_location("decision_delivery_packager", Path(__file__).resolve().parents[1] / "scripts/package_decision_delivery.py")
packager = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = packager
_SPEC.loader.exec_module(packager)


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _record(value):
    value = deepcopy(value)
    value["record_sha256"] = packager.contract_sha256(value)
    return value


def test_allowlist_omits_installed_environments_cache_and_executables(tmp_path):
    source = tmp_path / "source"
    for relative in ("module.py", ".venv/installed.py", ".local/dependency.py", "__pycache__/cache.py", "installer.exe"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    plan = []
    packager.add_tree(plan, source, "sistema/src", "system", {".py"}, tmp_path)
    assert [entry.destination for entry in plan] == ["sistema/src/module.py"]


def test_paths_cannot_escape_authorized_tree_or_collide(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    file = root / "module.py"
    file.write_bytes(b"fixture")
    plan = []
    with pytest.raises(ValueError):
        packager.add_file(plan, root / ".." / "other.py", "file.py", "system", root)
    packager.add_file(plan, file, "File.py", "system", root)
    with pytest.raises(ValueError, match="Duplicate"):
        packager.add_file(plan, file, "file.py", "system", root)


def test_final_article_qa_must_match_actual_pdf_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(packager, "ROOT", tmp_path)
    article = tmp_path / "article"
    article.mkdir()
    pdf = article / "article.pdf"
    pdf.write_bytes(b"final fixture PDF")
    (article / "article.tex").write_text("fixture TEX")
    (article / "article.bib").write_text("fixture BIB")
    qa = article / "qa_final.json"
    _save(qa, {"status": "passed", "pdf_sha256": packager.sha256_file(pdf)})
    assert packager.check_article(article, None)["pdf_sha256"] == packager.sha256_file(pdf)
    pdf.write_bytes(b"later draft")
    with pytest.raises(ValueError, match="PDF SHA"):
        packager.check_article(article, None)


@pytest.fixture
def validation(tmp_path, monkeypatch):
    monkeypatch.setattr(packager, "ROOT", tmp_path)
    stage = tmp_path / "output/validation"
    decision = stage / "decision"
    decision.mkdir(parents=True)
    data = stage / "standard/paired_dataset.csv"
    data.parent.mkdir()
    data.write_text("complex_id\n")
    report = {"status": "selected", "selected": {}, "baseline": {}, "difference": {},
        "criteria": {}, "decision_capable": False}
    policy = {"status": "selected", "record_sha256": "p" * 64,
        "protocol_path": str(tmp_path / "output/decision_study_20261119/protocol/decision_protocol.json"),
        "validation_source": {"sha256": packager.sha256_file(data)}, "validation_report": report}
    protocol = {"record_sha256": "a" * 64, "analysis_role": "prospective", "prospective_context_issues": [],
        "roles": {"validation_groups": ["d" * 64], "test_groups": ["c" * 64]},
        "model": {"artifact_path": str(tmp_path / "models" / packager.NEW_MODEL_NAME / "artifact.json"),
            "source_config": {"quality_metric": {}}}}
    _save(decision / "decision_policy.json", policy)
    _save(decision / "validation_report.json", _record({"schema_version": "gva_decision_development_report_v1", **report}))
    summary = {**report, "role": "validation", "protocol_sha256": protocol["record_sha256"],
        "standard_dataset_path": str(data), "standard_dataset_sha256": packager.sha256_file(data)}
    _save(stage / "stage_summary.json", summary)
    monkeypatch.setattr(packager, "_checked_policy", lambda path: (policy, protocol, {}))
    args = SimpleNamespace(validation_stage=str(stage), test_stage=None)
    return args, policy, protocol, report, summary


def test_selected_policy_requires_completed_final_stage(validation):
    args, *_ = validation
    with pytest.raises(ValueError, match="requires its completed.*final-test"):
        packager.audit_evidence(args)


def test_no_useful_rule_allows_reserved_test_but_not_an_already_started_collection(validation):
    args, policy, protocol, report, summary = validation
    policy["status"] = report["status"] = summary["status"] = "no_useful_rule"
    root = packager.ROOT
    _save(Path(args.validation_stage) / "decision/validation_report.json",
        _record({"schema_version": "gva_decision_development_report_v1", **report}))
    _save(Path(args.validation_stage) / "stage_summary.json", summary)
    context = packager.audit_evidence(args)
    assert context["decision_capable"] is False
    assert context["test_status"] == "reserved_not_collected_no_useful_rule"
    _save(root / "runs/decision_test_20261119/study_manifest.json", {})
    with pytest.raises(ValueError, match="already started"):
        packager.audit_evidence(args)


def test_zip_contents_are_rehashed_and_unknown_members_are_rejected(tmp_path):
    file = tmp_path / "evidence.json"
    file.write_bytes(b"fixture evidence")
    archive = tmp_path / "package.zip"
    with zipfile.ZipFile(archive, "x") as handle:
        handle.write(file, "evidence.json")
    expected = {"evidence.json": packager.sha256_file(file)}
    assert packager.verify_zip(archive, expected) == expected
    with zipfile.ZipFile(archive, "a") as handle:
        handle.writestr("unexpected.exe", b"not allowed")
    with pytest.raises(ValueError, match="members differ"):
        packager.verify_zip(archive, expected)


def test_global_ledger_prevents_claiming_a_custom_final_test_was_reserved(validation):
    args, policy, protocol, report, summary = validation
    policy["status"] = report["status"] = summary["status"] = "no_useful_rule"
    _save(Path(args.validation_stage) / "decision/validation_report.json",
        _record({"schema_version": "gva_decision_development_report_v1", **report}))
    _save(Path(args.validation_stage) / "stage_summary.json", summary)
    identity = {"protocol_sha256": protocol["record_sha256"], "role": "test", "dataset_sha256": "b" * 64}
    record = _record({"schema_version": packager.EXPOSURE_SCHEMA, **identity, "groups": ["c" * 64]})
    path = packager.ROOT / "output/decision_exposure_ledger" / (packager.contract_sha256(identity) + ".json")
    _save(path, record)
    assert len(packager.checked_exposure_ledger()) == 1
    with pytest.raises(ValueError, match="not reserved"):
        packager.audit_evidence(args)
    # Another protocol/role must not conceal prior exposure to the same holdout.
    path.unlink()
    identity.update(protocol_sha256="e" * 64, role="validation")
    path = path.parent / (packager.contract_sha256(identity) + ".json")
    record = _record({"schema_version": packager.EXPOSURE_SCHEMA, **identity, "groups": ["c" * 64]})
    _save(path, record)
    with pytest.raises(ValueError, match="not reserved"):
        packager.audit_evidence(args)
    record["groups"] = ["d" * 64]
    _save(path, record)
    with pytest.raises(ValueError, match="integrity"):
        packager.checked_exposure_ledger()


def test_existing_delivery_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(packager, "ROOT", tmp_path)
    output = tmp_path / "output/existing"
    output.mkdir(parents=True)
    monkeypatch.setattr(packager, "audit_evidence", lambda *a: pytest.fail("existing output must fail before reading evidence"))
    with pytest.raises(FileExistsError):
        packager.build_plan(SimpleNamespace(out_dir=str(output)))


def test_ledger_sync_copies_appended_events_once_and_rejects_removal(tmp_path, monkeypatch):
    monkeypatch.setattr(packager, "ROOT", tmp_path)
    ledger = tmp_path / "output/decision_exposure_ledger"

    def save_event(role, dataset):
        identity = {"protocol_sha256": "a" * 64, "role": role, "dataset_sha256": dataset * 64}
        path = ledger / (packager.contract_sha256(identity) + ".json")
        _save(path, _record({"schema_version": packager.EXPOSURE_SCHEMA, **identity,
            "groups": ["b" * 64]}))
        return path

    first = save_event("validation", "c")
    context = {"exposure_ledger": packager.checked_exposure_ledger()}
    plan, entries = [], []
    packager.add_file(plan, first, f"sistema/output/decision_exposure_ledger/{first.name}", "global_exposure_ledger")
    payload = tmp_path / "new_delivery"
    packager.copy_source(plan[0], payload, entries)
    second = save_event("test", "d")
    identities = packager.synchronize_exposure_ledger(plan, payload, entries, context)
    assert len(identities) == len(plan) == len(entries) == 2
    assert (payload / "sistema/output/decision_exposure_ledger" / second.name).read_bytes() == second.read_bytes()
    packager.synchronize_exposure_ledger(plan, payload, entries, context)
    assert len(plan) == len(entries) == 2
    first.unlink()
    with pytest.raises(ValueError, match="removed or changed"):
        packager.synchronize_exposure_ledger(plan, payload, entries, context)


def test_ledger_sync_rejects_replacement_even_with_valid_new_record_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(packager, "ROOT", tmp_path)
    identity = {"protocol_sha256": "a" * 64, "role": "validation", "dataset_sha256": "c" * 64}
    path = tmp_path / "output/decision_exposure_ledger" / (packager.contract_sha256(identity) + ".json")
    _save(path, _record({"schema_version": packager.EXPOSURE_SCHEMA, **identity, "groups": ["b" * 64]}))
    context = {"exposure_ledger": packager.checked_exposure_ledger()}
    _save(path, _record({"schema_version": packager.EXPOSURE_SCHEMA, **identity, "groups": ["e" * 64]}))
    with pytest.raises(ValueError, match="removed or changed"):
        packager.synchronize_exposure_ledger([], tmp_path / "delivery", [], context)


def test_archive_rebuild_includes_event_appended_during_first_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(packager, "ROOT", tmp_path)
    historical = tmp_path / "models" / packager.OLD_MODEL_NAME / "artifact.json"
    historical.parent.mkdir(parents=True)
    historical.write_bytes(b"unchanged frozen model fixture")
    monkeypatch.setattr(packager, "OLD_MODEL_SHA256", packager.sha256_file(historical))
    ledger = tmp_path / "output/decision_exposure_ledger"

    def event(role, dataset):
        identity = {"protocol_sha256": "a" * 64, "role": role, "dataset_sha256": dataset * 64}
        path = ledger / (packager.contract_sha256(identity) + ".json")
        _save(path, _record({"schema_version": packager.EXPOSURE_SCHEMA, **identity, "groups": ["b" * 64]}))
        return path

    original = event("validation", "c")
    context = {"policy": {"status": "selected"}, "test_status": "final_test_pending", "pending_final": True,
        "decision_capable": False, "delivery_status": "system_ready_final_test_pending",
        "article_main_pdf": "artigo/fixture.pdf", "article_qa": "artigo/qa_final.json",
        "exposure_ledger": packager.checked_exposure_ledger()}
    plan = []
    packager.add_file(plan, original, f"sistema/output/decision_exposure_ledger/{original.name}", "global_exposure_ledger")
    output = tmp_path / "output/new_delivery"
    monkeypatch.setattr(packager, "build_plan", lambda args: (output, plan, context, {}, {}))
    monkeypatch.setattr(packager, "guide_text", lambda context: "Honest fixed pending snapshot")
    actual_verify = packager.verify_zip
    calls = []

    def append_after_verify(archive, expected):
        result = actual_verify(archive, expected)
        calls.append(set(result))
        if len(calls) == 1:
            event("test", "d")
        return result

    monkeypatch.setattr(packager, "verify_zip", append_after_verify)
    result = packager.package_delivery(SimpleNamespace())
    assert len(calls) == 2
    verification = json.loads(Path(result["verification_path"]).read_text(encoding="utf-8"))
    assert verification["valid"] and verification["exposure_ledger_records"] == 2
    assert verification["exposure_events_added_during_packaging"] == 1
    with zipfile.ZipFile(result["archive_path"]) as handle:
        records = [name for name in handle.namelist() if "decision_exposure_ledger/" in name]
        assert len(records) == 2
        manifest = json.loads(handle.read("delivery_manifest.json"))
        assert manifest["scientific_status"] == "final_test_pending"
        assert manifest["decision_capable"] is False


def test_historical_artifact_cannot_silently_change(tmp_path, monkeypatch):
    monkeypatch.setattr(packager, "ROOT", tmp_path)
    model = tmp_path / "models" / packager.OLD_MODEL_NAME
    model.mkdir(parents=True)
    (model / "artifact.json").write_bytes(b"altered model")
    monkeypatch.setattr(packager, "load_calibrator", lambda *a: pytest.fail("changed historical hash must fail first"))
    with pytest.raises(ValueError, match="fixed original artifact"):
        packager.verified_model(model)


def test_explicit_pending_final_never_reads_final_csv_or_authorizes_rule(validation, monkeypatch):
    args, policy, protocol, report, summary = validation
    args.allow_pending_final = True
    partial = packager.ROOT / "output/partial_test_stage"
    poison = partial / "standard/paired_dataset.csv"
    poison.parent.mkdir(parents=True)
    poison.write_bytes(b"must never be read as final labels\xff")
    args.test_stage = str(partial)
    binding = packager.ROOT / "configs/calibrator.yaml"
    binding.parent.mkdir()
    binding.write_text("purpose: generalized_quality_prediction\nartifact_path: models/" + packager.NEW_MODEL_NAME + "/artifact.json\n")
    original = packager.stage_dataset
    reads = []
    def guarded(stage, *rest):
        reads.append(stage)
        assert stage != partial
        return original(stage, *rest)
    monkeypatch.setattr(packager, "stage_dataset", guarded)
    policy_bytes = (Path(args.validation_stage) / "decision/decision_policy.json").read_bytes()
    context = packager.audit_evidence(args)
    assert context["pending_final"] and context["test"] is None
    assert context["decision_capable"] is False
    assert context["delivery_status"] == "system_ready_final_test_pending"
    assert len(reads) == 1
    assert "NENHUMA REGRA ESTÁ AUTORIZADA" in packager.guide_text(context)
    assert (Path(args.validation_stage) / "decision/decision_policy.json").read_bytes() == policy_bytes
    assert not (Path(args.validation_stage) / "decision/final_evaluation_claim.json").exists()
    binding.write_text("purpose: automatic_decision\n")
    with pytest.raises(ValueError, match="prediction-only"):
        packager.audit_evidence(args)


def test_pending_article_must_disclose_unfinished_science():
    context = {"pending_final": True}
    packager.check_pending_article(context, {"scientific_status": "final_test_pending", "decision_capable": False})
    with pytest.raises(ValueError, match="final_test_pending"):
        packager.check_pending_article(context, {"scientific_status": "passed", "decision_capable": True})


def test_final_article_qa_cannot_reuse_pending_or_opposite_final_status():
    context = {"pending_final": False, "test": "completed_stage", "test_status": "failed", "decision_capable": False}
    packager.check_pending_article(context, {"scientific_status": "failed", "decision_capable": False})
    for article in ({"scientific_status": "final_test_pending", "decision_capable": False},
            {"scientific_status": "passed", "decision_capable": True}):
        with pytest.raises(ValueError, match="must match the completed final report"):
            packager.check_pending_article(context, article)


def test_completed_guide_distinguishes_preservation_from_unmet_coverage():
    context = {"pending_final": False, "test_status": "failed", "decision_capable": False,
        "policy": {"status": "selected"},
        "protocol": {"criteria": {"minimum_preservation": .9, "minimum_coverage": .75}},
        "final_report": {"selected": {"preservation": 104 / 110, "preserved_runs": 104,
            "accepted_runs": 110, "planned_runs": 150, "coverage": 110 / 150},
            "criteria": {"preservation_met": True, "coverage_met": False, "minimum_complexes_met": True,
                "selective_rule": True, "empirical_baseline_improvement": True, "runtime_met": None}}}
    text = packager.guide_text(context)
    assert "94,55%" in text and "104/110" in text and "90,00%" in text
    assert "73,33%" in text and "110/150" in text and "75,00%" in text
    failed_line = next(line for line in text.splitlines() if line.startswith("Critérios efetivamente"))
    assert failed_line == "Critérios efetivamente não atendidos: Cobertura das comparações planejadas."


def test_pending_collection_copies_only_metadata_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(packager, "ROOT", tmp_path)
    root = tmp_path / "runs/decision_test_20261119"
    manifest = _record({"schema_version": "frozen_decision_study_run_v1", "role": "test",
        "protocol_sha256": "a" * 64, "stage_record_sha256": "b" * 64, "partitions": [["1abc_AB"]]})
    _save(root / "study_manifest.json", manifest)
    status = root / "shard_01/batch/batch_status.json"
    _save(status, {"state": "RUNNING", "counts": {"succeeded": 1}})
    _save(root / "shard_01/batch/targets/1abc_AB/jobs/run/receipt.json", {"dockq": .99})
    (root / "labels.csv").write_text("private final labels must not be consumed")
    plan = []
    context = {"protocol": {"record_sha256": "a" * 64}, "policy": {"record_sha256": "b" * 64}}
    packager.add_pending_collection(plan, context)
    assert len(plan) == 2
    assert all("receipt" not in item.destination and not item.destination.endswith(".csv") for item in plan)
    captured = next(item for item in plan if item.destination.endswith("batch_status.json"))
    assert captured.snapshot and captured.captured_at_utc
    _save(status, {"state": "RUNNING", "counts": {"succeeded": 2}})
    assert json.loads(captured.snapshot)["counts"]["succeeded"] == 1


@pytest.fixture
def reduced_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(packager, "ROOT", tmp_path)
    root = tmp_path / "output/quality_delivery_20261119"
    exact = root / "red_only_verification_exact"
    exact.mkdir(parents=True)
    source = tmp_path / "output/standard_fit_20261119/paired_dataset.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture original fitting snapshot")
    (root / "verify_standard_red_only.py").write_text("# fixture helper; never executed")
    columns = ["complex_id", "complex_fingerprint_sha256", "viable_fraction_mean"]
    red = exact / "entrada_reduced.csv"
    with red.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for index in range(143):
            writer.writerow([f"complex_{index % 49}", f"{index % 49:064x}", "0.123456789123456789"])
    predictions = exact / "prediction_reduced_only.csv"
    predictions.write_bytes(b"fixture prediction output")
    (exact / "prediction_with_extra_columns.csv").write_bytes(b"must not be copied")
    (root / "red_only_verification").mkdir()
    (root / "red_only_verification/verification.json").write_text("old roundtrip attempt must not be copied")
    report = {"status": "passed", "scope": "software replay on fitting descriptors; not validation performance",
        "model_sha256": "a" * 64, "source_dataset_sha256": packager.sha256_file(source),
        "rows": 143, "complexes": 49, "input_columns": columns,
        "reference_or_full_outcomes_supplied": False, "maximum_score_difference": 0.,
        "red_only_input_sha256": packager.sha256_file(red), "predictions_sha256": packager.sha256_file(predictions)}
    _save(exact / "verification.json", report)
    return root, exact, report


def test_exact_reduced_proof_is_copied_without_old_roundtrip_attempt(reduced_replay):
    root, exact, report = reduced_replay
    plan = []
    original = (exact / "entrada_reduced.csv").read_bytes()
    packager.add_reduced_only_verification(plan, {"model": {"artifact_sha256": "a" * 64}})
    assert len(plan) == 4
    assert {entry.destination for entry in plan} == {
        f"evidencias/verificacao_reduced/{name}" for name in (
            "verification.json", "entrada_reduced.csv", "prediction_reduced_only.csv", "verify_standard_red_only.py")}
    assert (exact / "entrada_reduced.csv").read_bytes() == original
    assert all(entry.snapshot is None for entry in plan)
    report["maximum_score_difference"] = 1e-12
    _save(exact / "verification.json", report)
    with pytest.raises(ValueError, match="exact saved fitting replay"):
        packager.add_reduced_only_verification([], {"model": {"artifact_sha256": "a" * 64}})


def test_reduced_proof_rejects_observed_quality_columns_even_with_rehashed_report(reduced_replay):
    root, exact, report = reduced_replay
    path = exact / "entrada_reduced.csv"
    lines = path.read_text().splitlines()
    path.write_text(lines[0] + ",dockq_full\n" + "\n".join(line + ",0.9" for line in lines[1:]) + "\n")
    report["input_columns"].append("dockq_full")
    report["red_only_input_sha256"] = packager.sha256_file(path)
    _save(exact / "verification.json", report)
    with pytest.raises(ValueError, match="unexpected columns"):
        packager.add_reduced_only_verification([], {"model": {"artifact_sha256": "a" * 64}})


def test_real_standard_model_keeps_external_relative_source_and_replays_after_copy(tmp_path, monkeypatch):
    """Use the frozen local artifact read-only; this is fitting replay, not new evidence."""
    real_root = Path(__file__).resolve().parents[1]
    real_model = real_root / "models" / packager.NEW_MODEL_NAME
    real_source = real_root / "output/standard_fit_20261119/paired_dataset.csv"
    if not (real_model / "artifact.json").is_file() or not real_source.is_file():
        pytest.skip("Optional real frozen standard bundle is unavailable")
    original_manifest = (real_model / "audit_manifest.json").read_bytes()
    original_artifact = (real_model / "artifact.json").read_bytes()
    manifest = json.loads(original_manifest)
    clone = tmp_path / "models" / packager.NEW_MODEL_NAME
    clone.mkdir(parents=True)
    for entry in manifest["bundle_files"]:
        shutil.copyfile(real_model / entry["path"], clone / entry["path"])
    (clone / "audit_manifest.json").write_bytes(original_manifest)
    clone_source = tmp_path / "output/standard_fit_20261119/paired_dataset.csv"
    clone_source.parent.mkdir(parents=True)
    shutil.copyfile(real_source, clone_source)
    monkeypatch.setattr(packager, "ROOT", tmp_path)
    loaded, files = packager.verified_model(clone)
    assert loaded["verified"] and clone_source in files
    members = packager.model_bundle_members(clone, files)
    assert all(".." not in Path(member["archive_path"]).parts for member in members)
    assert any(member["archive_path"] == "output/standard_fit_20261119/paired_dataset.csv" for member in members)
    payload = tmp_path / "delivery"
    for member in members:
        destination = payload / member["package_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(member["source_path"], destination)
    packaged_model = payload / "sistema/models" / packager.NEW_MODEL_NAME / "artifact.json"
    packaged_source = payload / "sistema/output/standard_fit_20261119/paired_dataset.csv"
    assert packager.load_calibrator(packaged_model)["verified"]
    from docking_confidence.application.generalized_calibrator import predict_calibrator
    prediction = predict_calibrator(packaged_model, packaged_source, tmp_path / "replay.csv")
    assert len(prediction) == 143 and prediction.prediction.notna().all()
    assert (clone / "audit_manifest.json").read_bytes() == original_manifest
    assert (real_model / "audit_manifest.json").read_bytes() == original_manifest
    assert (real_model / "artifact.json").read_bytes() == original_artifact
    manifest["source_inputs"][0]["path"] = "../../output/unknown/paired_dataset.csv"
    _save(clone / "audit_manifest.json", manifest)
    with pytest.raises(ValueError, match="exact frozen standard source"):
        packager.verified_model(clone)

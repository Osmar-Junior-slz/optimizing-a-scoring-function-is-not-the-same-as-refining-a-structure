"""Dataset construction is tested against real batch and graph contracts.

The small deterministic engine below only supplies a fixture pose. Batch
receipts, hashes, reference postprocessing and graph descriptors are produced
by the same application code as an actual docking run.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from docking_confidence.adapters.dockingpp_archive_engine import DockingPPReducedError
from docking_confidence.application.batch import (
    _payload_sha256,
    run_batch,
)
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.application.training_dataset import build_training_dataset
from docking_confidence.domain.execution import (
    Algorithm,
    BaseSearchConfig,
    DockingRunResult,
    DockingTask,
    ProtocolMode,
)
from docking_confidence.domain.preservation import PreservationRule
from docking_confidence.evaluation.reference_dockq import read_engine_order_atoms
from docking_confidence.features.contact_graph import CONTACT_GRAPH_METADATA_COLUMNS
from docking_confidence.io.batch_discovery import discover_ppi4dock
from docking_confidence.io.config import config_sha256, load_config
from docking_confidence.io.dataset import load_paired_dataset


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _rewrite_receipt(path: Path, mutate) -> dict:
    payload = _json(path)
    payload.pop("record_sha256")
    mutate(payload)
    payload["record_sha256"] = _payload_sha256(payload)
    _write_json(path, payload)
    return payload


def _component(chain: str, residue_name: str, y: float, serial_start: int) -> str:
    lines = []
    serial = serial_start
    for residue in range(1, 6):
        for atom, dx, dy, dz in (
            ("N", 0.0, 0.0, 0.0),
            ("CA", 1.0, 0.0, 0.0),
            ("C", 1.0, 1.0, 0.0),
            ("O", 0.0, 1.0, 0.0),
        ):
            x = (residue - 1) * 3.0 + dx
            lines.append(
                f"ATOM  {serial:5d} {atom:^4s} {residue_name:>3s} {chain}{residue:4d}    "
                f"{x:8.3f}{y + dy:8.3f}{dz:8.3f}{1.0:6.2f}{20.0:6.2f}"
                f"          {atom[0]:>2s}\n"
            )
            serial += 1
    return "".join(lines)


class _FixturePoseEngine:
    name = "dockingpp_local"
    scientific = True
    supports_artifact_work_dir = True
    requires_reference_postprocess = True

    def __init__(self, failures: set[tuple[str, str]]):
        self.failures = failures

    def audit_descriptor(self):
        return {"adapter_class": "DatasetFixturePoseEngine", "fixture": True}

    def run(self, *, job, protein_path, peptide_path, work_dir):
        if (peptide_path.parent.name, job.protocol.value) in self.failures:
            if job.protocol is ProtocolMode.REDUCED:
                raise DockingPPReducedError("REDUCED_FAILED", "no_valid_reduced_pose; RUN_FULL_REQUIRED")
            raise RuntimeError("fixture FULL failed")
        coords = [list(atom.coord) for atom in read_engine_order_atoms(peptide_path)]
        pose = {
            "schema_version": "1.0",
            "pose_role": "best_computed_pose",
            "search_space_mode": job.protocol.value,
            "engine_name": job.algorithm.value,
            "coords": coords,
            "score_cheap": -1.0,
            "score_expensive": None,
            "final_rescore": None,
            "generation": 1,
            "pocket_id": "local_refinement" if job.profile.docking_task is DockingTask.LOCAL_REFINEMENT else "p1",
            "meta": {"docking_task": job.profile.docking_task.value},
        }
        _write_json(work_dir / "best_pose.json", pose)
        reduced = job.protocol is ProtocolMode.REDUCED
        local = job.profile.docking_task is DockingTask.LOCAL_REFINEMENT
        return DockingRunResult(
            job_id=job.job_id,
            algorithm=job.algorithm.value,
            protocol=job.protocol.value,
            seed=job.seed,
            generations=job.profile.generations,
            population=job.profile.population,
            top_pockets=job.profile.top_pockets,
            evaluation_budget=job.profile.evaluation_budget,
            runtime_sec=2.0 if reduced else 4.0,
            dockq=None,
            actual_eval_count=job.profile.evaluation_budget if job.profile.budget_policy == "fixed_evaluations" else None,
            stop_reason="objective_evaluation_ceiling" if job.profile.budget_policy == "fixed_evaluations" else None,
            scan_gap_norm=None if local else (0.1 if reduced else 901.0),
            viable_fraction_mean=0.75 if reduced else 902.0,
            elite_dispersion=0.2 if reduced else 903.0,
            uncertainty_count=1.0 if reduced else 904.0,
            energy_q20_norm=None if local else (-2.0 if reduced else 905.0),
            pose_artifact_path="best_pose.json",
            engine=self.name,
            scientific=True,
            exit_code=0,
        )


@pytest.fixture
def make_batch(tmp_path):
    serial = 0

    def make(
        *,
        task="local_refinement",
        target_count=1,
        algorithms=(Algorithm.AG,),
        seeds=(1,),
        protocol=ProtocolMode.REDUCED_VS_FULL,
        duplicate_fingerprint=False,
        duplicate_job=False,
        mixed_tasks=False,
        failures=(),
    ):
        nonlocal serial
        serial += 1
        base = tmp_path / f"fixture_{serial}"
        inputs = base / "inputs"
        inputs.mkdir(parents=True)
        for index in range(target_count):
            complex_id = f"{index + 1}abc_AB"
            directory = inputs / complex_id
            directory.mkdir()
            receptor = _component("A", "ALA", 0.0, 1)
            ligand_residue = "GLY" if duplicate_fingerprint else ("GLY", "SER", "LEU")[index]
            ligand = _component("B", ligand_residue, 3.0, 21)
            (directory / "A_model_st.pdb").write_text(receptor + "END\n", encoding="utf-8")
            (directory / "B_model_st.pdb").write_text(ligand + "END\n", encoding="utf-8")
            (directory / f"{complex_id}_st.pdb").write_text(receptor + ligand + "END\n", encoding="utf-8")
        report = discover_ppi4dock(inputs, validate_contents=True)
        jobs = build_execution_plan(
            algorithms=algorithms,
            protocol_mode=protocol,
            base_config=BaseSearchConfig(2, 4, 2, DockingTask(task)),
            seeds=seeds,
        )
        if duplicate_job:
            jobs.append(replace(jobs[0], job_id=jobs[0].job_id + "-replica"))
        if mixed_tasks:
            jobs[-1] = replace(jobs[-1], profile=replace(jobs[-1].profile, docking_task=DockingTask.SURFACE_DISCOVERY))
        output = base / "batch"
        run_batch(inputs, output, jobs=jobs, engine=_FixturePoseEngine(set(failures)), report=report)
        return output

    return make


def _receipt(batch: Path, *, complex_id="1abc_AB", protocol="reduced", algorithm="ag", seed=1) -> Path:
    return batch / "targets" / complex_id / "jobs" / f"{algorithm}-{protocol}-seed{seed}" / "receipt.json"


def _build(batch: Path, *, profile="pilot", fingerprint_alias_policy="reject", progress_callback=None) -> dict:
    return build_training_dataset(
        batch, batch.parent / "prepared", profile=profile,
        fingerprint_alias_policy=fingerprint_alias_policy,
        progress_callback=progress_callback,
    )


def _assert_invalid(result: dict) -> None:
    assert result["valid"] is False
    assert result["dataset_path"] is None
    assert result["counts"]["usable_rows"] == 0
    assert result["issues"]
    assert Path(result["report_path"]).is_file()


@pytest.mark.parametrize(
    ("task", "schema", "feature_count", "config_name"),
    [
        ("local_refinement", "local_refinement_decision_features_v1", 11, "local_refinement_pilot.yaml"),
        ("surface_discovery", "reduced_decision_features_v2", 13, "pilot.yaml"),
    ],
)
def test_builds_protocol_specific_paired_csv_and_validates_official_loader(
    make_batch, task, schema, feature_count, config_name
):
    batch = make_batch(task=task)
    result = _build(batch)
    assert result["valid"] is True, result["issues"]
    assert result["docking_task"] == task
    assert result["feature_schema"] == schema
    assert len(result["feature_columns"]) == feature_count
    assert Path(result["config_path"]).name == config_name
    config = load_config(result["config_path"])
    config["docking_task"] = task
    assert result["config_sha256"] == config_sha256(config)
    frame = load_paired_dataset(
        result["dataset_path"], result["feature_columns"], config["contact_graph"],
        docking_task=task, feature_schema=schema,
    )
    assert len(frame) == result["counts"]["valid_pairs"] == result["counts"]["usable_rows"] == 1
    assert frame["docking_task"].tolist() == [task]
    assert frame["feature_schema"].tolist() == [schema]
    assert frame["config_sha256"].tolist() == [result["config_sha256"]]
    assert set(CONTACT_GRAPH_METADATA_COLUMNS) <= set(frame.columns)
    assert frame.loc[0, "run_id"] == "ag-reduced-seed1"
    assert frame.loc[0, "graph_bound_run_id"] == frame.loc[0, "run_id"]
    assert frame.loc[0, "engine_reduced"] == frame.loc[0, "engine_full"] == "dockingpp_local"
    assert frame.loc[0, "runtime_reduced_sec"] == 2.0
    assert frame.loc[0, "runtime_full_sec"] == 4.0
    assert frame.loc[0, "viable_fraction_mean"] == 0.75  # FULL fixture value is 902.
    assert frame.loc[0, "elite_dispersion"] == 0.2
    assert frame.loc[0, "uncertainty_count"] == 1.0
    assert not any("full" in name.lower() or "dockq" in name.lower() for name in result["feature_columns"])
    if task == "local_refinement":
        assert "scan_gap_norm" not in result["feature_columns"]
        assert "energy_q20_norm" not in result["feature_columns"]
    else:
        assert frame.loc[0, "scan_gap_norm"] == 0.1
        assert frame.loc[0, "energy_q20_norm"] == -2.0
    dataset = Path(result["dataset_path"])
    assert dataset.name == "paired_dataset.csv"
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == result["dataset_sha256"]
    assert _json(Path(result["report_path"]))["dataset_sha256"] == result["dataset_sha256"]


def test_pairs_separately_by_fingerprint_algorithm_and_seed(make_batch):
    batch = make_batch(target_count=2, algorithms=(Algorithm.AG, Algorithm.PSO), seeds=(1, 7))
    events = []
    def progress(fraction, message):
        events.append((fraction, message))
        if fraction == 1.0:
            assert (batch.parent / "prepared/dataset_build_report.json").is_file()
    result = _build(batch, progress_callback=progress)
    assert result["valid"] is True, result["issues"]
    frame = pd.read_csv(result["dataset_path"])
    keys = ["complex_fingerprint_sha256", "algorithm", "seed"]
    assert len(frame) == 8
    assert not frame.duplicated(keys).any()
    assert frame["complex_fingerprint_sha256"].nunique() == 2
    assert set(frame["algorithm"]) == {"ag", "pso"}
    assert set(frame["seed"]) == {1, 7}
    assert result["counts"]["independent_complexes"] == 2
    fractions = [fraction for fraction, _ in events]
    assert fractions == sorted(fractions)
    assert fractions[0] == 0.0 and fractions[-1] == 1.0
    assert sum(message.startswith("Validando execução") for _, message in events) == 16
    assert sum(message.startswith("Validando par ") for _, message in events) == 8
    assert any("8/8" in message for _, message in events)


def test_labels_preview_using_existing_preservation_rule(make_batch):
    batch = make_batch(target_count=2)
    _rewrite_receipt(_receipt(batch, protocol="full"), lambda r: r["engine_result"].update(dockq=0.8))
    _rewrite_receipt(_receipt(batch), lambda r: r["engine_result"].update(dockq=0.6))
    result = _build(batch)
    assert result["valid"] is True, result["issues"]
    frame = pd.read_csv(result["dataset_path"])
    rule = PreservationRule()
    labels = [rule.label(row.dockq_reduced, row.dockq_full) for row in frame.itertuples()]
    preview = result["preview"]
    assert preview["preserved_count"] == sum(labels) == 1
    assert preview["non_preserved_count"] == len(labels) - sum(labels) == 1


def test_duplicate_fingerprint_aliases_invalidate_without_silent_deduplication(make_batch):
    batch = make_batch(target_count=2, duplicate_fingerprint=True)
    result = _build(batch)
    _assert_invalid(result)
    assert result["counts"]["complexes_found"] == 2
    assert result["counts"]["independent_complexes"] == 1
    assert result["counts"]["ambiguous_pairs"] >= 1
    assert result["counts"]["provenance_errors"] >= 1


def test_duplicate_protocol_jobs_cannot_choose_a_convenient_replica(make_batch):
    result = _build(make_batch(duplicate_job=True))
    _assert_invalid(result)
    assert result["counts"]["ambiguous_pairs"] >= 1


def test_explicit_alias_resolution_keeps_whole_target_and_original_provenance(make_batch):
    batch = make_batch(target_count=2, duplicate_fingerprint=True,
                       algorithms=(Algorithm.AG, Algorithm.PSO), seeds=(1, 7))
    before = {p.relative_to(batch): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in batch.rglob("*") if p.is_file()}
    result = _build(batch, fingerprint_alias_policy="first_complex_id")
    assert result["valid"] is True, result["issues"]
    frame = pd.read_csv(result["dataset_path"])
    assert len(frame) == 4
    assert set(frame.complex_id) == set(frame.graph_bound_complex_id) == {"1abc_AB"}
    assert not frame.duplicated(["complex_fingerprint_sha256", "algorithm", "seed"]).any()
    assert result["counts"]["resolved_alias_groups"] == 1
    assert result["counts"]["excluded_alias_targets"] == 1
    assert result["counts"]["excluded_alias_planned_runs"] == 8
    assert result["counts"]["reduced_succeeded"] == 8  # Includes the audited excluded target.
    assert result["counts"]["ambiguous_pairs"] == result["counts"]["provenance_errors"] == 0
    assert result["alias_resolutions"][0]["selected_complex_id"] == "1abc_AB"
    assert result["alias_resolutions"][0]["excluded_complex_ids"] == ["2abc_AB"]
    assert result["warnings"][0]["code"] == "fingerprint_alias_resolved"
    after = {p.relative_to(batch): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in batch.rglob("*") if p.is_file()}
    assert after == before


def test_alias_selection_does_not_prefer_a_better_dockq_or_runtime(make_batch):
    batch = make_batch(target_count=2, duplicate_fingerprint=True)
    _rewrite_receipt(_receipt(batch), lambda r: r["engine_result"].update(dockq=0.0, runtime_sec=999.0))
    result = _build(batch, fingerprint_alias_policy="first_complex_id")
    assert result["valid"] is True, result["issues"]
    frame = pd.read_csv(result["dataset_path"])
    assert frame.complex_id.tolist() == ["1abc_AB"]
    assert frame.dockq_reduced.tolist() == [0.0]
    assert frame.runtime_reduced_sec.tolist() == [999.0]
    assert result["preview"]["non_preserved_count"] == 1


@pytest.mark.parametrize("protocol", ["full", "reduced"])
def test_failed_alias_representative_is_not_replaced_by_successful_alias(make_batch, protocol):
    batch = make_batch(target_count=2, duplicate_fingerprint=True, failures=(("1abc_AB", protocol),))
    result = _build(batch, fingerprint_alias_policy="first_complex_id")
    _assert_invalid(result)  # The selected target has no complete pair.
    assert result["counts"]["candidate_rows"] == 0
    assert result["counts"]["unmatched_pairs"] == 1
    assert result["unmatched"][0]["complex_id"] == "1abc_AB"
    assert result["unmatched"][0]["missing"] == [protocol]
    assert result["alias_resolutions"][0]["selected_complex_id"] == "1abc_AB"
    assert result["failures"][0]["excluded_by_alias_policy"] is False


def test_excluded_alias_failure_remains_in_audit(make_batch):
    batch = make_batch(target_count=2, duplicate_fingerprint=True, failures=(("2abc_AB", "reduced"),))
    result = _build(batch, fingerprint_alias_policy="first_complex_id")
    assert result["valid"] is True, result["issues"]
    assert result["counts"]["reduced_failed"] == 1
    assert result["failures"][0]["complex_id"] == "2abc_AB"
    assert result["failures"][0]["excluded_by_alias_policy"] is True


def test_alias_resolution_does_not_bypass_excluded_receipt_integrity(make_batch):
    batch = make_batch(target_count=2, duplicate_fingerprint=True)
    receipt_path = _receipt(batch, complex_id="2abc_AB")
    receipt = _json(receipt_path)
    receipt["engine_result"]["dockq"] = 0.01
    _write_json(receipt_path, receipt)  # Hash is intentionally not updated.
    result = _build(batch, fingerprint_alias_policy="first_complex_id")
    _assert_invalid(result)
    assert any(issue["code"] == "receipt_provenance" and issue["complex_id"] == "2abc_AB"
               for issue in result["issues"])


def test_alias_resolution_still_rejects_duplicate_protocol_jobs(make_batch):
    batch = make_batch(target_count=2, duplicate_fingerprint=True, duplicate_job=True)
    result = _build(batch, fingerprint_alias_policy="first_complex_id")
    _assert_invalid(result)
    assert result["counts"]["ambiguous_pairs"] == 1


def test_unsupported_alias_policy_fails_before_creating_snapshot(tmp_path):
    with pytest.raises(ValueError, match="fingerprint_alias_policy"):
        build_training_dataset(tmp_path / "absent_batch", tmp_path / "prepared",
                               fingerprint_alias_policy="best_dockq")
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("protocol", [ProtocolMode.FULL, ProtocolMode.REDUCED])
def test_reports_absent_protocol_and_does_not_create_training_rows(make_batch, protocol):
    result = _build(make_batch(protocol=protocol))
    _assert_invalid(result)
    assert result["counts"]["unmatched_pairs"] == 1
    assert result["counts"]["valid_pairs"] == 0


def test_reduced_hard_failure_is_reported_separately_from_valid_pairs(make_batch):
    result = _build(make_batch(target_count=2, failures=(("2abc_AB", "reduced"),)))
    assert result["valid"] is True, result["issues"]
    assert result["counts"]["reduced_succeeded"] == 1
    assert result["counts"]["reduced_failed"] == 1
    assert result["counts"]["valid_pairs"] == result["counts"]["usable_rows"] == 1
    frame = pd.read_csv(result["dataset_path"])
    assert frame["complex_id"].tolist() == ["1abc_AB"]
    serialized = json.dumps(result)
    assert "deterministic_fallback" in serialized
    assert "reduced_hard_failure" in serialized
    assert "2abc_AB" in serialized


def test_missing_full_for_one_complex_is_explicit_and_other_pair_remains_usable(make_batch):
    result = _build(make_batch(target_count=2, failures=(("2abc_AB", "full"),)))
    assert result["valid"] is True, result["issues"]
    assert result["counts"]["unmatched_pairs"] == 1
    assert result["counts"]["usable_rows"] == 1
    assert result["counts"]["reduced_failed"] == 0
    assert "2abc_AB" in json.dumps(result)


def test_successful_receipt_takes_precedence_over_old_failed_attempt(make_batch):
    batch = make_batch()
    receipt = _receipt(batch)
    _write_json(receipt.parent / "error.json", {
        "status": "FAILED", "engine_status": "REDUCED_FAILED",
        "complex_id": "1abc_AB", "job_id": "ag-reduced-seed1",
        "message": "prior attempt", "failure_reason": "no_valid_reduced_pose",
    })
    result = _build(batch)
    assert result["valid"] is True, result["issues"]
    assert result["counts"]["reduced_failed"] == 0
    assert result["counts"]["usable_rows"] == 1


@pytest.mark.parametrize("field", ["viable_fraction_mean", "elite_dispersion", "uncertainty_count"])
def test_missing_required_local_feature_invalidates_all_rows(make_batch, field):
    batch = make_batch(target_count=2)
    _rewrite_receipt(_receipt(batch), lambda r: r["engine_result"].update({field: None}))
    result = _build(batch)
    _assert_invalid(result)
    assert result["counts"]["missing_features"] >= 1
    assert field in json.dumps(result)
    assert result["candidate_dataset_path"] is not None


@pytest.mark.parametrize("field", ["scan_gap_norm", "energy_q20_norm"])
def test_surface_schema_does_not_impute_missing_scan_feature(make_batch, field):
    batch = make_batch(task="surface_discovery")
    _rewrite_receipt(_receipt(batch), lambda r: r["engine_result"].update({field: None}))
    result = _build(batch)
    _assert_invalid(result)
    assert result["counts"]["missing_features"] >= 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("graph_bound_run_id", "another-run"), ("graph_pose_protocol", "full"),
     ("graph_config_sha256", "0" * 64), ("graph_interface_density", 0.999)],
)
def test_rejects_incompatible_or_modified_contact_graph(make_batch, field, value):
    batch = make_batch()
    receipt_path = _receipt(batch)
    graph_path = receipt_path.parent / _json(receipt_path)["engine_result"]["graph_features_path"]
    graph = _json(graph_path)
    graph[field] = value
    _write_json(graph_path, graph)
    result = _build(batch)
    _assert_invalid(result)
    assert result["counts"]["provenance_errors"] >= 1


def test_graph_pdb_raw_hash_is_checked_against_persisted_pose(make_batch):
    batch = make_batch()
    receipt_path = _receipt(batch)
    pdb = receipt_path.parent / _json(receipt_path)["engine_result"]["complex_pose_pdb_path"]
    pdb.write_text(pdb.read_text(encoding="utf-8") + "REMARK altered after graph\n", encoding="utf-8")
    result = _build(batch)
    _assert_invalid(result)
    assert result["counts"]["provenance_errors"] >= 1


def test_missing_required_graph_feature_is_counted_without_imputation(make_batch):
    batch = make_batch()
    receipt_path = _receipt(batch)
    graph_path = receipt_path.parent / _json(receipt_path)["engine_result"]["graph_features_path"]
    graph = _json(graph_path)
    graph["graph_interface_density"] = None
    _write_json(graph_path, graph)
    result = _build(batch)
    _assert_invalid(result)
    assert result["counts"]["missing_features"] == 1
    assert "graph_interface_density" in json.dumps(result["missing_feature_details"])


def test_receipt_metadata_cannot_claim_a_different_algorithm(make_batch):
    batch = make_batch()
    _rewrite_receipt(_receipt(batch, protocol="full"), lambda r: r["engine_result"].update(algorithm="pso"))
    result = _build(batch)
    _assert_invalid(result)
    assert result["counts"]["provenance_errors"] >= 1


def test_full_target_incompatible_with_manifest_is_rejected(make_batch):
    batch = make_batch()
    _rewrite_receipt(_receipt(batch, protocol="full"), lambda r: r["target"].update(reference_sha256="0" * 64))
    result = _build(batch)
    _assert_invalid(result)
    assert result["counts"]["provenance_errors"] >= 1


def test_mixed_local_and_surface_tasks_are_not_a_valid_training_contract(make_batch):
    result = _build(make_batch(mixed_tasks=True))
    _assert_invalid(result)


def test_graph_path_cannot_escape_its_job_directory(make_batch):
    batch = make_batch()
    _rewrite_receipt(_receipt(batch), lambda r: r["engine_result"].update(graph_features_path="../../outside.json"))
    result = _build(batch)
    _assert_invalid(result)
    assert result["counts"]["provenance_errors"] >= 1


def test_tampered_manifest_is_reported_without_accepting_receipts(make_batch):
    batch = make_batch()
    path = batch / "batch_manifest.json"
    manifest = _json(path)
    manifest["plan"]["targets"][0]["ligand_sha256"] = "0" * 64
    _write_json(path, manifest)
    result = _build(batch)
    _assert_invalid(result)
    assert result["counts"]["provenance_errors"] >= 1


def test_preparation_does_not_overwrite_existing_dataset(make_batch):
    batch = make_batch()
    first = _build(batch)
    assert first["valid"] is True
    dataset = Path(first["dataset_path"])
    original = dataset.read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        _build(batch)
    assert dataset.read_bytes() == original


@pytest.mark.parametrize("task", ["local_refinement", "surface_discovery"])
def test_definitive_profile_keeps_scientific_gates_and_preservation(make_batch, task):
    result = _build(make_batch(task=task), profile="default")
    assert result["valid"] is True, result["issues"]
    config = load_config(result["config_path"])
    assert Path(result["config_path"]).name == ("local_refinement_default.yaml" if task == "local_refinement" else "default.yaml")
    assert config["preservation"] == {"minimum_reduced_dockq": 0.23, "maximum_dockq_loss": 0.05}
    assert config["authorization"]["allow_automatic_decision"] is False
    assert config["calibration"]["minimum_independent_complexes"] >= 10

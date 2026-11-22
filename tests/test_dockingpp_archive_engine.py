from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
import zipfile

import pytest
import yaml

from docking_confidence.adapters.dockingpp_archive_engine import (
    CLI_MODULE,
    DockingPPArchiveEngine,
    DockingPPArchiveIntegrityError,
    DockingPPExecutionError,
    DockingPPOutputError,
    DockingPPReducedError,
)
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import (
    Algorithm,
    BaseSearchConfig,
    ExecutionJob,
    ProtocolMode,
)


_PREFIX = "DockingReduce_V5"
_CONFIG_ENTRY = f"{_PREFIX}/config/default.yaml"
_BASE_CONFIG = b"legacy_option: preserved\nconfidence_policy:\n  enabled: true\n"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_archive(path: Path, *, unsafe_member: str | None = None) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{_PREFIX}/dockingpp/__init__.py", "__version__ = '0.1.0'\n")
        archive.writestr(
            f"{_PREFIX}/dockingpp/adapters/inbound/cli.py",
            "# fake module; subprocess execution is mocked\n",
        )
        archive.writestr(
            f"{_PREFIX}/pyproject.toml",
            "[project]\nname='dockingpp'\nversion='0.1.0'\n",
        )
        archive.writestr(_CONFIG_ENTRY, _BASE_CONFIG)
        if unsafe_member is not None:
            archive.writestr(unsafe_member, "forbidden")


def _job(
    *, algorithm: Algorithm = Algorithm.PSO, protocol: ProtocolMode = ProtocolMode.REDUCED
) -> ExecutionJob:
    return build_execution_plan(
        algorithms=[algorithm],
        protocol_mode=protocol,
        base_config=BaseSearchConfig(generations=40, population=12, top_pockets=8),
        seeds=[17],
    )[0]


def test_unrecognized_archive_keeps_original_uninstrumented_sources(tmp_path):
    archive = tmp_path / "other.zip"
    _make_archive(archive)
    engine = DockingPPArchiveEngine(archive, tmp_path / "snap", config_entry=_CONFIG_ENTRY)
    assert engine.engine_patch is None
    assert (engine._ensure_snapshot() / "dockingpp/__init__.py").read_bytes() == b"__version__ = '0.1.0'\n"
    with pytest.raises(DockingPPArchiveIntegrityError, match="recognized DockingPP"):
        DockingPPArchiveEngine(archive, tmp_path / "snap", config_entry=_CONFIG_ENTRY, seed_lineage=True)


def _inputs(root: Path, *, mobile_partner_residues: int = 5) -> tuple[Path, Path]:
    receptor = root / "receptor.pdb"
    peptide = root / "peptide.pdb"
    receptor.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n", encoding="ascii")
    peptide.write_text(
        "".join(
            f"ATOM  {index:5d}  CA  GLY B{index:4d}    "
            f"{float(index):8.3f}{0.0:8.3f}{0.0:8.3f}\n"
            for index in range(1, mobile_partner_residues + 1)
        ),
        encoding="ascii",
    )
    return receptor, peptide


def _install_fake_success(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, Any]],
    *,
    fallback: bool = False,
    malformed_pose: bool = False,
    undefined_scan_gap: bool = False,
    raw_structural_metrics: dict | None = None,
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"command": command, **kwargs})
        kwargs["stdout"].write("fake dockingpp stdout\n")
        kwargs["stderr"].write("fake dockingpp stderr\n")
        config_path = Path(command[command.index("--config") + 1])
        output_dir = Path(command[command.index("--out") + 1])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        mode = config["search_space_mode"]
        relative_artifacts = (
            Path("pocket-1") / "artifacts"
            if mode == "reduced"
            else Path("artifacts")
        )
        artifacts = output_dir / relative_artifacts
        artifacts.mkdir(parents=True)
        confidence_features = {
            "scan_overview": {
                "scan_score_gap_top1_top2_per_peptide_atom": 0.41,
            },
            "run": {
                "popconf_best_pocket_viable_fraction_mean": 0.72,
                "popconf_best_pocket_elite_coords_dispersion_last": 0.18,
                "popconf_best_pocket_uncertainty_count": 2,
            },
            "best_pocket": {
                "features": {"energy_q20_per_peptide_atom": -4.7},
            },
        }
        if undefined_scan_gap:
            confidence_features["scan_overview"][
                "scan_score_gap_top1_top2_per_peptide_atom"
            ] = None
        summary = {
            "schema_version": "2.0",
            "mode": "reduced_aggregate" if mode == "reduced" else "full",
            "search_space_mode": mode,
            "runtime_sec": 2.5,
            "fallback_to_full": fallback,
            "fallback_executed": False,
            "final_pose_available": True,
            "pose_output_path": (relative_artifacts / "best_pose.json").as_posix(),
        }
        if mode == "reduced":
            summary["confidence_features_version"] = "1.3"
            summary["confidence_features"] = confidence_features
        result = {
            "schema_version": "2.0",
            "mode": mode,
            "search_space_mode": mode,
            "runtime_sec": 2.5,
            "fallback_to_full": fallback,
            "fallback_executed": False,
        }
        pose = {
            "schema_version": "1.0",
            "pose_role": "wrong" if malformed_pose else "best_computed_pose",
            "search_space_mode": mode,
            "engine_name": config["search_engine"],
            "coords": [
                [float(index), 2.0, 3.0]
                for index in range(1, 6)
            ],
            "score_cheap": -1.0,
            "score_expensive": -2.0,
            "final_rescore": -2.1,
            "pocket_id": "pocket-1",
            "generation": 3,
            "meta": {"docking_task": config["docking_task"],
                     "search_anchor_mode": "input_centroid",
                     "input_relative_placement_used": True},
        }
        (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (output_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        (output_dir / "metrics.jsonl").write_text('{"generation": 1}\n', encoding="utf-8")
        (artifacts / "best_pose.json").write_text(json.dumps(pose), encoding="utf-8")
        (artifacts / "structural_metrics.json").write_text(
            json.dumps(raw_structural_metrics if raw_structural_metrics is not None
                       else {"schema_version": "1.0", "metrics": {"dockq": 0.0}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "docking_confidence.adapters.dockingpp_archive_engine.subprocess.run",
        fake_run,
    )


def _engine(tmp_path: Path) -> tuple[DockingPPArchiveEngine, Path, Path]:
    archive = tmp_path / "DockingReduce_V5.zip"
    _make_archive(archive)
    config_sha = hashlib.sha256(_BASE_CONFIG).hexdigest()
    snapshot_dir = tmp_path / "snapshots"
    engine = DockingPPArchiveEngine(
        archive,
        snapshot_dir,
        config_entry=_CONFIG_ENTRY,
        expected_archive_sha256=_sha(archive),
        expected_config_sha256=config_sha,
        python_executable="python-for-test",
        timeout_seconds=30,
    )
    return engine, archive, snapshot_dir


def test_pinned_archive_runs_cli_lazily_and_returns_relative_audited_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    engine, archive, snapshot_dir = _engine(tmp_path)
    receptor, peptide = _inputs(tmp_path)
    calls: list[dict[str, Any]] = []
    _install_fake_success(monkeypatch, calls)

    descriptor = engine.audit_descriptor()
    assert descriptor["archive_sha256"] == _sha(archive)
    assert descriptor["config_sha256"] == hashlib.sha256(_BASE_CONFIG).hexdigest()
    assert descriptor["dockingpp_version"] == "0.1.0"
    assert descriptor["scientific"] is True
    assert descriptor["schema_version"] == "dockingpp_archive_adapter_v3"
    assert descriptor["mobile_partner_domain"]["maximum_residues"] == 1200
    assert engine.requires_reference_postprocess is True
    assert not snapshot_dir.exists()

    job = _job()
    work_dir = tmp_path / "work"
    result = engine.run(
        job=job,
        protein_path=receptor,
        peptide_path=peptide,
        work_dir=work_dir,
    )

    assert snapshot_dir.is_dir()
    assert len(calls) == 1
    invocation = calls[0]
    command = invocation["command"]
    assert command[:4] == ["python-for-test", "-m", CLI_MODULE, "run"]
    assert invocation["shell"] is False
    assert invocation["check"] is False
    assert invocation["timeout"] == 30
    config = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
    assert config["legacy_option"] == "preserved"
    assert config["search_engine"] == "pso"
    assert config["search_space_mode"] == "reduced"
    assert config["full_search"] is False
    assert config["generations"] == 30
    assert config["pop_size"] == 12
    assert config["top_pockets"] == 4
    assert config["max_pockets_used"] == 4
    assert config["peptide_input_validation_enabled"] is True
    assert config["receptor_peptide_pair_validation_enabled"] is True
    assert config["peptide_min_residues"] == 5
    assert config["peptide_max_residues"] == 1200
    assert config["docking_confidence_audit"][
        "mobile_partner_residue_count"
    ] == 5
    assert config["docking_confidence_audit"]["mobile_partner_domain"][
        "scope"
    ] == "input_acceptance_not_calibration_guarantee"
    assert config["confidence_policy"]["enabled"] is False
    assert config["confidence_policy"]["apply_fallback"] is False
    assert config["confidence_policy"]["search_control"]["enabled"] is False
    assert config["reduced_geometric_fallback_enabled"] is False
    assert result.scan_gap_norm == pytest.approx(0.41)
    assert result.viable_fraction_mean == pytest.approx(0.72)
    assert result.elite_dispersion == pytest.approx(0.18)
    assert result.uncertainty_count == pytest.approx(2.0)
    assert result.energy_q20_norm == pytest.approx(-4.7)
    assert result.dockq is result.fnat is result.irmsd is result.lrmsd is None
    for artifact_path in (
        result.pose_artifact_path,
        result.engine_summary_path,
        result.engine_result_path,
        result.resolved_config_path,
        result.stdout_log_path,
        result.stderr_log_path,
    ):
        assert artifact_path is not None
        assert not Path(artifact_path).is_absolute()
        assert (work_dir / artifact_path).is_file()


@pytest.mark.parametrize("algorithm", [Algorithm.AG, Algorithm.PSO, Algorithm.HYBRID])
@pytest.mark.parametrize("protocol", [ProtocolMode.FULL, ProtocolMode.REDUCED])
def test_resolved_yaml_uses_each_canonical_algorithm_and_concrete_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    algorithm: Algorithm,
    protocol: ProtocolMode,
):
    engine, _, _ = _engine(tmp_path)
    receptor, peptide = _inputs(tmp_path)
    calls: list[dict[str, Any]] = []
    _install_fake_success(monkeypatch, calls)
    result = engine.run(
        job=_job(algorithm=algorithm, protocol=protocol),
        protein_path=receptor,
        peptide_path=peptide,
        work_dir=tmp_path / "work",
    )
    command = calls[0]["command"]
    config = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
    assert config["search_engine"] == algorithm.value
    assert config["search_space_mode"] == protocol.value
    assert config["full_search"] is (protocol == ProtocolMode.FULL)
    if protocol == ProtocolMode.FULL:
        assert config["generations"] == 40
        assert config["top_pockets"] == 8
        assert result.scan_gap_norm is None
    else:
        assert config["generations"] == 30
        assert config["top_pockets"] == 4
        assert result.scan_gap_norm == pytest.approx(0.41)


def test_reduced_full_fallback_is_rejected_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    engine, _, _ = _engine(tmp_path)
    receptor, peptide = _inputs(tmp_path)
    calls: list[dict[str, Any]] = []
    _install_fake_success(monkeypatch, calls, fallback=True)

    with pytest.raises(DockingPPOutputError, match="FULL fallback"):
        engine.run(
            job=_job(),
            protein_path=receptor,
            peptide_path=peptide,
            work_dir=tmp_path / "work",
        )


def test_undefined_single_pocket_gap_is_preserved_without_fabrication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    engine, _, _ = _engine(tmp_path)
    receptor, peptide = _inputs(tmp_path)
    calls: list[dict[str, Any]] = []
    _install_fake_success(monkeypatch, calls, undefined_scan_gap=True)

    result = engine.run(
        job=_job(),
        protein_path=receptor,
        peptide_path=peptide,
        work_dir=tmp_path / "work",
    )

    assert result.scan_gap_norm is None
    assert result.viable_fraction_mean == pytest.approx(0.72)


@pytest.mark.parametrize("protocol", [ProtocolMode.FULL, ProtocolMode.REDUCED])
@pytest.mark.parametrize("status", ["unavailable", "not_applicable"])
def test_unavailable_native_metrics_keep_pose_for_external_reference_evaluation(
    tmp_path, monkeypatch, protocol, status,
):
    engine, _, _ = _engine(tmp_path)
    receptor, peptide = _inputs(tmp_path)
    payload = {
        "schema_version": "1.0", "status": status, "available": False, "metrics": None,
        "error": {"type": "ValueError", "message": "The native complex has no receptor-peptide contacts at the selected cutoff"},
    }
    _install_fake_success(monkeypatch, [], raw_structural_metrics=payload)
    work = tmp_path / "work"
    result = engine.run(job=_job(protocol=protocol), protein_path=receptor,
                        peptide_path=peptide, work_dir=work)
    pose = work / result.pose_artifact_path
    assert pose.is_file()
    assert json.loads((pose.parent / "structural_metrics.json").read_text()) == payload
    assert result.dockq is result.fnat is result.irmsd is result.lrmsd is None
    assert result.reference_metrics_path is None
    assert engine.requires_reference_postprocess is True


@pytest.mark.parametrize("changes", [
    {"schema_version": "2.0"}, {"available": True}, {"available": "false"},
    {"status": "available"}, {"status": None}, {"status": {}},
    {"metrics": []}, {"error": None}, {"error": {"message": ""}},
])
def test_malformed_native_metric_contracts_remain_rejected(tmp_path, monkeypatch, changes):
    engine, _, _ = _engine(tmp_path)
    receptor, peptide = _inputs(tmp_path)
    payload = {
        "schema_version": "1.0", "status": "unavailable", "available": False,
        "metrics": None, "error": {"message": "No native contacts"},
    }
    payload.update(changes)
    _install_fake_success(monkeypatch, [], raw_structural_metrics=payload)
    with pytest.raises(DockingPPOutputError, match="invalid contract"):
        engine.run(job=_job(), protein_path=receptor, peptide_path=peptide, work_dir=tmp_path / "work")


def test_unsafe_zip_member_and_bad_pins_are_rejected_before_extraction(tmp_path: Path):
    unsafe = tmp_path / "unsafe.zip"
    _make_archive(unsafe, unsafe_member="../escape.py")
    with pytest.raises(DockingPPArchiveIntegrityError, match="unsafe ZIP member"):
        DockingPPArchiveEngine(
            unsafe,
            tmp_path / "snapshots-a",
            config_entry=_CONFIG_ENTRY,
        )

    safe = tmp_path / "safe.zip"
    _make_archive(safe)
    with pytest.raises(DockingPPArchiveIntegrityError, match="does not match"):
        DockingPPArchiveEngine(
            safe,
            tmp_path / "snapshots-b",
            config_entry=_CONFIG_ENTRY,
            expected_archive_sha256="0" * 64,
        )
    assert not (tmp_path / "snapshots-a").exists()
    assert not (tmp_path / "snapshots-b").exists()


def test_adapter_rejects_only_above_the_unified_hard_ceiling(tmp_path: Path):
    engine, _, _ = _engine(tmp_path)
    receptor, peptide = _inputs(tmp_path, mobile_partner_residues=1201)

    with pytest.raises(
        DockingPPArchiveIntegrityError,
        match="outside the configured operational range 5–1200 aa",
    ):
        engine.run(
            job=_job(),
            protein_path=receptor,
            peptide_path=peptide,
            work_dir=tmp_path / "work",
        )


def test_timeout_preserves_logs_and_malformed_pose_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    engine, _, _ = _engine(tmp_path)
    receptor, peptide = _inputs(tmp_path)

    def timeout(command: list[str], **kwargs: Any) -> None:
        kwargs["stdout"].write("started\n")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(
        "docking_confidence.adapters.dockingpp_archive_engine.subprocess.run", timeout
    )
    with pytest.raises(DockingPPExecutionError, match="timeout"):
        engine.run(
            job=_job(),
            protein_path=receptor,
            peptide_path=peptide,
            work_dir=tmp_path / "timeout-work",
        )
    stderr_logs = list((tmp_path / "timeout-work").rglob("stderr.log"))
    assert len(stderr_logs) == 1
    assert "timeout" in stderr_logs[0].read_text(encoding="utf-8").lower()

    calls: list[dict[str, Any]] = []
    _install_fake_success(monkeypatch, calls, malformed_pose=True)
    with pytest.raises(DockingPPOutputError, match="pose_role"):
        engine.run(
            job=_job(),
            protein_path=receptor,
            peptide_path=peptide,
            work_dir=tmp_path / "malformed-work",
        )


@pytest.mark.parametrize("status", ["NO_FEASIBLE_POCKET", "REDUCED_FAILED", "RUN_FULL_REQUIRED"])
@pytest.mark.parametrize("returncode", [0, 1])
def test_reduced_failure_precedes_missing_metrics_and_exit_code(tmp_path, monkeypatch, status, returncode):
    engine, _, _ = _engine(tmp_path)
    receptor, peptide = _inputs(tmp_path)

    def fail(command, **kwargs):
        out = Path(command[command.index("--out") + 1])
        (out / "summary.json").write_text(json.dumps({
            "status": status, "failure_reason": "specific_engine_failure",
        }))
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr("docking_confidence.adapters.dockingpp_archive_engine.subprocess.run", fail)
    with pytest.raises(DockingPPReducedError, match="specific_engine_failure") as error:
        engine.run(job=_job(), protein_path=receptor, peptide_path=peptide, work_dir=tmp_path / "work")
    assert error.value.status == status
    assert "metrics.jsonl" not in str(error.value)


def _mutate_fake_output(monkeypatch, transform):
    calls = []
    _install_fake_success(monkeypatch, calls)
    original = subprocess.run

    def wrapped(command, **kwargs):
        completed = original(command, **kwargs)
        transform(Path(command[command.index("--out") + 1]))
        return completed

    monkeypatch.setattr("docking_confidence.adapters.dockingpp_archive_engine.subprocess.run", wrapped)
    return calls


def test_legacy_full_fallback_is_diagnosed_before_absent_parent_metrics(tmp_path, monkeypatch):
    engine, _, _ = _engine(tmp_path)
    receptor, peptide = _inputs(tmp_path)

    def transform(out):
        (out / "metrics.jsonl").unlink()
        summary = json.loads((out / "summary.json").read_text())
        summary.update(fallback_reason="no_feasible_pocket", fallback_executed=True)
        (out / "summary.json").write_text(json.dumps(summary))

    _mutate_fake_output(monkeypatch, transform)
    with pytest.raises(DockingPPOutputError, match="FULL fallback"):
        engine.run(job=_job(), protein_path=receptor, peptide_path=peptide, work_dir=tmp_path / "work")


@pytest.mark.parametrize("field,value", [("engine_name", "AGSearch"), ("search_space_mode", "full")])
def test_adapter_checks_actual_pose_engine_and_protocol(tmp_path, monkeypatch, field, value):
    engine, _, _ = _engine(tmp_path)
    receptor, peptide = _inputs(tmp_path)

    def transform(out):
        path = next(out.rglob("best_pose.json"))
        pose = json.loads(path.read_text())
        pose[field] = value
        path.write_text(json.dumps(pose))

    _mutate_fake_output(monkeypatch, transform)
    with pytest.raises(DockingPPOutputError, match="does not match"):
        engine.run(job=_job(), protein_path=receptor, peptide_path=peptide, work_dir=tmp_path / "work")


def test_rejected_initial_pose_cannot_be_a_successful_reduced_result(tmp_path, monkeypatch):
    engine, _, _ = _engine(tmp_path)
    receptor, peptide = _inputs(tmp_path)

    def transform(out):
        path = next(out.rglob("best_pose.json"))
        pose = json.loads(path.read_text())
        pose["generation"] = 0
        pose["meta"]["final_hard_clash_gate"] = {"triggered": True}
        path.write_text(json.dumps(pose))

    _mutate_fake_output(monkeypatch, transform)
    with pytest.raises(DockingPPReducedError, match="final_pose_hard_clash"):
        engine.run(job=_job(), protein_path=receptor, peptide_path=peptide, work_dir=tmp_path / "work")


@pytest.mark.parametrize("content,message", [(None, "missing or unreadable"), ("bad-json", "invalid"), ("", "no JSON records")])
def test_genuine_metrics_failures_remain_distinguishable(tmp_path, monkeypatch, content, message):
    engine, _, _ = _engine(tmp_path)
    receptor, peptide = _inputs(tmp_path)

    def transform(out):
        path = out / "metrics.jsonl"
        if content is None:
            path.unlink()
        else:
            path.write_text(content)

    _mutate_fake_output(monkeypatch, transform)
    with pytest.raises(DockingPPOutputError, match=message):
        engine.run(job=_job(), protein_path=receptor, peptide_path=peptide, work_dir=tmp_path / "work")


def test_repeated_jobs_and_algorithms_have_private_output_cwd_and_tmp(tmp_path, monkeypatch):
    engine, _, _ = _engine(tmp_path)
    receptor, peptide = _inputs(tmp_path)
    calls = []
    _install_fake_success(monkeypatch, calls)
    results = [engine.run(job=_job(algorithm=a), protein_path=receptor,
                          peptide_path=peptide, work_dir=tmp_path / "work")
               for a in (Algorithm.AG, Algorithm.PSO, Algorithm.HYBRID, Algorithm.AG)]
    assert len({result.pose_artifact_path for result in results}) == 4
    assert len({call["cwd"] for call in calls}) == 4
    assert len({call["env"]["TEMP"] for call in calls}) == 4
    for call in calls:
        out = Path(call["command"][call["command"].index("--out") + 1])
        assert Path(call["cwd"]).resolve() != out.parent.resolve()
        assert call["env"]["TMP"] == call["cwd"]
        assert not Path(call["cwd"]).exists()  # ephemeral scratch was cleaned


def test_specific_algorithm_parameters_survive_config_resolution():
    base = {"ag_sbx_eta": 22.0, "pso_inertia_weight": 0.3, "ga_interval": 4}
    for algorithm in Algorithm:
        config = DockingPPArchiveEngine._resolved_job_config(_job(algorithm=algorithm), base)
        for key, value in base.items():
            assert config[key] == value
        assert config["seed"] == 17
    assert base == {"ag_sbx_eta": 22.0, "pso_inertia_weight": 0.3, "ga_interval": 4}

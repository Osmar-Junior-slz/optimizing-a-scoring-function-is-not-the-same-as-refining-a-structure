from __future__ import annotations

from pathlib import Path

import pytest

from docking_confidence.application.batch_launcher import (
    DOCKINGPP_ARCHIVE_FILENAME,
    EngineArchiveDiscoveryError,
    default_dockingpp_snapshot_dir,
    discover_dockingpp_archive,
    load_dockingpp_archive_contract,
    real_batch_command,
)


def test_versioned_dockingpp_contract_is_closed_and_pinned() -> None:
    contract = load_dockingpp_archive_contract()

    assert contract.archive_filename == DOCKINGPP_ARCHIVE_FILENAME
    assert contract.config_entry == "dockingpp/configs/default.yaml"
    assert len(contract.archive_sha256) == 64
    assert len(contract.config_sha256) == 64
    assert contract.source_path.is_file()
    assert default_dockingpp_snapshot_dir().name == "engine_snapshots"


def test_archive_discovery_is_explicit_or_unambiguous(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    archive = first / DOCKINGPP_ARCHIVE_FILENAME
    archive.write_bytes(b"not opened during discovery")

    assert discover_dockingpp_archive(search_roots=[first, second]) == archive.resolve()
    assert discover_dockingpp_archive(archive) == archive.resolve()

    duplicate = second / DOCKINGPP_ARCHIVE_FILENAME
    duplicate.write_bytes(b"another candidate")
    with pytest.raises(EngineArchiveDiscoveryError, match="multiple"):
        discover_dockingpp_archive(search_roots=[first, second])


def test_real_worker_command_pins_engine_and_expands_seeds(tmp_path: Path) -> None:
    archive = tmp_path / DOCKINGPP_ARCHIVE_FILENAME
    archive.write_bytes(b"selected explicitly; adapter verifies content later")
    command = real_batch_command(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        engine_archive=archive,
        algorithms=["AG", "pso", "ag"],
        protocol="reduced_vs_full",
        generations=12,
        population=8,
        top_pockets=3,
        seed_start=4,
        seed_count=2,
        timeout_seconds=90.0,
        resume_after_cancel=True,
    )

    assert command[1:4] == ("-m", "docking_confidence", "batch-run")
    assert command[command.index("--algorithms") + 1 : command.index("--protocol")] == (
        "ag",
        "pso",
    )
    assert command[command.index("--protocol") + 1] == "reduced_vs_full"
    assert command[command.index("--docking-task") + 1] == "local_refinement"
    assert command[command.index("--seed") + 1 : command.index("--generations")] == (
        "4",
        "5",
    )
    assert "--engine-archive-sha256" in command
    assert "--engine-config-sha256" in command
    assert command[-1] == "--resume-after-cancel"


def test_real_worker_rejects_non_finite_timeout(tmp_path: Path) -> None:
    archive = tmp_path / DOCKINGPP_ARCHIVE_FILENAME
    archive.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="finite"):
        real_batch_command(
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            engine_archive=archive,
            algorithms=["pso"],
            protocol="reduced",
            generations=1,
            population=1,
            top_pockets=1,
            seed_start=1,
            seed_count=1,
            timeout_seconds=float("inf"),
        )


def test_default_worker_is_self_contained_and_does_not_discover_a_zip(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("local engine must not read or discover an external ZIP")
    monkeypatch.setattr("docking_confidence.application.batch_launcher.discover_dockingpp_archive", forbidden)
    monkeypatch.setattr("docking_confidence.application.batch_launcher.load_dockingpp_archive_contract", forbidden)
    command = real_batch_command(
        input_dir=tmp_path / "input", output_dir=tmp_path / "output",
        algorithms=["ag", "pso", "hybrid"], protocol="reduced",
        generations=4, population=6, top_pockets=2, seed_start=1, seed_count=1,
        timeout_seconds=60,
    )
    assert "batch-run" in command
    assert not any(value.startswith("--engine-archive") for value in command)


@pytest.mark.parametrize("docking_task", ["surface_discovery", "local_refinement"])
def test_real_worker_passes_the_explicit_docking_task(tmp_path, docking_task):
    command = real_batch_command(
        input_dir=tmp_path / "input", output_dir=tmp_path / "output",
        algorithms=["pso"], protocol="reduced_vs_full", docking_task=docking_task,
        generations=4, population=6, top_pockets=2, seed_start=1, seed_count=1,
        timeout_seconds=60,
    )
    assert command[command.index("--docking-task") + 1] == docking_task


def test_real_worker_rejects_unknown_docking_task(tmp_path):
    with pytest.raises(ValueError):
        real_batch_command(
            input_dir=tmp_path / "input", output_dir=tmp_path / "output",
            algorithms=["pso"], protocol="reduced", docking_task="unknown",
            generations=4, population=6, top_pockets=2, seed_start=1, seed_count=1,
            timeout_seconds=60,
        )


def test_calibrated_worker_passes_selected_model_as_one_argv_value(tmp_path):
    artifact = tmp_path / "model with spaces" / "artifact.json"
    artifact.parent.mkdir()
    artifact.write_text("{}", encoding="utf-8")
    command = real_batch_command(
        input_dir=tmp_path / "input", output_dir=tmp_path / "output",
        algorithms=["pso"], protocol="reduced_vs_full",
        generations=4, population=6, top_pockets=2, seed_start=1, seed_count=1,
        timeout_seconds=60, confidence_artifact=artifact,
    )
    assert command[command.index("--confidence-artifact") + 1] == str(artifact.resolve())
    assert command[command.index("--protocol") + 1] == "reduced_vs_full"


@pytest.mark.parametrize("protocol", ["reduced", "full"])
def test_calibrated_worker_requires_full_fallback_plan(tmp_path, protocol):
    with pytest.raises(ValueError, match="requires protocol='reduced_vs_full'"):
        real_batch_command(
            input_dir=tmp_path / "input", output_dir=tmp_path / "output",
            algorithms=["pso"], protocol=protocol,
            generations=4, population=6, top_pockets=2, seed_start=1, seed_count=1,
            timeout_seconds=60, confidence_artifact=tmp_path / "artifact.json",
        )


def test_calibrated_cli_forwards_model_without_running_a_docking(tmp_path, monkeypatch, capsys):
    import sys
    from docking_confidence.cli import main
    from docking_confidence.application.batch import BatchRunSummary

    artifact = str(tmp_path / "artifact.json")
    engine = object()
    calls = []
    monkeypatch.setattr(
        "docking_confidence.adapters.dockingpp_local_engine.DockingPPLocalEngine",
        lambda **kwargs: engine,
    )
    def run_batch(*args, **kwargs):
        calls.append((args, kwargs))
        return BatchRunSummary(
            batch_id="example", state="COMPLETED", planned_jobs=2, succeeded=1,
            failed=0, pending=0, attempted=1, skipped_existing=0,
            manifest_path=tmp_path / "batch_manifest.json", status_path=tmp_path / "batch_status.json",
            skipped_calibrator=1,
        )
    monkeypatch.setattr("docking_confidence.application.batch.run_batch", run_batch)
    monkeypatch.setattr(sys, "argv", [
        "docking-confidence", "batch-run", "--input-dir", str(tmp_path / "input"),
        "--output-dir", str(tmp_path / "output"), "--protocol", "reduced_vs_full",
        "--confidence-artifact", artifact,
    ])
    main()
    assert len(calls) == 1
    assert calls[0][1]["confidence_artifact"] == artifact
    assert calls[0][1]["engine"] is engine
    assert {job.protocol.value for job in calls[0][1]["jobs"]} == {"reduced", "full"}
    assert '"skipped_calibrator": 1' in capsys.readouterr().out


def test_cli_rejects_calibrator_without_full_fallback_before_engine_setup(tmp_path, monkeypatch, capsys):
    import sys
    from docking_confidence.cli import main

    def forbidden(**kwargs):
        pytest.fail("invalid calibration arguments must be rejected before engine setup")
    monkeypatch.setattr("docking_confidence.adapters.dockingpp_local_engine.DockingPPLocalEngine", forbidden)
    monkeypatch.setattr(sys, "argv", [
        "docking-confidence", "batch-run", "--input-dir", str(tmp_path / "input"),
        "--output-dir", str(tmp_path / "output"), "--protocol", "reduced",
        "--confidence-artifact", str(tmp_path / "artifact.json"),
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "--confidence-artifact requires --protocol reduced_vs_full" in capsys.readouterr().err

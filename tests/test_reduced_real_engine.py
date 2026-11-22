"""Integration regressions against the actual, hash-pinned V5 implementation.

The motor is bundled in the project. Analytical scoring isolates operator
behavior; no search operator is mocked.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

from docking_confidence.adapters.dockingpp_archive_engine import DockingPPArchiveEngine
from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
from docking_confidence.adapters import dockingpp_v5_patch as patch
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, DockingTask, ProtocolMode


@pytest.fixture(scope="module")
def real_v5(tmp_path_factory):
    engine = DockingPPLocalEngine()
    module_root = engine._ensure_snapshot()
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(module_root))
    yield engine, module_root
    sys.path.remove(str(module_root))
    sys.dont_write_bytecode = previous_bytecode
    for name in list(sys.modules):
        if name == "dockingpp" or name.startswith("dockingpp."):
            del sys.modules[name]


def _config(algorithm, mode="reduced"):
    from dockingpp.application.config import Config
    job = build_execution_plan(
        algorithms=[Algorithm(algorithm)], protocol_mode=ProtocolMode(mode),
        base_config=BaseSearchConfig(generations=12, population=8, top_pockets=4, docking_task=DockingTask.SURFACE_DISCOVERY), seeds=[1],
    )[0]
    return Config(**DockingPPArchiveEngine._resolved_job_config(job, {
        "scan": {"enabled": False}, "structural_metrics_enabled": False,
        "ag_sbx_eta": 19.0, "pso_inertia_weight": 0.4,
        "ga_interval": 2, "sw_interval": 2, "sw_elite_iter": 2,
        "sw_heavy_iter": 2, "sw_max_iter": 2, "sw_iter_budget": 12,
    }))


def _scoring(monkeypatch, *, clash=False):
    from dockingpp.pipeline.execucao.busca_execucao import _execucao_unica_coordenador as coordinator
    from dockingpp.pipeline.execucao.busca_execucao import _rescoring_final as rescoring

    def cheap(pose, pocket, weights):
        pose.meta["contatos"] = 10
        pose.meta["colisao_severa"] = clash
        delta = np.mean(pose.coords, axis=0) - np.asarray(pocket.center) - [0.2, 0.3, 0.4]
        return -1e6 if clash else -float(delta @ delta)

    def expensive(pose, *args, **kwargs):
        pose.meta["final_hard_clash_gate"] = {"enabled": True, "triggered": clash}
        pose.final_rescore = pose.score_cheap
        pose.score_expensive = pose.score_cheap
        return pose.score_cheap

    monkeypatch.setattr(coordinator, "score_pose_cheap", cheap)
    # The service imports the scorer in its own module.
    monkeypatch.setattr(rescoring, "score_pose_expensive", expensive)


@pytest.mark.parametrize("algorithm,expected_class", [
    ("ag", "AGSearch"), ("pso", "PSOSearch"), ("hybrid", "ABCGAVGOSSearch"),
])
def test_reduced_calls_real_operator_with_resolved_parameters(real_v5, tmp_path, monkeypatch, algorithm, expected_class):
    from dockingpp.pipeline.run import run_pipeline
    from dockingpp.search.engine import resolve_search_engine
    _scoring(monkeypatch)
    cfg = _config(algorithm)
    search, name = resolve_search_engine(cfg)
    assert name == expected_class
    original = type(search)._next_generation
    calls = []

    def observed(self, context):
        calls.append((context.cfg, context.pop_size, context.generation_index))
        assert context.cfg.search_engine == algorithm
        assert context.cfg.seed == 1
        assert context.cfg.search_space_mode == "reduced"
        assert context.cfg.ag_sbx_eta == 19.0
        assert context.cfg.pso_inertia_weight == 0.4
        assert context.cfg.ga_interval == 2
        return original(self, context)

    monkeypatch.setattr(type(search), "_next_generation", observed)
    run_pipeline(cfg, "__dummy__", "__dummy__", str(tmp_path))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert calls, "The concrete algorithm must execute beyond common initialization"
    assert summary["search_engine_name"] == expected_class
    assert summary["search_strategy"] == algorithm
    assert summary["fallback_executed"] is False
    assert summary["final_pose_available"] is True
    assert all(population <= cfg.pop_size for _, population, _ in calls)
    assert all(call_cfg.generations <= cfg.generations for call_cfg, _, _ in calls)
    assert (tmp_path / "metrics.jsonl").is_file()
    events = [json.loads(line) for path in tmp_path.glob("*/metrics.jsonl")
              for line in path.read_text().splitlines() if line.strip()]
    executed = [event for event in events if event.get("name") == "search.operator_completed"]
    assert len(executed) == len(calls)
    assert {event["engine_name"] for event in executed} == {expected_class}


def test_same_seed_flat_clash_objective_is_failure_not_recycled_winner(real_v5, tmp_path, monkeypatch):
    from dockingpp.pipeline.run import run_pipeline
    _scoring(monkeypatch, clash=True)
    candidate_coords = []
    for algorithm in ("ag", "pso", "hybrid"):
        out = tmp_path / algorithm
        with pytest.raises(RuntimeError, match="REDUCED_FAILED: no_valid_reduced_pose"):
            run_pipeline(_config(algorithm), "__dummy__", "__dummy__", str(out))
        summary = json.loads((out / "summary.json").read_text())
        assert summary["final_pose_available"] is False
        assert summary["pose_output_path"] is None
        assert summary["fallback_executed"] is False
        assert not (out / "fallback_full").exists()
        assert not list(out.rglob("predicted_complex.pdb"))
        assert json.loads((out / "metrics.jsonl").read_text())["status"] == "REDUCED_FAILED"
        pose = json.loads(next(out.glob("*/artifacts/best_pose.json")).read_text())
        candidate_coords.append(pose["coords"])
        assert pose["generation"] == 0
    # An identical diagnostic candidate is legitimate under a flat objective.
    # The regression is publishing it as a successful REDUCED prediction.
    assert candidate_coords[0] == candidate_coords[1] == candidate_coords[2]


def test_no_pocket_stops_before_search_and_full_remains_independent(real_v5, tmp_path, monkeypatch):
    from dockingpp.pipeline import run as pipeline
    from dockingpp.pipeline.execucao.selecao_bolsoes import SelecionadorBolsoesPipeline
    original = SelecionadorBolsoesPipeline.selecionar

    def empty_reduced(self, **kwargs):
        selection = original(self, **kwargs)
        if kwargs["cfg"].search_space_mode == "reduced":
            return replace(selection, accepted_pockets=[], feasible_pockets=[])
        return selection

    monkeypatch.setattr(SelecionadorBolsoesPipeline, "selecionar", empty_reduced)
    _scoring(monkeypatch)
    out = tmp_path / "reduced"
    with pytest.raises(RuntimeError, match="NO_FEASIBLE_POCKET"):
        pipeline.run_pipeline(_config("ag"), "__dummy__", "__dummy__", str(out))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["total_n_eval"] == 0
    assert summary["executed_pockets"] == []
    assert summary["fallback_executed"] is False
    assert not list(out.rglob("best_pose.json"))
    assert not (out / "fallback_full").exists()
    pipeline.run_pipeline(_config("ag", "full"), "__dummy__", "__dummy__", str(tmp_path / "full"))
    full = json.loads((tmp_path / "full/summary.json").read_text())
    assert full["search_space_mode"] == "full"
    assert full["final_pose_available"] is True


@pytest.mark.parametrize("source", ["confidence_policy", "geometric"])
def test_post_reduced_fallback_requires_separate_full_job(real_v5, tmp_path, source):
    from dockingpp.pipeline.execucao.busca_execucao.executor_busca import ExecutorBuscaPipeline
    executor = ExecutorBuscaPipeline(
        executar_execucao_unica=lambda **kwargs: pytest.fail("FULL must not execute"),
        escrever_debug_summary=lambda **kwargs: None, alocar_orcamento=lambda *args: [],
        construir_bolsao_global=lambda *args: pytest.fail("FULL pocket must not be built"), classe_config=None,
    )
    with pytest.raises(RuntimeError, match="RUN_FULL_REQUIRED"):
        executor._execute_post_reduced_fallback(
            cfg=_config("pso"), receptor=None, peptide=None,
            receptor_path="r", peptide_path="p", out_dir=str(tmp_path), run_id="test",
            tracer=None, parent_summary={}, fallback_trigger_source=source,
        )
    assert not (tmp_path / "fallback_full").exists()


def test_patch_provenance_and_operator_sources_are_preserved(real_v5):
    engine, root = real_v5
    descriptor = engine.audit_descriptor()
    assert descriptor["engine_patch"]["version"] == "factorial_fixed_evaluations_v5"
    assert descriptor["source"] == "bundled_in_this_distribution"
    assert engine.archive_sha256 == patch.ARCHIVE_SHA256
    engine._validate_local_source()
    with pytest.raises(ValueError, match="preimage"):
        patch.patch_source(b"unexpected source")


def test_legacy_archive_patch_matches_bundled_source_when_archive_is_available(real_v5, tmp_path):
    candidates = [parent / "DockingReduce_V5.zip" for parent in Path(__file__).resolve().parents]
    candidates.append(Path(__file__).resolve().parents[2] / "SistemaV5" / "DockingReduce_V5.zip")
    archive_path = next((path for path in candidates if path.is_file()), None)
    if archive_path is None:
        pytest.skip("optional archive compatibility test; bundled engine needs no ZIP")
    engine, root = real_v5
    with zipfile.ZipFile(archive_path) as archive:
        # Operators remain identical; the shared REDUCED coordinate placement
        # now belongs to the explicitly hash-audited correction set below.
        for name in ("ag", "pso", "hibrido"):
            relative = f"dockingpp/search/algoritmos/{name}.py"
            assert (root / relative).read_bytes() == archive.read("dockingpp/" + relative)
        tracked = DockingPPLocalEngine(seed_lineage=True)
        config = next(n for n in archive.namelist() if n.endswith("/configs/default.yaml"))
        config_hash = hashlib.sha256(archive.read(config)).hexdigest()
    snapshots = []
    for local, options in [(engine, {}), (tracked, {"seed_lineage": True})]:
        zipped = DockingPPArchiveEngine(
            archive_path, tmp_path, config_entry=config,
            expected_archive_sha256=patch.ARCHIVE_SHA256, expected_config_sha256=config_hash,
            **options,
        )
        assert zipped.scientific and zipped.engine_patch == local.engine_patch
        snapshot = zipped._ensure_snapshot()
        snapshots.append(snapshot)
        for file in local.package_root.rglob("*.py"):
            assert (snapshot / "dockingpp" / file.relative_to(local.package_root)).read_bytes() == file.read_bytes()
    assert snapshots[0] != snapshots[1]

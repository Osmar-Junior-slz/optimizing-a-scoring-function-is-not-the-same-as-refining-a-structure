"""Independent conditions and actual objective ceilings on the real engines."""
from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from docking_confidence.adapters.dockingpp_archive_engine import DockingPPArchiveEngine, DockingPPOutputError
from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import (
    Algorithm, BaseSearchConfig, FactorialExperimentConfig, ProtocolMode,
)
from dockingpp.application.config import Config


def _jobs(algorithm="ag", *, ceiling=73, generations=4, population=8, fixed=True):
    jobs = build_execution_plan(
        algorithms=[Algorithm(algorithm)], protocol_mode=ProtocolMode.FACTORIAL,
        base_config=BaseSearchConfig(
            generations=generations, population=population, top_pockets=9,
            experiment=FactorialExperimentConfig(
                evaluation_budget=ceiling, reduced_search_scale=.3,
                max_translation=4., max_rotation_degrees=20.,
                pose_source="synthetic supplied pose independent of scoring target",
            ),
        ), seeds=[19],
    )
    return jobs if fixed else tuple(replace(job, profile=replace(job.profile, budget_policy="common_ceiling")) for job in jobs)


def _sha_record(record):
    unsigned = {key: value for key, value in record.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _execute(job, output: Path, monkeypatch):
    from dockingpp.pipeline.run import run_pipeline
    from dockingpp.pipeline.execucao.busca_execucao import _execucao_unica_coordenador as coordinator
    from dockingpp.pipeline.execucao.busca_execucao import _rescoring_final as rescoring

    calls = []
    def cheap(pose, pocket, weights):
        calls.append(np.array(pose.meta["solution_vector"]))
        pose.meta.update(contatos=10, colisao_severa=False)
        # The seeded identity is deliberately worse than another feasible pose.
        offset = np.array(pose.meta["solution_vector"][:3])
        return -float(np.sum((offset - [.4, .5, .6]) ** 2))
    def expensive(pose, *args):
        pose.meta["final_hard_clash_gate"] = {"enabled": True, "triggered": False}
        return pose.score_cheap

    monkeypatch.setattr(coordinator, "score_pose_cheap", cheap)
    monkeypatch.setattr(rescoring, "score_pose_expensive", expensive)
    config = DockingPPArchiveEngine._resolved_job_config(job, {
        "structural_metrics_enabled": False, "scan": {"enabled": True},
        "ga_interval": 2, "sw_interval": 2, "sw_elite_iter": 2, "sw_heavy_iter": 2,
        "sw_max_iter": 2, "sw_iter_budget": 12,
    })
    run_pipeline(Config(**config), "__dummy__", "__dummy__", str(output))
    summary = json.loads((output / "summary.json").read_text())
    pocket = summary["population_confidence_by_pocket"][0]
    result = json.loads((output / "result.json").read_text())
    verified = DockingPPArchiveEngine._factorial_audit_fields(job, summary, result)
    assert verified["actual_eval_count"] == len(calls)
    assert verified["initial_population_sha256"] == pocket["initial_population_audit"]["sha256"]
    events = [json.loads(line) for path in output.rglob("metrics.jsonl")
              for line in path.read_text().splitlines() if line.strip()]
    return pocket, calls, events


def test_adapter_resolves_all_four_conditions_without_legacy_coupling():
    configs = {job.job_id: DockingPPArchiveEngine._resolved_job_config(job, {}) for job in _jobs()}
    assert len(configs) == 4
    for config in configs.values():
        scale = .3 if config["search_extent"] == "reduced" else 1.
        assert config["generations"] == 4
        assert config["pop_size"] == 8
        assert config["top_pockets"] == 9
        assert config["objective_evaluation_budget_per_pocket"] == 73
        assert config["budget_policy"] == "replicated"
        audit = config["docking_confidence_audit"]
        assert audit["initialization"] == config["initialization"]
        assert audit["effective_max_trans"] == 4. * scale
        assert audit["effective_max_rot_deg"] == 20. * scale
        assert audit["evaluation_budget_is_hard_cap"]
        assert audit["evaluation_budget_is_fixed_consumption"]
        assert config["hybrid_plateau_patience"] == 0


@pytest.mark.parametrize("algorithm", ["ag", "pso", "hybrid"])
def test_real_engines_record_ordered_initial_population_and_true_consumption(algorithm, tmp_path, monkeypatch):
    populations = {}
    for job in _jobs(algorithm):
        pocket, calls, events = _execute(job, tmp_path / job.job_id, monkeypatch)
        initial = pocket["initial_population_audit"]
        execution = pocket["execution_audit"]
        assert initial["sha256"] == _sha_record(initial)
        assert execution["sha256"] == _sha_record(execution)
        assert initial["recorded_before_objective_evaluations"] == 0
        assert len(initial["solution_vectors"]) == 8
        assert initial["seed"] == job.seed
        assert initial["initialization"] == job.profile.initialization
        assert execution["initial_population_sha256"] == initial["sha256"]
        assert execution["objective_evaluations_actual"] == len(calls)
        assert sum(execution["objective_evaluations_by_generation"]) == len(calls)
        assert len(calls) == job.profile.evaluation_budget
        assert execution["stop_reason"] == "objective_evaluation_ceiling"
        assert pocket["generations"] == job.profile.generations
        initial_events = [event for event in events if event.get("name") == "search.initial_population"]
        assert len(initial_events) == 1
        assert initial_events[0]["initial_population_audit"] == initial
        vectors = np.array(initial["solution_vectors"])
        if job.profile.initialization == "seeded":
            np.testing.assert_array_equal(vectors[0], np.zeros(6))
            assert initial["seeded_candidate_count"] == 1
            assert np.count_nonzero(np.all(vectors == 0, axis=1)) == 1
            np.testing.assert_array_equal(calls[0], np.zeros(6))
            lineage = pocket["seed_lineage"]
            records = lineage["generations"]
            assert records[0]["generation"] == 0
            assert records[0]["unchanged_seed_count"] == records[0]["seed_lineage_count"] == 1
            assert records[-1]["n_eval_cumulative"] == len(calls)
            assert [row["n_eval_cumulative"] for row in records] == list(np.cumsum(
                execution["objective_evaluations_by_generation"]
            ))
            assert all(0 <= row["unchanged_seed_count"] <= row["seed_lineage_count"]
                       <= row["population_size"] for row in records)
            if algorithm == "pso":
                assert all(row["seed_lineage_count"] == 1 for row in records)
                assert all(isinstance(row["pso_global_best_from_seed_particle"], bool)
                           for row in records[1:])
            output = tmp_path / job.job_id
            final = json.loads((output / "result.json").read_text())["best_pose_meta"]
            assert isinstance(final["seed_lineage"], bool)
            if final["seed_unchanged"]:
                assert final["seed_lineage"] and final["solution_vector"] == [0.] * 6
            assert [e["seed_lineage"] for e in events if e["name"] == "search.seed_lineage"] == [lineage]
        else:
            assert initial["seeded_candidate_count"] == 0
            assert not np.any(np.all(vectors == 0, axis=1))
            assert "seed_lineage" not in pocket
        populations[(job.profile.search_extent, job.profile.initialization)] = vectors
    for extent in ("reduced", "expanded"):
        np.testing.assert_array_equal(populations[(extent, "uniform")][1:],
                                      populations[(extent, "seeded")][1:])
    np.testing.assert_allclose(populations[("reduced", "uniform")],
                               populations[("expanded", "uniform")] * .3, atol=1e-14)


@pytest.mark.parametrize("algorithm", ["ag", "pso", "hybrid"])
def test_common_ceiling_can_stop_at_initial_population_without_changing_requested_generations(algorithm, tmp_path, monkeypatch):
    for job in _jobs(algorithm, ceiling=8, generations=20, fixed=False):
        pocket, calls, _ = _execute(job, tmp_path / job.job_id, monkeypatch)
        assert len(calls) == 8
        assert pocket["generations"] == 20
        assert pocket["execution_audit"]["completed_generations"] == 0
        assert pocket["execution_audit"]["stop_reason"] == "objective_evaluation_ceiling"


def test_bundled_source_remains_verified_against_versioned_manifest():
    engine = DockingPPLocalEngine(seed_lineage=True)
    descriptor = engine.audit_descriptor()
    assert descriptor["engine_patch"]["version"] == "factorial_seed_lineage_v6"
    assert descriptor["source_manifest_sha256"]


@pytest.mark.parametrize("algorithm", ["ag", "pso", "hybrid"])
def test_fixed_7600_counts_real_objective_calls_beyond_nominal_generation_count(algorithm, tmp_path, monkeypatch):
    job = _jobs(algorithm, ceiling=7600, generations=1, population=100)[0]
    pocket, calls, _ = _execute(job, tmp_path / job.job_id, monkeypatch)
    execution = pocket["execution_audit"]
    assert len(calls) == execution["objective_evaluations_actual"] == 7600
    assert execution["completed_generations"] > job.profile.generations
    assert execution["stop_reason"] == "objective_evaluation_ceiling"


def test_no_new_evaluations_fails_without_fabricating_budget_consumption(tmp_path, monkeypatch):
    from dockingpp.search.algoritmos.ag import AGSearch

    monkeypatch.setattr(AGSearch, "_next_generation", lambda self, context: list(context.previous))
    with pytest.raises(RuntimeError, match="8/73 avaliações efetivas"):
        _execute(_jobs()[0], tmp_path / "stagnant", monkeypatch)


def test_factorial_adapter_rejects_inconsistent_executable_arm():
    job = _jobs()[0]
    job = replace(job, profile=replace(job.profile, search_extent="expanded"))
    with pytest.raises(ValueError, match="does not match executable protocol"):
        DockingPPArchiveEngine._resolved_job_config(job, {})


@pytest.mark.parametrize("tamper", ["initial_hash", "execution_hash", "ceiling", "different_records", "missing_record"])
def test_factorial_receipt_validation_rejects_inconsistent_audit(tamper, tmp_path, monkeypatch):
    job = _jobs()[0]
    pocket, _, _ = _execute(job, tmp_path / job.job_id, monkeypatch)
    summary = {"population_confidence_by_pocket": [deepcopy(pocket)]}
    if tamper == "initial_hash":
        summary["population_confidence_by_pocket"][0]["initial_population_audit"]["solution_vectors"][0][0] += .1
    elif tamper == "execution_hash":
        summary["population_confidence_by_pocket"][0]["execution_audit"]["objective_evaluations_actual"] += 1
    elif tamper == "ceiling":
        record = summary["population_confidence_by_pocket"][0]["execution_audit"]
        record["objective_evaluations_actual"] = job.profile.evaluation_budget + 1
        record["sha256"] = _sha_record(record)
    elif tamper == "missing_record":
        del summary["population_confidence_by_pocket"][0]["initial_population_audit"]
    result = deepcopy(summary)
    if tamper == "different_records":
        result["population_confidence_by_pocket"][0]["execution_audit"]["stop_reason"] = "hybrid_plateau"
    with pytest.raises(DockingPPOutputError, match="factorial"):
        DockingPPArchiveEngine._factorial_audit_fields(job, summary, result)


def test_initial_population_is_persisted_before_an_objective_failure(tmp_path, monkeypatch):
    from dockingpp.pipeline.run import run_pipeline
    from dockingpp.pipeline.execucao.busca_execucao import _execucao_unica_coordenador as coordinator

    def fail_objective(*args):
        raise RuntimeError("synthetic objective failure")

    monkeypatch.setattr(coordinator, "score_pose_cheap", fail_objective)
    cfg = Config(**DockingPPArchiveEngine._resolved_job_config(_jobs()[0], {
        "structural_metrics_enabled": False,
    }))
    with pytest.raises(RuntimeError, match="synthetic objective failure"):
        run_pipeline(cfg, "__dummy__", "__dummy__", str(tmp_path / "failed"))
    events = [json.loads(line) for path in (tmp_path / "failed").rglob("metrics.jsonl")
              for line in path.read_text().splitlines() if line.strip()]
    initial = next(event["initial_population_audit"] for event in events
                   if event.get("name") == "search.initial_population")
    assert initial["recorded_before_objective_evaluations"] == 0
    assert len(initial["solution_vectors"]) == 8
    assert initial["sha256"] == _sha_record(initial)

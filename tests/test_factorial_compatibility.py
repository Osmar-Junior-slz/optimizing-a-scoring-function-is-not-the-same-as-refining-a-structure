from dataclasses import asdict
from types import SimpleNamespace

import pytest

from docking_confidence.application.batch import _job_snapshot
from docking_confidence.application.batch_launcher import real_batch_command
from docking_confidence.application.calibrated_batch import CalibratedBatchGate
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, ProtocolMode


@pytest.mark.parametrize("optional_nones", [False, True])
def test_optional_profile_fields_do_not_force_full_for_identical_legacy_contract(optional_nones):
    jobs = build_execution_plan(algorithms=[Algorithm.PSO], protocol_mode=ProtocolMode.REDUCED_VS_FULL, base_config=BaseSearchConfig(), seeds=[1])
    engine = SimpleNamespace(name="fixture", scientific=False, audit_descriptor=lambda: {"source": "test"})
    snapshots = [_job_snapshot(job) for job in jobs]
    if optional_nones:
        for job, snapshot in zip(jobs, snapshots):
            snapshot["profile"] = asdict(job.profile)
    gate = CalibratedBatchGate.__new__(CalibratedBatchGate)
    gate.artifact = {"docking_task": "local_refinement"}
    gate.contract = {"jobs": snapshots, "engine": {"name": engine.name, "scientific": engine.scientific, "descriptor": engine.audit_descriptor()}}
    gate.validate_jobs(jobs, engine)
    assert gate.protocol_issues == []


@pytest.mark.parametrize("options", [{"evaluation_budget": 10100}, {"max_translation": 5.0},
    {"max_rotation_degrees": 25.0}, {"reduced_search_scale": .5}, {"execution_order_seed": 20260925}])
def test_explicit_factorial_controls_in_legacy_launcher_are_rejected(tmp_path, options):
    with pytest.raises(ValueError, match="experimental controls"):
        real_batch_command(input_dir=tmp_path / "input", output_dir=tmp_path / "out", algorithms=["pso"], protocol="full",
            generations=100, population=100, top_pockets=1, seed_start=1, seed_count=1, timeout_seconds=60, **options)

from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, ProtocolMode


def test_multiple_algorithms_compare_mode_builds_all_jobs():
    jobs = build_execution_plan(
        algorithms=[Algorithm.AG, Algorithm.PSO, Algorithm.HYBRID],
        protocol_mode=ProtocolMode.REDUCED_VS_FULL,
        base_config=BaseSearchConfig(100, 100, 32),
        seeds=[1, 2],
    )
    assert len(jobs) == 12
    assert {j.algorithm for j in jobs} == {Algorithm.AG, Algorithm.PSO, Algorithm.HYBRID}
    assert {j.protocol for j in jobs} == {ProtocolMode.REDUCED, ProtocolMode.FULL}

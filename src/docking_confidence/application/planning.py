from __future__ import annotations

from collections.abc import Iterable

from docking_confidence.application.protocols import ProtocolResolver
from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, ExecutionJob, ProtocolMode


def protocol_arms(mode: ProtocolMode) -> tuple[ProtocolMode, ...]:
    if mode == ProtocolMode.FACTORIAL:
        raise ValueError("factorial_2x2 has independent conditions; use build_execution_plan")
    if mode in {ProtocolMode.REDUCED_VS_FULL, ProtocolMode.OPERATIONAL}:
        return (ProtocolMode.REDUCED, ProtocolMode.FULL)
    return (mode,)


def build_execution_plan(
    *,
    algorithms: Iterable[Algorithm],
    protocol_mode: ProtocolMode,
    base_config: BaseSearchConfig,
    seeds: Iterable[int],
    resolver: ProtocolResolver | None = None,
) -> list[ExecutionJob]:
    resolver = resolver or ProtocolResolver()
    algorithms = tuple(dict.fromkeys(algorithms))
    seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not algorithms:
        raise ValueError("select at least one algorithm")
    if not seeds:
        raise ValueError("provide at least one seed")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative")
    if protocol_mode != ProtocolMode.FACTORIAL and base_config.experiment is not None:
        raise ValueError("experimental controls require factorial_2x2 mode")
    if (protocol_mode == ProtocolMode.OPERATIONAL) != (base_config.operational is not None):
        raise ValueError("operational controls require operational_cascade mode")

    jobs: list[ExecutionJob] = []
    for algorithm in algorithms:
        for seed in seeds:
            if protocol_mode == ProtocolMode.FACTORIAL:
                for extent in ("reduced", "expanded"):
                    for initialization in ("uniform", "seeded"):
                        profile = resolver.resolve_condition(extent, initialization, base_config)
                        jobs.append(ExecutionJob(
                            job_id=f"{algorithm.value}-{extent}-{initialization}-seed{seed}",
                            algorithm=algorithm, protocol=profile.mode, seed=seed, profile=profile,
                        ))
                continue
            for arm in protocol_arms(protocol_mode):
                profile = resolver.resolve(arm, base_config)
                jobs.append(
                    ExecutionJob(
                        job_id=f"{algorithm.value}-{'operational-' if base_config.operational is not None else ''}{arm.value}-seed{seed}",
                        algorithm=algorithm,
                        protocol=arm,
                        seed=seed,
                        profile=profile,
                    )
                )
    return jobs

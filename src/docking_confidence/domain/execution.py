from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

FACTORIAL_EVALUATION_BUDGET = 7600
FACTORIAL_DOCKQ_THRESHOLD = 0.23
FACTORIAL_PRIMARY_OUTCOME = "valid_pose_dockq_ge_023"
FIXED_EVALUATION_BUDGET = "fixed_evaluations"
OPERATIONAL_CASCADE = "operational_cascade_v1"


class ProtocolMode(str, Enum):
    FULL = "full"
    REDUCED = "reduced"
    REDUCED_VS_FULL = "reduced_vs_full"
    FACTORIAL = "factorial_2x2"
    OPERATIONAL = "operational_cascade"


class Algorithm(str, Enum):
    AG = "ag"
    PSO = "pso"
    HYBRID = "hybrid"


class DockingTask(str, Enum):
    LOCAL_REFINEMENT = "local_refinement"
    SURFACE_DISCOVERY = "surface_discovery"


@dataclass(frozen=True)
class FactorialExperimentConfig:
    """Independent factors with the same number of effective evaluations."""

    evaluation_budget: int = FACTORIAL_EVALUATION_BUDGET
    reduced_search_scale: float = 0.5
    max_translation: float = 5.0
    max_rotation_degrees: float = 25.0
    pose_source: str = ""
    execution_order_seed: int = 20260925

    def validate(self, population: int) -> None:
        if isinstance(self.evaluation_budget, bool) or not isinstance(self.evaluation_budget, int) or self.evaluation_budget < population:
            raise ValueError("common evaluation budget must be an integer >= population")
        if isinstance(self.reduced_search_scale, bool) or not isinstance(self.reduced_search_scale, (int, float)) or not math.isfinite(self.reduced_search_scale) or not 0 < self.reduced_search_scale < 1:
            raise ValueError("reduced_search_scale must be finite and strictly between 0 and 1")
        for name in ("max_translation", "max_rotation_degrees"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_rotation_degrees > 180:
            raise ValueError("max_rotation_degrees must be <= 180")
        if not isinstance(self.pose_source, str) or not self.pose_source.strip():
            raise ValueError("describe the supplied pose origin and its relationship to the reference")
        if isinstance(self.execution_order_seed, bool) or not isinstance(self.execution_order_seed, int) or self.execution_order_seed < 0:
            raise ValueError("execution_order_seed must be a nonnegative integer")


@dataclass(frozen=True)
class OperationalSearchConfig:
    """Separate runtime regime; never changes the historical/factorial recipes."""

    full_evaluation_budget: int = 7600
    reduced_search_scale: float = 0.5
    max_translation: float = 5.0
    max_rotation_degrees: float = 25.0
    initialization: str = "uniform"
    pose_source: str = "Supplied partner models; origin relative to reference must be reported separately."

    @property
    def reduced_evaluation_budget(self) -> int:
        return self.full_evaluation_budget * 3 // 4

    def validate(self, population: int) -> None:
        if (isinstance(self.full_evaluation_budget, bool)
                or not isinstance(self.full_evaluation_budget, int)
                or self.full_evaluation_budget % 4 != 0):
            raise ValueError("operational FULL budget must be an integer divisible by 4 for an exact 25% reduction")
        FactorialExperimentConfig(
            evaluation_budget=self.reduced_evaluation_budget,
            reduced_search_scale=self.reduced_search_scale,
            max_translation=self.max_translation,
            max_rotation_degrees=self.max_rotation_degrees,
            pose_source=self.pose_source,
        ).validate(population)
        if self.initialization not in {"uniform", "seeded"}:
            raise ValueError("operational initialization must be uniform or seeded")


@dataclass(frozen=True)
class BaseSearchConfig:
    generations: int = 100
    population: int = 100
    top_pockets: int = 32
    docking_task: DockingTask = DockingTask.LOCAL_REFINEMENT
    experiment: FactorialExperimentConfig | None = None
    operational: OperationalSearchConfig | None = None

    def validate(self) -> None:
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if self.population < 1:
            raise ValueError("population must be >= 1")
        if self.top_pockets < 1:
            raise ValueError("top_pockets must be >= 1")
        DockingTask(self.docking_task)
        if self.operational is not None:
            if self.experiment is not None:
                raise ValueError("operational and factorial configurations are mutually exclusive")
            self.operational.validate(self.population)
            if DockingTask(self.docking_task) != DockingTask.LOCAL_REFINEMENT or self.top_pockets != 1:
                raise ValueError("operational cascade requires local_refinement with one shared pocket")
        if self.experiment is not None:
            self.experiment.validate(self.population)
            if DockingTask(self.docking_task) != DockingTask.LOCAL_REFINEMENT:
                raise ValueError("factorial_2x2 requires local_refinement with a shared input frame")


@dataclass(frozen=True)
class ProtocolProfile:
    mode: ProtocolMode
    generations: int
    population: int
    top_pockets: int
    evaluation_budget: int
    generation_factor: float
    pocket_factor: float
    docking_task: DockingTask = DockingTask.LOCAL_REFINEMENT
    search_extent: str | None = None
    initialization: str | None = None
    budget_policy: str | None = None
    search_scale: float | None = None
    max_translation: float | None = None
    max_rotation_degrees: float | None = None
    pose_source: str | None = None
    execution_order_seed: int | None = None
    execution_strategy: str | None = None


@dataclass(frozen=True)
class ExecutionJob:
    job_id: str
    algorithm: Algorithm
    protocol: ProtocolMode
    seed: int
    profile: ProtocolProfile


@dataclass(frozen=True)
class DockingRunResult:
    job_id: str
    algorithm: str
    protocol: str
    seed: int
    generations: int
    population: int
    top_pockets: int
    evaluation_budget: int
    runtime_sec: float
    dockq: float | None
    scan_gap_norm: float | None = None
    viable_fraction_mean: float | None = None
    elite_dispersion: float | None = None
    uncertainty_count: float | None = None
    energy_q20_norm: float | None = None
    fnat: float | None = None
    irmsd: float | None = None
    lrmsd: float | None = None
    pose_artifact_path: str | None = None
    complex_pose_pdb_path: str | None = None
    reference_metrics_path: str | None = None
    graph_features_path: str | None = None
    engine_summary_path: str | None = None
    engine_result_path: str | None = None
    resolved_config_path: str | None = None
    stdout_log_path: str | None = None
    stderr_log_path: str | None = None
    exit_code: int | None = None
    engine: str = "unknown"
    scientific: bool = False
    actual_eval_count: int | None = None
    stop_reason: str | None = None
    initial_population_sha256: str | None = None
    execution_audit_sha256: str | None = None

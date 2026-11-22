from __future__ import annotations

import math

from docking_confidence.domain.execution import (
    BaseSearchConfig,
    DockingTask,
    FIXED_EVALUATION_BUDGET,
    OPERATIONAL_CASCADE,
    ProtocolMode,
    ProtocolProfile,
)


class ProtocolResolver:
    """Single source of truth for FULL and REDUCED computational budgets."""

    REDUCED_GENERATION_FACTOR = 0.75
    REDUCED_POCKET_FACTOR = 0.50

    def resolve(self, mode: ProtocolMode, base: BaseSearchConfig) -> ProtocolProfile:
        base.validate()
        if base.experiment is not None:
            raise ValueError("use resolve_condition for independent experimental controls")
        if base.operational is not None:
            return self.resolve_operational(mode, base)
        if mode in (ProtocolMode.REDUCED_VS_FULL, ProtocolMode.FACTORIAL, ProtocolMode.OPERATIONAL):
            raise ValueError("REDUCED_VS_FULL is a planning mode, not an executable profile")

        if mode == ProtocolMode.REDUCED:
            generations = max(1, math.ceil(base.generations * self.REDUCED_GENERATION_FACTOR))
            top_pockets = max(1, math.ceil(base.top_pockets * self.REDUCED_POCKET_FACTOR))
            generation_factor = self.REDUCED_GENERATION_FACTOR
            pocket_factor = self.REDUCED_POCKET_FACTOR
        else:
            generations = base.generations
            top_pockets = base.top_pockets
            generation_factor = 1.0
            pocket_factor = 1.0

        budget = (generations + 1) * base.population
        return ProtocolProfile(
            mode=mode,
            generations=generations,
            population=base.population,
            top_pockets=top_pockets,
            evaluation_budget=budget,
            generation_factor=generation_factor,
            pocket_factor=pocket_factor,
            docking_task=DockingTask(base.docking_task),
        )

    def resolve_operational(self, mode: ProtocolMode, base: BaseSearchConfig) -> ProtocolProfile:
        base.validate()
        controls = base.operational
        if controls is None or mode not in {ProtocolMode.REDUCED, ProtocolMode.FULL}:
            raise ValueError("operational profiles require explicit controls and a FULL/REDUCED arm")
        reduced = mode == ProtocolMode.REDUCED
        return ProtocolProfile(
            mode=mode, generations=base.generations, population=base.population,
            top_pockets=1, evaluation_budget=(controls.reduced_evaluation_budget
                if reduced else controls.full_evaluation_budget),
            generation_factor=1.0, pocket_factor=1.0,
            docking_task=DockingTask.LOCAL_REFINEMENT,
            search_extent="reduced" if reduced else "expanded",
            initialization=controls.initialization, budget_policy=FIXED_EVALUATION_BUDGET,
            search_scale=controls.reduced_search_scale if reduced else 1.0,
            max_translation=controls.max_translation,
            max_rotation_degrees=controls.max_rotation_degrees,
            pose_source=controls.pose_source, execution_strategy=OPERATIONAL_CASCADE,
        )

    def resolve_condition(self, extent: str, initialization: str, base: BaseSearchConfig) -> ProtocolProfile:
        base.validate()
        experiment = base.experiment
        if experiment is None:
            raise ValueError("factorial_2x2 requires explicit experimental controls")
        if extent not in ("reduced", "expanded") or initialization not in ("seeded", "uniform"):
            raise ValueError("invalid factorial condition")
        return ProtocolProfile(
            mode=ProtocolMode.REDUCED if extent == "reduced" else ProtocolMode.FULL,
            generations=base.generations, population=base.population, top_pockets=base.top_pockets,
            evaluation_budget=experiment.evaluation_budget, generation_factor=1.0, pocket_factor=1.0,
            docking_task=DockingTask(base.docking_task), search_extent=extent,
            initialization=initialization, budget_policy=FIXED_EVALUATION_BUDGET,
            search_scale=experiment.reduced_search_scale if extent == "reduced" else 1.0,
            max_translation=experiment.max_translation, max_rotation_degrees=experiment.max_rotation_degrees,
            pose_source=experiment.pose_source.strip(), execution_order_seed=experiment.execution_order_seed,
        )

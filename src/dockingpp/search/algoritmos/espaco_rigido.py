"""Representação rígida 6D, transformação de poses e cache de avaliações."""

from __future__ import annotations

from typing import Any

import numpy as np

from dockingpp.domain.models import Pose
from dockingpp.application.ports.scoring import CheapObjective
from dockingpp.search.controlador_populacional import pose_centroid
from dockingpp.utils.geometry import apply_transform, place_surface_contact, surface_anchor_normal

from .comum import _safe_float


class ObjectiveEvaluationBudgetExhausted(RuntimeError):
    """Indica que o orçamento científico de chamadas objetivas foi consumido."""


class RigidSearchSpace:
    """Converte vetores ``[tx, ty, tz, rx, ry, rz]`` em poses avaliáveis.

    A classe concentra regras que são idênticas nos três motores: limites,
    âncora, rotação Euler, cache por oito casas e direção ``fitness = -score``.
    """

    dim = 6

    def __init__(
        self,
        *,
        cfg: Any,
        rng: np.random.Generator,
        pocket: Any,
        base_coords: np.ndarray,
        score_cheap_fn: CheapObjective,
        max_trans: float,
        max_rot_deg: float,
    ) -> None:
        self.cfg = cfg
        self.rng = rng
        self.pocket = pocket
        self.base_coords = np.asarray(base_coords, dtype=float)
        self.score_cheap_fn = score_cheap_fn
        self.docking_task = str(getattr(cfg, "docking_task", "surface_discovery"))
        if self.docking_task not in {"surface_discovery", "local_refinement"}:
            raise ValueError("unsupported docking_task")
        self.local_refinement = self.docking_task == "local_refinement"
        self.search_scale = 1.0
        if self.local_refinement and getattr(cfg, "search_space_mode", "full") == "reduced":
            self.search_scale = float(getattr(cfg, "local_refinement_reduced_scale", 0.5))
            if not 0.0 < self.search_scale <= 1.0:
                raise ValueError("local_refinement_reduced_scale must be in (0, 1]")
        max_trans *= self.search_scale
        max_rot_deg *= self.search_scale
        self.domain_low = np.array(
            [-max_trans, -max_trans, -max_trans, -max_rot_deg, -max_rot_deg, -max_rot_deg],
            dtype=float,
        )
        self.domain_high = np.array(
            [max_trans, max_trans, max_trans, max_rot_deg, max_rot_deg, max_rot_deg],
            dtype=float,
        )
        self.base_centroid = pose_centroid(self.base_coords)
        requested_anchor_mode = str(
            getattr(cfg, "search_anchor_mode", "auto") or "auto"
        ).strip().lower()
        if self.local_refinement:
            if requested_anchor_mode not in {"auto", "input_centroid"}:
                raise ValueError("local_refinement requires auto or input_centroid anchor")
            self.anchor_mode = "input_centroid"
        elif requested_anchor_mode == "auto":
            search_space_mode = str(
                getattr(cfg, "search_space_mode", "full") or "full"
            ).strip().lower()
            self.anchor_mode = (
                "pocket_center" if search_space_mode == "reduced" else "input_centroid"
            )
        elif requested_anchor_mode in {"pocket_center", "input_centroid"}:
            self.anchor_mode = requested_anchor_mode
        else:
            raise ValueError(
                "search_anchor_mode must be one of: auto, pocket_center, input_centroid"
            )
        self.anchor_center = (
            np.asarray(
                getattr(pocket, "center", self.base_centroid),
                dtype=float,
            ).reshape(3)
            if self.anchor_mode == "pocket_center"
            else self.base_centroid.copy()
        )
        self.surface_normal = None
        if requested_anchor_mode == "auto" and self.anchor_mode == "pocket_center":
            self.surface_normal = surface_anchor_normal(pocket)
            if self.surface_normal is not None:
                self.anchor_mode = "surface_contact"
        self.evaluation_cache: dict[
            tuple[float, ...], tuple[float, dict[str, Any]]
        ] = {}
        self.actual_eval_count = 0
        requested_budget = getattr(
            cfg,
            "objective_evaluation_budget_per_pocket",
            None,
        )
        if requested_budget is None:
            requested_budget = getattr(cfg, "max_objective_evaluations", None)
        self.evaluation_budget = (
            None if requested_budget is None else int(requested_budget)
        )
        if self.evaluation_budget is not None and self.evaluation_budget <= 0:
            raise ValueError(
                "objective_evaluation_budget_per_pocket deve ser maior que zero"
            )

    @property
    def remaining_evaluations(self) -> int | None:
        """Número de novas chamadas ainda permitido, ou ``None`` sem limite."""

        if self.evaluation_budget is None:
            return None
        return max(0, int(self.evaluation_budget) - int(self.actual_eval_count))

    @property
    def evaluation_budget_exhausted(self) -> bool:
        remaining = self.remaining_evaluations
        return remaining is not None and remaining <= 0

    @staticmethod
    def euler_rotation_matrix(rot_deg: np.ndarray) -> np.ndarray:
        """Monta a matriz de rotação rígida para ângulos Euler XYZ em graus."""

        rx, ry, rz = np.deg2rad(np.asarray(rot_deg, dtype=float)[:3])
        cx, sx = float(np.cos(rx)), float(np.sin(rx))
        cy, sy = float(np.cos(ry)), float(np.sin(ry))
        cz, sz = float(np.cos(rz)), float(np.sin(rz))
        rot_x = np.array(
            [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
            dtype=float,
        )
        rot_y = np.array(
            [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
            dtype=float,
        )
        rot_z = np.array(
            [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        return rot_z @ rot_y @ rot_x

    def solution_to_pose(
        self,
        solution: np.ndarray,
        score: float | None = None,
    ) -> Pose:
        solution = np.clip(
            np.asarray(solution, dtype=float),
            self.domain_low,
            self.domain_high,
        )
        rotation = self.euler_rotation_matrix(solution[3:6])
        centered_coords = self.base_coords - self.base_centroid.reshape(1, 3)
        coords = apply_transform(
            centered_coords,
            rotation,
            self.anchor_center + solution[:3],
        )
        meta = {
            "docking_task": self.docking_task,
            "input_relative_placement_used": bool(self.local_refinement or self.anchor_mode == "input_centroid"),
            "search_domain_low": self.domain_low.tolist(),
            "search_domain_high": self.domain_high.tolist(),
            "local_refinement_search_scale": self.search_scale,
            "solution_vector": [float(value) for value in solution],
            "translation": [float(value) for value in solution[:3]],
            "rotation_deg": [float(value) for value in solution[3:6]],
            "input_centroid": [float(value) for value in self.base_centroid],
            "search_anchor_center": [float(value) for value in self.anchor_center],
            "search_anchor_mode": self.anchor_mode,
            "rotation_center": [
                float(value) for value in self.anchor_center + solution[:3]
            ],
            "transform_semantics": (
                "center_input_rotate_then_place_at_search_anchor_plus_offset"
            ),
        }
        if self.surface_normal is not None:
            contact_point = self.anchor_center + solution[:3]
            coords, support_index = place_surface_contact(
                self.base_coords, rotation, contact_point, self.surface_normal,
            )
            meta.update(
                surface_anchor_version="surface_contact_v1",
                surface_contact_point=contact_point.tolist(),
                surface_support_atom_index=support_index,
                surface_normal=self.surface_normal.tolist(),
                rotation_center=coords.mean(axis=0).tolist(),
                transform_semantics="rotate_input_then_place_surface_support_at_pocket_anchor_plus_offset",
            )
        return Pose(coords=coords, score_cheap=score, meta=meta)

    def random_solution(self) -> np.ndarray:
        return self.rng.uniform(self.domain_low, self.domain_high, size=self.dim)

    def random_pose(self) -> Pose:
        return self.solution_to_pose(self.random_solution())

    def evaluate_pose_score(self, pose: Pose) -> float:
        if self.evaluation_budget_exhausted:
            raise ObjectiveEvaluationBudgetExhausted(
                "orçamento de avaliações da função objetivo esgotado"
            )
        self.actual_eval_count += 1
        return _safe_float(
            self.score_cheap_fn(pose, self.pocket, self.cfg.cheap_weights),
            default=0.0,
        )

    def evaluate_solution(self, solution: np.ndarray) -> tuple[Pose, float, float]:
        solution = np.clip(
            np.asarray(solution, dtype=float),
            self.domain_low,
            self.domain_high,
        )
        key = tuple(float(value) for value in np.round(solution, decimals=8))
        pose = self.solution_to_pose(solution)
        cached = self.evaluation_cache.get(key)
        if cached is None:
            score = self.evaluate_pose_score(pose)
            self.evaluation_cache[key] = (float(score), dict(pose.meta or {}))
        else:
            score, cached_meta = cached
            pose.meta.update(dict(cached_meta))
        pose.score_cheap = float(score)
        return pose, score, -float(score)

    def existing_solution(self, pose: Pose) -> np.ndarray:
        meta = pose.meta if isinstance(pose.meta, dict) else {}
        solution = meta.get("solution_vector")
        if solution is not None:
            array = np.asarray(solution, dtype=float)
            if array.shape == (self.dim,) and np.all(np.isfinite(array)):
                return np.clip(array, self.domain_low, self.domain_high)
        return self.random_solution()

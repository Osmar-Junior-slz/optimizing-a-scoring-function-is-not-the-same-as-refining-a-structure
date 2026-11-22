"""Selecao de bolsoes para a execucao do pipeline."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from dockingpp.data.io import load_pockets
from dockingpp.domain.models import Pocket
from dockingpp.pipeline.scan import (
    build_receptor_kdtree,
    scan_pocket_feasibility,
    scan_pocket_feasibility_geom_kdtree,
    select_pockets_from_scan,
)
from dockingpp.priors.pocket import rank_pockets
from dockingpp.pipeline.confianca.features import build_single_confidence_metadata
from dockingpp.pipeline.confianca.policy import evaluate_confidence_policy, merge_confidence_policy_metadata


class _ScanConfigOverlay(dict):
    """Overlay dict for scan config preserving fallback reads from the base config."""

    def __init__(self, base_cfg: Any, overrides: dict[str, Any]) -> None:
        super().__init__(overrides)
        self._base_cfg = base_cfg

    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            return super().get(key, default)
        if self._base_cfg is None:
            return default
        if isinstance(self._base_cfg, dict):
            return self._base_cfg.get(key, default)
        return getattr(self._base_cfg, key, default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_positive_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    parsed = _safe_int(value, 0)
    if parsed <= 0:
        return None
    return parsed


def _derive_pocket_scan_seed(base_seed: int, pocket_id: str) -> int:
    """Derive a stable per-pocket seed without depending on list order.

    A single shared RNG makes the samples of a pocket change whenever another
    candidate is inserted, removed or stops early.  Deriving the seed from the
    audited base seed and pocket id keeps each pocket's experiment reproducible.
    """

    payload = f"dockingpp-scan-v1:{int(base_seed)}:{str(pocket_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


@dataclass(frozen=True)
class ContextoSelecaoBolsoes:
    """Estado consolidado da selecao de bolsoes para o pipeline."""

    search_space_mode: str
    modo_legado_pockets: bool
    candidate_pockets: list[Pocket]
    selected_pockets: list[Pocket]
    pockets: list[Pocket]
    total_pockets: int
    scan_results: dict[str, dict[str, Any]]
    scan_params: dict[str, Any]
    pocketing_time: float
    scan_time: float
    accepted_pockets: list[tuple[int, Pocket]]
    feasible_pockets: list[tuple[int, Pocket]]
    rejected: list[dict[str, Any]]
    selection_confidence_metadata: dict[str, Any] = field(default_factory=dict)
    selection_confidence_policy: dict[str, Any] = field(default_factory=dict)


class SelecionadorBolsoesPipeline:
    """Encapsula selecao de bolsoes preservando comportamento existente."""

    def __init__(
        self,
        *,
        normalizar_modo_busca: Callable[[str | None, bool], str],
        obter_valor_cfg: Callable[[Any, str, Any], Any],
        extrair_coords: Callable[[Any], np.ndarray],
        aplicar_reducao_condicionada: Callable[..., list[Pocket]],
    ) -> None:
        self._normalizar_modo_busca = normalizar_modo_busca
        self._obter_valor_cfg = obter_valor_cfg
        self._extrair_coords = extrair_coords
        self._aplicar_reducao_condicionada = aplicar_reducao_condicionada

    def selecionar(
        self,
        *,
        cfg: Any,
        receptor: Any,
        peptide: Any,
        dummy_pockets: list[Pocket],
        tracer: Any,
        debug_logger: Any,
    ) -> ContextoSelecaoBolsoes:
        """Seleciona bolsoes para execucao mantendo semantica atual."""

        modo_requisitado_bruto = getattr(cfg, "search_space_mode", None)
        modo_legado_pockets = str(modo_requisitado_bruto or "").strip().lower() == "pockets"
        search_space_mode = self._normalizar_modo_busca(
            modo_requisitado_bruto,
            bool(getattr(cfg, "full_search", True)),
        )
        cfg.search_space_mode = search_space_mode
        cfg.full_search = search_space_mode == "full"
        tracer.search_space_mode = search_space_mode
        tracer.event(
            stage="config",
            event_type="config_loaded",
            substage="raw",
            payload={
                "requested_mode": modo_requisitado_bruto,
                "budget_policy": getattr(cfg, "budget_policy", "split"),
            },
            level="BASIC",
        )
        tracer.event(
            stage="config",
            event_type="config_normalized",
            substage="mode",
            payload={
                "search_space_mode": search_space_mode,
                "full_search": bool(cfg.full_search),
                "compare_policy": "best_pocket_vs_full",
            },
            level="BASIC",
            decision=True,
        )
        debug_logger.log({"type": "config_resolved", "search_space_mode": search_space_mode})

        pocketing_start = time.perf_counter()
        local_refinement = getattr(cfg, "docking_task", "surface_discovery") == "local_refinement"
        if local_refinement:
            # The supplied complex defines an explicit local refinement task.
            # Use the complete receptor for scoring in both arms, and let the
            # rigid search space apply the audited nested motion bounds.
            from dockingpp.pipeline import run as run_mod

            local_pocket = run_mod._build_global_pocket(receptor, cfg)
            local_pocket.id = "local_refinement"
            local_pocket.meta["detection_mode"] = "supplied_pose_local_refinement"
            local_pocket.meta["input_relative_placement_used"] = True
            loaded_pockets = [local_pocket]
        elif search_space_mode == "full" and not modo_legado_pockets:
            from dockingpp.pipeline import run as run_mod

            loaded_pockets = [run_mod._build_global_pocket(receptor, cfg)]
        else:
            loaded_pockets = dummy_pockets or load_pockets(
                receptor,
                cfg=cfg,
                pockets_path=getattr(cfg, "pockets_path", None),
            )
        tracer.event(
            stage="pocket_filter",
            event_type="pockets_loaded",
            payload={"loaded_pockets": [str(p.id) for p in loaded_pockets], "n_loaded": int(len(loaded_pockets))},
            level="TRACE",
        )
        debug_logger.log(
            {
                "type": "pockets_loaded",
                "loaded": [str(p.id) for p in loaded_pockets],
                "n_loaded": int(len(loaded_pockets)),
            }
        )

        if local_refinement or (search_space_mode == "full" and not modo_legado_pockets):
            ranked_pockets = list(loaded_pockets)
        else:
            ranked = rank_pockets(
                receptor,
                loaded_pockets,
                peptide=peptide,
                debug_logger=debug_logger,
                cfg=cfg,
            )
            ranked_pockets = [p for p, _ in ranked] if ranked else list(loaded_pockets)

        n_pockets_input = int(len(loaded_pockets))
        n_pockets_after_rank = int(len(ranked_pockets))
        total_pockets = len(ranked_pockets)
        candidate_pockets = list(ranked_pockets)
        if search_space_mode != "full":
            top_pockets = int(getattr(cfg, "top_pockets", len(candidate_pockets)) or len(candidate_pockets))
            if top_pockets > 0:
                candidate_pockets = candidate_pockets[:top_pockets]

        pocketing_time = time.perf_counter() - pocketing_start
        tracer.event(
            stage="pocket_filter",
            event_type="pockets_ranked",
            payload={"ranked_pockets": [str(p.id) for p in ranked_pockets], "n_ranked": int(len(ranked_pockets))},
            level="TRACE",
            decision=True,
        )
        debug_logger.log(
            {
                "type": "pocket_ranking",
                "ranked": [str(p.id) for p in ranked_pockets],
                "n_ranked": int(len(ranked_pockets)),
            }
        )
        tracer.event(
            stage="pocket_filter",
            event_type="pocket_selected",
            payload={"selected_pockets": [str(p.id) for p in candidate_pockets], "n_pockets_total": int(total_pockets)},
            level="BASIC",
            decision=True,
        )
        debug_logger.log(
            {
                "type": "pocket_selection",
                "selected": [str(p.id) for p in candidate_pockets],
                "n_total": int(total_pockets),
            }
        )

        scan_cfg = self._obter_valor_cfg(cfg, "scan", None)
        # Surface viability must use the same final collision veto as search.
        # Keep the original fractional proxy as an additional filter, not as a
        # replacement for the whole-pose hard gate.
        final_gate_cfg = self._obter_valor_cfg(cfg, "rescoring", {}) or {}
        scan_cfg = _ScanConfigOverlay(scan_cfg, {
            "surface_anchor_enabled": str(getattr(cfg, "search_anchor_mode", "auto") or "auto").lower() == "auto",
            "hard_clash_gate_enabled": self._obter_valor_cfg(final_gate_cfg, "hard_clash_gate_enabled", True),
            "hard_clash_min_distance": self._obter_valor_cfg(final_gate_cfg, "hard_clash_min_distance", 1.2),
        })
        scan_enabled = bool(self._obter_valor_cfg(scan_cfg, "enabled", False)) and search_space_mode == "reduced" and not local_refinement
        selector_mode = str(self._obter_valor_cfg(scan_cfg, "selector_mode", "legacy") or "legacy").strip().lower()
        if selector_mode not in {"legacy", "geom_kdtree"}:
            selector_mode = "legacy"

        scan_selected_pockets = list(candidate_pockets)
        scan_results: dict[str, dict[str, Any]] = {}
        scan_time_per_pocket: dict[str, float] = {}
        scan_seed_by_pocket: dict[str, int] = {}
        n_scan_candidates = int(len(scan_selected_pockets))

        samples_per_pocket_requested = max(0, _safe_int(self._obter_valor_cfg(scan_cfg, "samples_per_pocket", 64), 64))
        max_total_samples = _safe_positive_int_or_none(self._obter_valor_cfg(scan_cfg, "max_total_samples", None))
        min_effective_samples_per_pocket = max(
            1,
            _safe_int(self._obter_valor_cfg(scan_cfg, "min_effective_samples_per_pocket", 8), 8),
        )
        multi_pocket_hard_cap = _safe_positive_int_or_none(self._obter_valor_cfg(scan_cfg, "multi_pocket_hard_cap", None))
        skip_scan_when_single_pocket = bool(self._obter_valor_cfg(scan_cfg, "skip_scan_when_single_pocket", True))
        single_candidate_fastpath_applicable = bool(skip_scan_when_single_pocket and n_scan_candidates <= 1)
        samples_per_pocket_budget_cap: int | None = None
        samples_per_pocket_after_hard_cap: int | None = None
        samples_per_pocket_effective = int(samples_per_pocket_requested)
        if (
            scan_enabled
            and not single_candidate_fastpath_applicable
            and n_scan_candidates > 0
            and samples_per_pocket_requested > 0
            and max_total_samples is not None
        ):
            samples_per_pocket_budget_cap = max(1, int(max_total_samples // n_scan_candidates))
            samples_per_pocket_effective = min(
                int(samples_per_pocket_requested),
                max(int(min_effective_samples_per_pocket), int(samples_per_pocket_budget_cap)),
            )
        if scan_enabled and not single_candidate_fastpath_applicable and n_scan_candidates > 1 and multi_pocket_hard_cap is not None:
            samples_per_pocket_after_hard_cap = min(int(samples_per_pocket_effective), int(multi_pocket_hard_cap))
            samples_per_pocket_effective = max(1, int(samples_per_pocket_after_hard_cap))
        scan_budget_applied = bool(samples_per_pocket_effective != samples_per_pocket_requested)
        multi_pocket_hard_cap_applied = bool(
            n_scan_candidates > 1
            and multi_pocket_hard_cap is not None
            and samples_per_pocket_after_hard_cap is not None
            and samples_per_pocket_after_hard_cap < samples_per_pocket_requested
        )
        scan_cfg_effective = (
            _ScanConfigOverlay(scan_cfg, {"samples_per_pocket": int(samples_per_pocket_effective)})
            if scan_budget_applied
            else scan_cfg
        )

        scan_executed = False
        scan_skipped = False
        scan_skip_reason: str | None = None
        n_pockets_after_scan = int(len(scan_selected_pockets))
        n_pockets_after_reducao_condicionada = int(len(scan_selected_pockets))
        scan_params = {
            "docking_task": getattr(cfg, "docking_task", "surface_discovery"),
            "not_applicable_reason": "supplied_pose_local_refinement" if local_refinement else None,
            "enabled": scan_enabled,
            "selector_mode": selector_mode,
            "samples_per_pocket": int(samples_per_pocket_effective),
            "samples_per_pocket_requested": int(samples_per_pocket_requested),
            "samples_per_pocket_effective": int(samples_per_pocket_effective),
            "samples_per_pocket_budget_cap": int(samples_per_pocket_budget_cap) if samples_per_pocket_budget_cap is not None else None,
            "samples_per_pocket_after_hard_cap": int(samples_per_pocket_after_hard_cap) if samples_per_pocket_after_hard_cap is not None else None,
            "scan_budget_applied": bool(scan_budget_applied),
            "max_total_samples": int(max_total_samples) if max_total_samples is not None else None,
            "min_effective_samples_per_pocket": int(min_effective_samples_per_pocket),
            "multi_pocket_hard_cap": int(multi_pocket_hard_cap) if multi_pocket_hard_cap is not None else None,
            "multi_pocket_hard_cap_applied": bool(multi_pocket_hard_cap_applied),
            "skip_scan_when_single_pocket": bool(skip_scan_when_single_pocket),
            "single_candidate_fastpath_applicable": bool(single_candidate_fastpath_applicable),
            "max_clash_ratio": self._obter_valor_cfg(scan_cfg, "max_clash_ratio", None),
            "select_top_k": self._obter_valor_cfg(scan_cfg, "select_top_k", None),
            "reject_if_feasible_fraction_leq": self._obter_valor_cfg(scan_cfg, "reject_if_feasible_fraction_leq", 0.0),
            "reject_if_severe_clash_fraction_geq": self._obter_valor_cfg(scan_cfg, "reject_if_severe_clash_fraction_geq", 0.95),
            "allow_relaxed_acceptance_when_empty": self._obter_valor_cfg(scan_cfg, "allow_relaxed_acceptance_when_empty", False),
            "rng_policy": "sha256(base_seed,pocket_id)_independent_v1",
            "seed_offset": int(self._obter_valor_cfg(scan_cfg, "seed_offset", 0) or 0),
        }
        scan_start = time.perf_counter()
        if scan_enabled and scan_selected_pockets:
            if skip_scan_when_single_pocket and len(scan_selected_pockets) <= 1:
                scan_skipped = True
                scan_skip_reason = "single_candidate_fastpath"
                tracer.event(
                    stage="scan",
                    event_type="scan_skipped",
                    payload={
                        "reason": scan_skip_reason,
                        "selector_mode": selector_mode,
                        "n_scan_candidates": int(n_scan_candidates),
                    },
                    level="BASIC",
                    decision=True,
                )
                debug_logger.log(
                    {
                        "type": "scan_skipped",
                        "reason": scan_skip_reason,
                        "selector_mode": selector_mode,
                        "n_scan_candidates": int(n_scan_candidates),
                    }
                )
            else:
                scan_executed = True
                receptor_coords = self._extrair_coords(receptor)
                peptide_coords = self._extrair_coords(peptide)
                receptor_kdtree = build_receptor_kdtree(receptor_coords)
                scan_base_seed = int(cfg.seed) + int(self._obter_valor_cfg(scan_cfg, "seed_offset", 0) or 0)
                for pocket in scan_selected_pockets:
                    scan_single_start = time.perf_counter()
                    pocket_id = str(pocket.id)
                    pocket_seed = _derive_pocket_scan_seed(scan_base_seed, pocket_id)
                    scan_seed_by_pocket[pocket_id] = pocket_seed
                    rng = np.random.default_rng(pocket_seed)
                    if selector_mode == "geom_kdtree":
                        scan_results[pocket_id] = scan_pocket_feasibility_geom_kdtree(
                            receptor_kdtree,
                            receptor_coords,
                            peptide_coords,
                            pocket,
                            scan_cfg_effective,
                            rng,
                        )
                    else:
                        scan_results[pocket_id] = scan_pocket_feasibility(
                            receptor_kdtree,
                            peptide_coords,
                            pocket,
                            scan_cfg_effective,
                            rng,
                        )
                    scan_single_time = float(time.perf_counter() - scan_single_start)
                    scan_time_per_pocket[pocket_id] = scan_single_time
                    scan_results[pocket_id]["scan_time_sec"] = scan_single_time
                    scan_results[pocket_id]["scan_seed"] = pocket_seed
                    scan_results[pocket_id]["scan_seed_base"] = scan_base_seed
                    scan_results[pocket_id]["scan_rng_policy"] = "sha256(base_seed,pocket_id)_independent_v1"

                scan_selected_pockets = select_pockets_from_scan(
                    scan_selected_pockets,
                    scan_results,
                    self._obter_valor_cfg(scan_cfg, "select_top_k", None),
                    selector_mode=selector_mode,
                )
                n_pockets_after_scan = int(len(scan_selected_pockets))
                ranking_payload = [
                    {
                        "pocket_id": str(p.id),
                        "rank": int(idx),
                        "score": float(
                            scan_results.get(str(p.id), {}).get(
                                "pocket_scan_score" if selector_mode == "geom_kdtree" else "scan_score",
                                float("-inf"),
                            )
                        ),
                        "feasible_fraction": float(scan_results.get(str(p.id), {}).get("feasible_fraction", 0.0)),
                        "clash_indicator": float(
                            scan_results.get(str(p.id), {}).get(
                                "severe_clash_fraction" if selector_mode == "geom_kdtree" else "clash_ratio_best",
                                1.0,
                            )
                        ),
                        "energy_q20": float(scan_results.get(str(p.id), {}).get("energy_q20", float("inf"))),
                    }
                    for idx, p in enumerate(scan_selected_pockets)
                ]
                tracer.event(
                    stage="scan",
                    event_type="scan_selection_ranked",
                    payload={"selector_mode": selector_mode, "ranking": ranking_payload},
                    level="TRACE",
                    decision=True,
                )
                tracer.event(
                    stage="scan",
                    event_type="scan_selected",
                    payload={
                        "selector_mode": selector_mode,
                        "selected_pockets": [str(p.id) for p in scan_selected_pockets],
                    },
                    level="TRACE",
                    decision=True,
                )
                debug_logger.log(
                    {
                        "type": "scan_selected",
                        "selector_mode": selector_mode,
                        "selected": [str(p.id) for p in scan_selected_pockets],
                    }
                )
        scan_time = time.perf_counter() - scan_start

        if search_space_mode == "reduced" and not local_refinement and bool(getattr(cfg, "usar_reducao_condicionada_ao_peptideo", False)):
            tracer.event(
                stage="pocket_filter",
                event_type="reducao_condicionada_ativada",
                payload={"mensagem": "Aplicando subpipeline opcional de reducao guiada por peptideo."},
                level="BASIC",
            )
            scan_selected_pockets = self._aplicar_reducao_condicionada(
                peptide=peptide,
                pockets=scan_selected_pockets,
                cfg=cfg,
                tracer=tracer,
                debug_logger=debug_logger,
            )
        n_pockets_after_reducao_condicionada = int(len(scan_selected_pockets))

        accepted_pockets: list[tuple[int, Pocket]] = [(idx, pocket) for idx, pocket in enumerate(scan_selected_pockets)]
        rejected: list[dict[str, Any]] = []
        relaxed_acceptance_applied = False
        relaxed_acceptance_reason: str | None = None
        relaxed_acceptance_pocket_id: str | None = None
        if search_space_mode == "reduced" and not modo_legado_pockets:
            accepted_pockets = []
            max_clash_ratio = float(self._obter_valor_cfg(scan_cfg, "max_clash_ratio", 0.02))
            reject_if_feasible_fraction_leq = float(self._obter_valor_cfg(scan_cfg, "reject_if_feasible_fraction_leq", 0.0))
            reject_if_severe_clash_fraction_geq = float(self._obter_valor_cfg(scan_cfg, "reject_if_severe_clash_fraction_geq", 0.95))
            for idx, pocket in enumerate(scan_selected_pockets):
                metrics = scan_results.get(str(pocket.id), {})
                feasible_fraction = float(metrics.get("feasible_fraction", 1.0))
                if selector_mode == "geom_kdtree":
                    severe_clash_fraction = float(metrics.get("severe_clash_fraction", 0.0))
                    if (
                        feasible_fraction <= reject_if_feasible_fraction_leq
                        and severe_clash_fraction >= reject_if_severe_clash_fraction_geq
                    ):
                        reason = "geom_kdtree_clearly_inviable"
                        rejected.append({"pocket_id": pocket.id, "reason": reason})
                        tracer.event(
                            stage="pocket_filter",
                            event_type="pocket_rejected",
                            payload={
                                "selector_mode": selector_mode,
                                "reason": reason,
                                "feasible_fraction": feasible_fraction,
                                "severe_clash_fraction": severe_clash_fraction,
                                "reject_if_feasible_fraction_leq": reject_if_feasible_fraction_leq,
                                "reject_if_severe_clash_fraction_geq": reject_if_severe_clash_fraction_geq,
                                "pocket_scan_score": float(metrics.get("pocket_scan_score", float("-inf"))),
                            },
                            pocket_id=str(pocket.id),
                            level="TRACE",
                            decision=True,
                        )
                        continue
                else:
                    clash_ratio_best = float(metrics.get("clash_ratio_best", 0.0))
                    if feasible_fraction <= 0.0:
                        rejected.append({"pocket_id": pocket.id, "reason": "feasible_fraction<=0.0"})
                        tracer.event(
                            stage="pocket_filter",
                            event_type="pocket_rejected",
                            payload={
                                "selector_mode": selector_mode,
                                "reason": "feasible_fraction<=0.0",
                                "feasible_fraction": feasible_fraction,
                            },
                            pocket_id=str(pocket.id),
                            level="TRACE",
                            decision=True,
                        )
                        continue
                    if clash_ratio_best > max_clash_ratio:
                        rejected.append({"pocket_id": pocket.id, "reason": "clash_ratio_best>max_clash_ratio"})
                        tracer.event(
                            stage="pocket_filter",
                            event_type="pocket_rejected",
                            payload={
                                "selector_mode": selector_mode,
                                "reason": "clash_ratio_best>max_clash_ratio",
                                "clash_ratio_best": clash_ratio_best,
                                "max_clash_ratio": max_clash_ratio,
                            },
                            pocket_id=str(pocket.id),
                            level="TRACE",
                            decision=True,
                        )
                        continue
                tracer.event(
                    stage="pocket_filter",
                    event_type="pocket_metrics_evaluated",
                    payload={"selector_mode": selector_mode, "metrics": metrics, "accepted": True},
                    pocket_id=str(pocket.id),
                    level="TRACE",
                )
                accepted_pockets.append((idx, pocket))

            allow_relaxed_acceptance = bool(
                self._obter_valor_cfg(scan_cfg, "allow_relaxed_acceptance_when_empty", False)
            )
            if not accepted_pockets and scan_selected_pockets and allow_relaxed_acceptance:
                ranked_relaxed: list[tuple[tuple[float, float, float, int], int, Pocket, dict[str, Any]]] = []
                for idx, pocket in enumerate(scan_selected_pockets):
                    metrics = scan_results.get(str(pocket.id), {})
                    feasible_fraction = float(metrics.get("feasible_fraction", 0.0))
                    clash_indicator = float(
                        metrics.get(
                            "severe_clash_fraction" if selector_mode == "geom_kdtree" else "clash_ratio_best",
                            1.0,
                        )
                    )
                    score_raw = metrics.get("pocket_scan_score" if selector_mode == "geom_kdtree" else "scan_score", float("-inf"))
                    try:
                        score_value = float(score_raw)
                    except (TypeError, ValueError):
                        score_value = float("-inf")
                    ranked_relaxed.append(
                        ((-feasible_fraction, clash_indicator, -score_value, idx), idx, pocket, metrics)
                    )
                ranked_relaxed.sort(key=lambda item: item[0])
                _sort_key, idx_escolhido, pocket_escolhido, metrics_escolhido = ranked_relaxed[0]
                accepted_pockets.append((idx_escolhido, pocket_escolhido))
                relaxed_acceptance_applied = True
                relaxed_acceptance_reason = "no_pocket_passed_hard_thresholds"
                relaxed_acceptance_pocket_id = str(pocket_escolhido.id)
                rejected = [
                    item
                    for item in rejected
                    if str(item.get("pocket_id")) != str(pocket_escolhido.id)
                ]
                tracer.event(
                    stage="pocket_filter",
                    event_type="pocket_acceptance_relaxed",
                    payload={
                        "selector_mode": selector_mode,
                        "reason": relaxed_acceptance_reason,
                        "selected_pocket_id": str(pocket_escolhido.id),
                        "selected_metrics": metrics_escolhido,
                    },
                    pocket_id=str(pocket_escolhido.id),
                    level="BASIC",
                    decision=True,
                )
                debug_logger.log(
                    {
                        "type": "pocket_acceptance_relaxed",
                        "selector_mode": selector_mode,
                        "reason": relaxed_acceptance_reason,
                        "selected_pocket_id": str(pocket_escolhido.id),
                    }
                )

        pre_policy_accepted_ids = [str(p.id) for _, p in accepted_pockets]
        best_scan_pocket_id = pre_policy_accepted_ids[0] if pre_policy_accepted_ids else (
            str(scan_selected_pockets[0].id) if scan_selected_pockets else None
        )
        selection_confidence_metadata = build_single_confidence_metadata(
            search_space_mode=search_space_mode,
            best_score_cheap=None,
            best_score_expensive=None,
            total_pockets=total_pockets,
            selected_pockets=len(scan_selected_pockets),
            best_pocket_id=best_scan_pocket_id,
            selected_pocket_ids=[str(p.id) for p in scan_selected_pockets],
            scan_by_pocket=scan_results,
            fallback_to_full=False,
            calibrator=getattr(cfg, "confidence_calibrator", None),
        )
        selection_confidence_policy = evaluate_confidence_policy(
            cfg=cfg,
            confidence_features=selection_confidence_metadata.get("confidence_features"),
            confidence_output=selection_confidence_metadata.get("confidence_output"),
            candidate_pocket_ids=[str(p.id) for p in candidate_pockets],
            selected_pocket_ids=[str(p.id) for p in scan_selected_pockets],
            accepted_pocket_ids=pre_policy_accepted_ids,
        )

        policy_details = selection_confidence_policy.get("confidence_policy_details")
        if not isinstance(policy_details, dict):
            policy_details = {}
            selection_confidence_policy["confidence_policy_details"] = policy_details

        pruned_ids = set(str(pid) for pid in policy_details.get("recommended_pruned_pockets", []) if pid is not None)
        apply_pruning = bool(policy_details.get("apply_conservative_pruning", False) and pruned_ids)
        applied_pruned_ids: list[str] = []
        if apply_pruning:
            kept_accepted: list[tuple[int, Pocket]] = []
            for idx, pocket in accepted_pockets:
                pocket_id = str(pocket.id)
                if pocket_id in pruned_ids:
                    applied_pruned_ids.append(pocket_id)
                    rejected.append({"pocket_id": pocket.id, "reason": "confidence_policy_conservative_prune"})
                    tracer.event(
                        stage="pocket_filter",
                        event_type="pocket_rejected",
                        payload={
                            "selector_mode": selector_mode,
                            "reason": "confidence_policy_conservative_prune",
                            "confidence_policy_mode": selection_confidence_policy.get("confidence_policy_mode"),
                        },
                        pocket_id=pocket_id,
                        level="TRACE",
                        decision=True,
                    )
                    continue
                kept_accepted.append((idx, pocket))
            accepted_pockets = kept_accepted
            selection_confidence_policy["confidence_assisted_selection"] = bool(applied_pruned_ids)
            selection_confidence_policy["confidence_policy_effect"] = (
                "selection_pruned" if applied_pruned_ids else selection_confidence_policy.get("confidence_policy_effect", "none")
            )
            debug_logger.log(
                {
                    "type": "confidence_policy_selection",
                    "mode": selection_confidence_policy.get("confidence_policy_mode"),
                    "applied_pruned_pockets": applied_pruned_ids,
                }
            )

        policy_details["applied_pruned_pockets"] = applied_pruned_ids
        policy_details["accepted_pockets_after_policy"] = [str(p.id) for _, p in accepted_pockets]
        selection_confidence_policy = merge_confidence_policy_metadata(selection_confidence_policy)

        n_pockets_accepted = int(len(accepted_pockets))
        scan_params["telemetry"] = {
            "n_pockets_input": n_pockets_input,
            "n_pockets_after_rank": n_pockets_after_rank,
            "n_pockets_candidates_topk": int(len(candidate_pockets)),
            "n_scan_candidates": int(n_scan_candidates),
            "n_pockets_after_scan": n_pockets_after_scan,
            "n_pockets_after_conditional_reduction": n_pockets_after_reducao_condicionada,
            "n_pockets_accepted": n_pockets_accepted,
            "scan_executed": bool(scan_executed),
            "scan_skipped": bool(scan_skipped),
            "scan_skip_reason": scan_skip_reason,
            "acceptance_relaxed": bool(relaxed_acceptance_applied),
            "acceptance_relaxed_reason": relaxed_acceptance_reason,
            "acceptance_relaxed_pocket_id": relaxed_acceptance_pocket_id,
            "confidence_features_available": bool(selection_confidence_metadata.get("confidence_features_available", False)),
            "confidence_policy": selection_confidence_policy,
            "samples_per_pocket": int(samples_per_pocket_effective),
            "samples_per_pocket_requested": int(samples_per_pocket_requested),
            "samples_per_pocket_effective": int(samples_per_pocket_effective),
            "samples_per_pocket_budget_cap": int(samples_per_pocket_budget_cap) if samples_per_pocket_budget_cap is not None else None,
            "samples_per_pocket_after_hard_cap": int(samples_per_pocket_after_hard_cap) if samples_per_pocket_after_hard_cap is not None else None,
            "scan_budget_applied": bool(scan_budget_applied),
            "max_total_samples": int(max_total_samples) if max_total_samples is not None else None,
            "min_effective_samples_per_pocket": int(min_effective_samples_per_pocket),
            "multi_pocket_hard_cap": int(multi_pocket_hard_cap) if multi_pocket_hard_cap is not None else None,
            "multi_pocket_hard_cap_applied": bool(multi_pocket_hard_cap_applied),
            "skip_scan_when_single_pocket": bool(skip_scan_when_single_pocket),
            "single_candidate_fastpath_applicable": bool(single_candidate_fastpath_applicable),
            "scan_time_total": float(scan_time),
            "scan_time_per_pocket": {str(k): float(v) for k, v in scan_time_per_pocket.items()},
            "scan_seed_base": int(cfg.seed) + int(self._obter_valor_cfg(scan_cfg, "seed_offset", 0) or 0),
            "scan_seed_by_pocket": {str(k): int(v) for k, v in scan_seed_by_pocket.items()},
            "scan_rng_policy": "sha256(base_seed,pocket_id)_independent_v1",
            "score_components_by_pocket": {
                str(pid): (metrics.get("score_components") or {})
                for pid, metrics in scan_results.items()
            },
            "scan_early_stop_by_pocket": {
                str(pid): {
                    "enabled": bool(metrics.get("scan_early_stop_enabled", False)),
                    "triggered": bool(metrics.get("scan_early_stop_triggered", False)),
                    "reason": metrics.get("scan_early_stop_reason"),
                    "sample_idx": metrics.get("scan_early_stop_sample_idx"),
                }
                for pid, metrics in scan_results.items()
            },
        }
        tracer.event(
            stage="scan",
            event_type="scan_telemetry",
            payload=scan_params["telemetry"],
            level="TRACE",
            decision=True,
        )
        debug_logger.log({"type": "scan_telemetry", **scan_params["telemetry"]})

        tracer.event(
            stage="pocket_filter",
            event_type="pocket_acceptance_summary",
            payload={
                "n_candidates": int(len(candidate_pockets)),
                "n_selected_pre_filter": int(len(scan_selected_pockets)),
                "n_accepted": n_pockets_accepted,
                "accepted_pockets": [str(p.id) for _, p in accepted_pockets],
                "rejected_pockets": rejected,
                "confidence_policy_effect": selection_confidence_policy.get("confidence_policy_effect"),
                "confidence_assisted_selection": bool(selection_confidence_policy.get("confidence_assisted_selection", False)),
                "confidence_assisted_fallback": bool(selection_confidence_policy.get("confidence_assisted_fallback", False)),
            },
            level="BASIC",
            decision=True,
        )
        debug_logger.log(
            {
                "type": "pocket_acceptance_summary",
                "n_candidates": int(len(candidate_pockets)),
                "n_selected_pre_filter": int(len(scan_selected_pockets)),
                "n_accepted": n_pockets_accepted,
                "accepted": [str(p.id) for _, p in accepted_pockets],
                "rejected": rejected,
                "confidence_policy_effect": selection_confidence_policy.get("confidence_policy_effect"),
                "confidence_assisted_selection": bool(selection_confidence_policy.get("confidence_assisted_selection", False)),
                "confidence_assisted_fallback": bool(selection_confidence_policy.get("confidence_assisted_fallback", False)),
            }
        )

        return ContextoSelecaoBolsoes(
            search_space_mode=search_space_mode,
            modo_legado_pockets=modo_legado_pockets,
            candidate_pockets=candidate_pockets,
            selected_pockets=scan_selected_pockets,
            pockets=scan_selected_pockets,
            total_pockets=total_pockets,
            scan_results=scan_results,
            scan_params=scan_params,
            pocketing_time=pocketing_time,
            scan_time=scan_time,
            accepted_pockets=accepted_pockets,
            feasible_pockets=accepted_pockets,
            rejected=rejected,
            selection_confidence_metadata=selection_confidence_metadata,
            selection_confidence_policy=selection_confidence_policy,
        )

"""Configuração de orçamento e execução determinística do scan de bolsões."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from dockingpp.domain.models import Pocket


class _ScanConfigOverlay(dict):
    """Overlay que preserva leituras de fallback da configuração base."""

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


@dataclass(frozen=True)
class ConfiguracaoScan:
    """Configuração resolvida e valores de auditoria do orçamento de scan."""

    scan_cfg: Any
    scan_cfg_effective: Any
    enabled: bool
    selector_mode: str
    n_scan_candidates: int
    samples_per_pocket_requested: int
    samples_per_pocket_effective: int
    samples_per_pocket_budget_cap: int | None
    samples_per_pocket_after_hard_cap: int | None
    scan_budget_applied: bool
    max_total_samples: int | None
    min_effective_samples_per_pocket: int
    multi_pocket_hard_cap: int | None
    multi_pocket_hard_cap_applied: bool
    skip_scan_when_single_pocket: bool
    single_candidate_fastpath_applicable: bool
    params: dict[str, Any]


@dataclass(frozen=True)
class ResultadoScan:
    """Saída do scan antes da redução e dos limiares de aceitação."""

    selected_pockets: list[Pocket]
    results: dict[str, dict[str, Any]]
    scan_time: float
    scan_time_per_pocket: dict[str, float]
    n_pockets_after_scan: int
    executed: bool
    skipped: bool
    skip_reason: str | None


def preparar_configuracao_scan(
    *,
    cfg: Any,
    candidate_pockets: list[Pocket],
    search_space_mode: str,
    obter_valor_cfg: Callable[[Any, str, Any], Any],
) -> ConfiguracaoScan:
    """Resolve limites do scan exatamente na ordem histórica."""

    scan_cfg = obter_valor_cfg(cfg, "scan", None)
    scan_enabled = bool(obter_valor_cfg(scan_cfg, "enabled", False)) and (
        search_space_mode == "reduced"
    )
    selector_mode = str(
        obter_valor_cfg(scan_cfg, "selector_mode", "legacy") or "legacy"
    ).strip().lower()
    if selector_mode not in {"legacy", "geom_kdtree"}:
        selector_mode = "legacy"

    n_scan_candidates = int(len(candidate_pockets))
    samples_per_pocket_requested = max(
        0,
        _safe_int(obter_valor_cfg(scan_cfg, "samples_per_pocket", 64), 64),
    )
    max_total_samples = _safe_positive_int_or_none(
        obter_valor_cfg(scan_cfg, "max_total_samples", None)
    )
    min_effective_samples_per_pocket = max(
        1,
        _safe_int(
            obter_valor_cfg(scan_cfg, "min_effective_samples_per_pocket", 8),
            8,
        ),
    )
    multi_pocket_hard_cap = _safe_positive_int_or_none(
        obter_valor_cfg(scan_cfg, "multi_pocket_hard_cap", None)
    )
    skip_scan_when_single_pocket = bool(
        obter_valor_cfg(scan_cfg, "skip_scan_when_single_pocket", True)
    )
    single_candidate_fastpath_applicable = bool(
        skip_scan_when_single_pocket and n_scan_candidates <= 1
    )
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
        samples_per_pocket_budget_cap = max(
            1,
            int(max_total_samples // n_scan_candidates),
        )
        samples_per_pocket_effective = min(
            int(samples_per_pocket_requested),
            max(
                int(min_effective_samples_per_pocket),
                int(samples_per_pocket_budget_cap),
            ),
        )
    if (
        scan_enabled
        and not single_candidate_fastpath_applicable
        and n_scan_candidates > 1
        and multi_pocket_hard_cap is not None
    ):
        samples_per_pocket_after_hard_cap = min(
            int(samples_per_pocket_effective),
            int(multi_pocket_hard_cap),
        )
        samples_per_pocket_effective = max(
            1,
            int(samples_per_pocket_after_hard_cap),
        )
    scan_budget_applied = bool(
        samples_per_pocket_effective != samples_per_pocket_requested
    )
    multi_pocket_hard_cap_applied = bool(
        n_scan_candidates > 1
        and multi_pocket_hard_cap is not None
        and samples_per_pocket_after_hard_cap is not None
        and samples_per_pocket_after_hard_cap < samples_per_pocket_requested
    )
    scan_cfg_effective = (
        _ScanConfigOverlay(
            scan_cfg,
            {"samples_per_pocket": int(samples_per_pocket_effective)},
        )
        if scan_budget_applied
        else scan_cfg
    )
    params = {
        "enabled": scan_enabled,
        "selector_mode": selector_mode,
        "samples_per_pocket": int(samples_per_pocket_effective),
        "samples_per_pocket_requested": int(samples_per_pocket_requested),
        "samples_per_pocket_effective": int(samples_per_pocket_effective),
        "samples_per_pocket_budget_cap": (
            int(samples_per_pocket_budget_cap)
            if samples_per_pocket_budget_cap is not None
            else None
        ),
        "samples_per_pocket_after_hard_cap": (
            int(samples_per_pocket_after_hard_cap)
            if samples_per_pocket_after_hard_cap is not None
            else None
        ),
        "scan_budget_applied": bool(scan_budget_applied),
        "max_total_samples": (
            int(max_total_samples) if max_total_samples is not None else None
        ),
        "min_effective_samples_per_pocket": int(
            min_effective_samples_per_pocket
        ),
        "multi_pocket_hard_cap": (
            int(multi_pocket_hard_cap)
            if multi_pocket_hard_cap is not None
            else None
        ),
        "multi_pocket_hard_cap_applied": bool(multi_pocket_hard_cap_applied),
        "skip_scan_when_single_pocket": bool(skip_scan_when_single_pocket),
        "single_candidate_fastpath_applicable": bool(
            single_candidate_fastpath_applicable
        ),
        "max_clash_ratio": obter_valor_cfg(scan_cfg, "max_clash_ratio", None),
        "select_top_k": obter_valor_cfg(scan_cfg, "select_top_k", None),
        "reject_if_feasible_fraction_leq": obter_valor_cfg(
            scan_cfg,
            "reject_if_feasible_fraction_leq",
            0.0,
        ),
        "reject_if_severe_clash_fraction_geq": obter_valor_cfg(
            scan_cfg,
            "reject_if_severe_clash_fraction_geq",
            0.95,
        ),
        "allow_relaxed_acceptance_when_empty": obter_valor_cfg(
            scan_cfg,
            "allow_relaxed_acceptance_when_empty",
            True,
        ),
    }
    return ConfiguracaoScan(
        scan_cfg=scan_cfg,
        scan_cfg_effective=scan_cfg_effective,
        enabled=scan_enabled,
        selector_mode=selector_mode,
        n_scan_candidates=n_scan_candidates,
        samples_per_pocket_requested=samples_per_pocket_requested,
        samples_per_pocket_effective=samples_per_pocket_effective,
        samples_per_pocket_budget_cap=samples_per_pocket_budget_cap,
        samples_per_pocket_after_hard_cap=samples_per_pocket_after_hard_cap,
        scan_budget_applied=scan_budget_applied,
        max_total_samples=max_total_samples,
        min_effective_samples_per_pocket=min_effective_samples_per_pocket,
        multi_pocket_hard_cap=multi_pocket_hard_cap,
        multi_pocket_hard_cap_applied=multi_pocket_hard_cap_applied,
        skip_scan_when_single_pocket=skip_scan_when_single_pocket,
        single_candidate_fastpath_applicable=single_candidate_fastpath_applicable,
        params=params,
    )


def executar_scan(
    *,
    cfg: Any,
    receptor: Any,
    peptide: Any,
    candidate_pockets: list[Pocket],
    config: ConfiguracaoScan,
    tracer: Any,
    debug_logger: Any,
    obter_valor_cfg: Callable[[Any, str, Any], Any],
    extrair_coords: Callable[[Any], np.ndarray],
    construir_kdtree: Callable[[np.ndarray], Any],
    scan_legacy: Callable[..., dict[str, Any]],
    scan_geom_kdtree: Callable[..., dict[str, Any]],
    selecionar_por_scan: Callable[..., list[Pocket]],
) -> ResultadoScan:
    """Executa o scan com o mesmo RNG único e a mesma ordem de bolsões."""

    selected_pockets = list(candidate_pockets)
    scan_results: dict[str, dict[str, Any]] = {}
    scan_time_per_pocket: dict[str, float] = {}
    scan_executed = False
    scan_skipped = False
    scan_skip_reason: str | None = None
    n_pockets_after_scan = int(len(selected_pockets))
    scan_start = time.perf_counter()
    if config.enabled and selected_pockets:
        if config.skip_scan_when_single_pocket and len(selected_pockets) <= 1:
            scan_skipped = True
            scan_skip_reason = "single_candidate_fastpath"
            tracer.event(
                stage="scan",
                event_type="scan_skipped",
                payload={
                    "reason": scan_skip_reason,
                    "selector_mode": config.selector_mode,
                    "n_scan_candidates": int(config.n_scan_candidates),
                },
                level="BASIC",
                decision=True,
            )
            debug_logger.log(
                {
                    "type": "scan_skipped",
                    "reason": scan_skip_reason,
                    "selector_mode": config.selector_mode,
                    "n_scan_candidates": int(config.n_scan_candidates),
                }
            )
        else:
            scan_executed = True
            receptor_coords = extrair_coords(receptor)
            peptide_coords = extrair_coords(peptide)
            receptor_kdtree = construir_kdtree(receptor_coords)
            rng = np.random.default_rng(
                cfg.seed
                + int(
                    obter_valor_cfg(config.scan_cfg, "seed_offset", 0) or 0
                )
            )
            for pocket in selected_pockets:
                scan_single_start = time.perf_counter()
                if config.selector_mode == "geom_kdtree":
                    scan_results[str(pocket.id)] = scan_geom_kdtree(
                        receptor_kdtree,
                        receptor_coords,
                        peptide_coords,
                        pocket,
                        config.scan_cfg_effective,
                        rng,
                    )
                else:
                    scan_results[str(pocket.id)] = scan_legacy(
                        receptor_kdtree,
                        peptide_coords,
                        pocket,
                        config.scan_cfg_effective,
                        rng,
                    )
                scan_single_time = float(time.perf_counter() - scan_single_start)
                scan_time_per_pocket[str(pocket.id)] = scan_single_time
                scan_results[str(pocket.id)]["scan_time_sec"] = scan_single_time

            selected_pockets = selecionar_por_scan(
                selected_pockets,
                scan_results,
                obter_valor_cfg(config.scan_cfg, "select_top_k", None),
                selector_mode=config.selector_mode,
            )
            n_pockets_after_scan = int(len(selected_pockets))
            score_key = (
                "pocket_scan_score"
                if config.selector_mode == "geom_kdtree"
                else "scan_score"
            )
            clash_key = (
                "severe_clash_fraction"
                if config.selector_mode == "geom_kdtree"
                else "clash_ratio_best"
            )
            ranking_payload = [
                {
                    "pocket_id": str(p.id),
                    "rank": int(idx),
                    "score": float(
                        scan_results.get(str(p.id), {}).get(
                            score_key,
                            float("-inf"),
                        )
                    ),
                    "feasible_fraction": float(
                        scan_results.get(str(p.id), {}).get(
                            "feasible_fraction",
                            0.0,
                        )
                    ),
                    "clash_indicator": float(
                        scan_results.get(str(p.id), {}).get(clash_key, 1.0)
                    ),
                    "energy_q20": float(
                        scan_results.get(str(p.id), {}).get(
                            "energy_q20",
                            float("inf"),
                        )
                    ),
                }
                for idx, p in enumerate(selected_pockets)
            ]
            tracer.event(
                stage="scan",
                event_type="scan_selection_ranked",
                payload={
                    "selector_mode": config.selector_mode,
                    "ranking": ranking_payload,
                },
                level="TRACE",
                decision=True,
            )
            tracer.event(
                stage="scan",
                event_type="scan_selected",
                payload={
                    "selector_mode": config.selector_mode,
                    "selected_pockets": [str(p.id) for p in selected_pockets],
                },
                level="TRACE",
                decision=True,
            )
            debug_logger.log(
                {
                    "type": "scan_selected",
                    "selector_mode": config.selector_mode,
                    "selected": [str(p.id) for p in selected_pockets],
                }
            )
    scan_time = time.perf_counter() - scan_start
    return ResultadoScan(
        selected_pockets=selected_pockets,
        results=scan_results,
        scan_time=scan_time,
        scan_time_per_pocket=scan_time_per_pocket,
        n_pockets_after_scan=n_pockets_after_scan,
        executed=scan_executed,
        skipped=scan_skipped,
        skip_reason=scan_skip_reason,
    )

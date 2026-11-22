"""Normalização do modo, carregamento, ranking e corte top-k."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from dockingpp.domain.models import Pocket


@dataclass(frozen=True)
class ResultadoPreparacao:
    """Dados produzidos antes da avaliação geométrica dos bolsões."""

    search_space_mode: str
    modo_legado_pockets: bool
    candidate_pockets: list[Pocket]
    total_pockets: int
    pocketing_time: float
    n_pockets_input: int
    n_pockets_after_rank: int


def preparar_ranking(
    *,
    cfg: Any,
    receptor: Any,
    peptide: Any,
    dummy_pockets: list[Pocket],
    tracer: Any,
    debug_logger: Any,
    normalizar_modo_busca: Callable[[str | None, bool], str],
    carregar_pockets: Callable[..., list[Pocket]],
    ranquear_pockets: Callable[..., Any],
) -> ResultadoPreparacao:
    """Preserva a preparação e a sequência histórica de eventos."""

    modo_requisitado_bruto = getattr(cfg, "search_space_mode", None)
    modo_legado_pockets = str(modo_requisitado_bruto or "").strip().lower() == "pockets"
    search_space_mode = normalizar_modo_busca(
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
    if search_space_mode == "full" and not modo_legado_pockets:
        from dockingpp.pipeline import run as run_mod

        loaded_pockets = [run_mod._build_global_pocket(receptor, cfg)]
    else:
        loaded_pockets = dummy_pockets or carregar_pockets(
            receptor,
            cfg=cfg,
            pockets_path=getattr(cfg, "pockets_path", None),
        )
    tracer.event(
        stage="pocket_filter",
        event_type="pockets_loaded",
        payload={
            "loaded_pockets": [str(p.id) for p in loaded_pockets],
            "n_loaded": int(len(loaded_pockets)),
        },
        level="TRACE",
    )
    debug_logger.log(
        {
            "type": "pockets_loaded",
            "loaded": [str(p.id) for p in loaded_pockets],
            "n_loaded": int(len(loaded_pockets)),
        }
    )

    if search_space_mode == "full" and not modo_legado_pockets:
        ranked_pockets = list(loaded_pockets)
    else:
        ranked = ranquear_pockets(
            receptor,
            loaded_pockets,
            peptide=peptide,
            debug_logger=debug_logger,
        )
        ranked_pockets = [p for p, _ in ranked] if ranked else list(loaded_pockets)

    n_pockets_input = int(len(loaded_pockets))
    n_pockets_after_rank = int(len(ranked_pockets))
    total_pockets = len(ranked_pockets)
    candidate_pockets = list(ranked_pockets)
    if search_space_mode != "full":
        top_pockets = int(
            getattr(cfg, "top_pockets", len(candidate_pockets)) or len(candidate_pockets)
        )
        if top_pockets > 0:
            candidate_pockets = candidate_pockets[:top_pockets]

    pocketing_time = time.perf_counter() - pocketing_start
    tracer.event(
        stage="pocket_filter",
        event_type="pockets_ranked",
        payload={
            "ranked_pockets": [str(p.id) for p in ranked_pockets],
            "n_ranked": int(len(ranked_pockets)),
        },
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
        payload={
            "selected_pockets": [str(p.id) for p in candidate_pockets],
            "n_pockets_total": int(total_pockets),
        },
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

    return ResultadoPreparacao(
        search_space_mode=search_space_mode,
        modo_legado_pockets=modo_legado_pockets,
        candidate_pockets=candidate_pockets,
        total_pockets=total_pockets,
        pocketing_time=pocketing_time,
        n_pockets_input=n_pockets_input,
        n_pockets_after_rank=n_pockets_after_rank,
    )

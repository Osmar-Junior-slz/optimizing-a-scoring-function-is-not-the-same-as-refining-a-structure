"""Redução opcional guiada pelo peptídeo."""

from __future__ import annotations

from typing import Any, Callable

from dockingpp.domain.models import Pocket


def aplicar_reducao_condicionada(
    *,
    cfg: Any,
    peptide: Any,
    pockets: list[Pocket],
    search_space_mode: str,
    tracer: Any,
    debug_logger: Any,
    reduzir: Callable[..., list[Pocket]],
) -> list[Pocket]:
    """Executa o subpipeline opcional sem alterar sua posição no fluxo."""

    selected_pockets = pockets
    if search_space_mode == "reduced" and bool(
        getattr(cfg, "usar_reducao_condicionada_ao_peptideo", False)
    ):
        tracer.event(
            stage="pocket_filter",
            event_type="reducao_condicionada_ativada",
            payload={
                "mensagem": "Aplicando subpipeline opcional de reducao guiada por peptideo."
            },
            level="BASIC",
        )
        selected_pockets = reduzir(
            peptide=peptide,
            pockets=selected_pockets,
            cfg=cfg,
            tracer=tracer,
            debug_logger=debug_logger,
        )
    return selected_pockets

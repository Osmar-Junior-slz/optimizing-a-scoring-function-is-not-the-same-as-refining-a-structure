"""Composition root do DockingPP.

Este e o unico modulo novo que conhece simultaneamente o caso de uso central e
os adaptadores concretos do pipeline legado. As fachadas antigas fornecem seus
callables para preservar compatibilidade durante a migracao.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Callable

import numpy as np

from dockingpp.application.config import Config
from dockingpp.application.use_cases.run_docking import RunDocking, RunDockingDependencies
from dockingpp.adapters.outbound.structure_loader import FileStructureLoader
from dockingpp.pipeline.execucao.auditoria_execucao import AuditoriaExecucaoPipeline
from dockingpp.pipeline.execucao.busca_execucao.executor_busca import (
    ContextoExecucaoBusca,
    ExecutorBuscaPipeline,
)
from dockingpp.pipeline.execucao.selecao_bolsoes import SelecionadorBolsoesPipeline


def build_run_docking(
    *,
    dummy_inputs: Callable[[], tuple[Any, Any, list[Any]]],
    load_receptor: Callable[[str], Any],
    load_peptide: Callable[[str], Any],
    extract_coords: Callable[[Any], Any],
    normalize_search_space_mode: Callable[[str | None, bool], str],
    cfg_value: Callable[[Any, str, Any], Any],
    apply_conditional_reduction: Callable[..., list[Any]],
    execute_single_run: Callable[..., Any],
    allocate_split_budget: Callable[[int, int, int], list[tuple[int, int]]],
    build_global_pocket: Callable[[Any, Config], Any],
    config_class: type[Config] = Config,
) -> RunDocking:
    """Liga portas e adaptadores para construir o caso de uso executavel."""

    return RunDocking(
        RunDockingDependencies(
            new_run_id=lambda: f"run-{uuid.uuid4().hex[:12]}",
            seed_random=lambda seed: np.random.seed(seed),
            ensure_output_directory=lambda path: os.makedirs(path, exist_ok=True),
            audit_factory=AuditoriaExecucaoPipeline,
            pocket_selector_factory=SelecionadorBolsoesPipeline,
            search_executor_factory=ExecutorBuscaPipeline,
            search_context_factory=ContextoExecucaoBusca,
            dummy_inputs=dummy_inputs,
            structure_loader=FileStructureLoader(
                receptor_loader=load_receptor,
                peptide_loader=load_peptide,
            ),
            extract_coords=extract_coords,
            normalize_search_space_mode=normalize_search_space_mode,
            cfg_value=cfg_value,
            apply_conditional_reduction=apply_conditional_reduction,
            execute_single_run=execute_single_run,
            allocate_split_budget=allocate_split_budget,
            build_global_pocket=build_global_pocket,
            config_class=config_class,
        )
    )


__all__ = ["build_run_docking"]

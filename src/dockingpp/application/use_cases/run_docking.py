"""Caso de uso principal de execucao de docking.

O caso de uso conhece apenas contratos comportamentais recebidos na
composicao. Leitura de arquivos, UUID, filesystem, auditoria, selecao e busca
sao fornecidos por adaptadores externos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from dockingpp.application.config import Config
from dockingpp.application.ports.structures import StructureLoader
from dockingpp.domain.models import RunResult


@dataclass(frozen=True)
class RunDockingDependencies:
    """Colaboradores externos usados por :class:`RunDocking`."""

    new_run_id: Callable[[], str]
    seed_random: Callable[[int], None]
    ensure_output_directory: Callable[[str], None]
    audit_factory: Callable[[], Any]
    pocket_selector_factory: Callable[..., Any]
    search_executor_factory: Callable[..., Any]
    search_context_factory: Callable[..., Any]
    dummy_inputs: Callable[[], tuple[Any, Any, list[Any]]]
    structure_loader: StructureLoader
    extract_coords: Callable[[Any], Any]
    normalize_search_space_mode: Callable[[str | None, bool], str]
    cfg_value: Callable[[Any, str, Any], Any]
    apply_conditional_reduction: Callable[..., list[Any]]
    execute_single_run: Callable[..., Any]
    allocate_split_budget: Callable[[int, int, int], list[tuple[int, int]]]
    build_global_pocket: Callable[[Any, Config], Any]
    config_class: type[Config]


class RunDocking:
    """Coordena setup, selecao, busca e finalizacao de um run."""

    def __init__(self, dependencies: RunDockingDependencies) -> None:
        self._deps = dependencies

    def execute(
        self,
        cfg: Config,
        receptor_path: str,
        peptide_path: str,
        out_dir: str,
    ) -> RunResult:
        """Executa o caso de uso usando somente colaboradores injetados."""

        deps = self._deps
        run_id = deps.new_run_id()
        deps.seed_random(int(cfg.seed))
        deps.ensure_output_directory(out_dir)

        auditoria = deps.audit_factory()
        debug_logger = auditoria.criar_logger_debug(out_dir=out_dir, run_id=run_id, cfg=cfg)
        tracer = auditoria.criar_tracer(out_dir=out_dir, run_id=run_id, cfg=cfg)
        auditoria.iniciar_execucao(
            tracer=tracer,
            receptor_path=receptor_path,
            peptide_path=peptide_path,
            out_dir=out_dir,
            cfg=cfg,
        )

        try:
            receptor, peptide, dummy_pockets = self._load_inputs(
                receptor_path,
                peptide_path,
            )
            auditoria.registrar_inputs_carregados(
                tracer=tracer,
                receptor=receptor,
                peptide=peptide,
                extrair_coords=deps.extract_coords,
            )

            selecionador = deps.pocket_selector_factory(
                normalizar_modo_busca=deps.normalize_search_space_mode,
                obter_valor_cfg=deps.cfg_value,
                extrair_coords=deps.extract_coords,
                aplicar_reducao_condicionada=deps.apply_conditional_reduction,
            )
            contexto_selecao = selecionador.selecionar(
                cfg=cfg,
                receptor=receptor,
                peptide=peptide,
                dummy_pockets=dummy_pockets,
                tracer=tracer,
                debug_logger=debug_logger,
            )

            executor = deps.search_executor_factory(
                executar_execucao_unica=deps.execute_single_run,
                escrever_debug_summary=auditoria.escrever_debug_summary,
                alocar_orcamento=deps.allocate_split_budget,
                construir_bolsao_global=deps.build_global_pocket,
                classe_config=deps.config_class,
            )
            return executor.executar(
                cfg=cfg,
                receptor=receptor,
                peptide=peptide,
                receptor_path=receptor_path,
                peptide_path=peptide_path,
                out_dir=out_dir,
                run_id=run_id,
                tracer=tracer,
                contexto=deps.search_context_factory(
                    search_space_mode=contexto_selecao.search_space_mode,
                    modo_legado_pockets=contexto_selecao.modo_legado_pockets,
                    candidate_pockets=contexto_selecao.candidate_pockets,
                    selected_pockets=contexto_selecao.selected_pockets,
                    pockets=contexto_selecao.pockets,
                    total_pockets=contexto_selecao.total_pockets,
                    scan_results=contexto_selecao.scan_results,
                    scan_params=contexto_selecao.scan_params,
                    pocketing_time=contexto_selecao.pocketing_time,
                    scan_time=contexto_selecao.scan_time,
                    accepted_pockets=contexto_selecao.accepted_pockets,
                    feasible_pockets=contexto_selecao.feasible_pockets,
                    rejected=contexto_selecao.rejected,
                    selection_confidence_metadata=(
                        contexto_selecao.selection_confidence_metadata
                    ),
                    selection_confidence_policy=(
                        contexto_selecao.selection_confidence_policy
                    ),
                ),
            )
        except Exception as exc:
            auditoria.registrar_falha_execucao(tracer=tracer, erro=exc)
            raise
        finally:
            auditoria.finalizar_execucao(
                tracer=tracer,
                debug_logger=debug_logger,
                cfg=cfg,
                out_dir=out_dir,
            )

    def _load_inputs(
        self,
        receptor_path: str,
        peptide_path: str,
    ) -> tuple[Any, Any, list[Any]]:
        if receptor_path == "__dummy__" and peptide_path == "__dummy__":
            return self._deps.dummy_inputs()
        return (
            self._deps.structure_loader.load_receptor(receptor_path),
            self._deps.structure_loader.load_peptide(peptide_path),
            [],
        )

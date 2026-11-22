"""Pipeline entrypoints."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from dockingpp.application.config import Config
from dockingpp.application.use_cases.run_docking import RunDocking
from dockingpp.data.io import load_peptide, load_pockets, load_receptor
from dockingpp.data.pdb_identity import (
    build_input_id,
    normalize_pdb_id,
    resolve_complex_identity,
)
from dockingpp.data.pdb_pair_validation import (
    ReceptorPeptidePairValidation,
    validate_receptor_peptide_pair,
)
from dockingpp.data.pdb_validation import (
    PeptidePDBValidation,
    resolve_peptide_residue_limits,
    validate_peptide_pdb,
)
from dockingpp.domain.models import Pocket, RunResult
from dockingpp.pipeline.logging import AuditTracer
from dockingpp.pipeline.confianca.artifact_calibrator import load_confidence_calibrator
from dockingpp.priors.pocket import rank_pockets
from dockingpp.priors.pose import PriorNetPose
from dockingpp.reducao.admissibilidade import avaliar_admissibilidade_bolsao
from dockingpp.reducao.geometria_bolsao import descrever_geometria_bolsao
from dockingpp.reducao.perfil_peptideo import construir_perfil_peptideo
from dockingpp.reducao.pre_afinidade import estimar_pre_afinidade_bolsao
from dockingpp.reducao.ranking_bolsoes import ranquear_bolsoes_candidatos
from dockingpp.reducao.seletor_bolsoes import selecionar_bolsoes_para_busca
from dockingpp.scoring.cheap import score_pose_cheap
from dockingpp.scoring.expensive import score_pose_expensive
from dockingpp.search.algoritmos.hibrido import ABCGAVGOSSearch
from dockingpp.utils.debug_logger import DebugLogger


class PeptideInputValidationError(ValueError):
    """Raised before search when the supplied partner is not a valid peptide."""

    def __init__(self, validation: PeptidePDBValidation) -> None:
        self.validation = validation
        code = validation.error_code or "invalid_peptide_reference"
        super().__init__(
            "O arquivo informado como peptídeo não é válido para o protocolo "
            f"proteína–peptídeo ({code}). {validation.message}"
        )


class ReceptorPeptidePairValidationError(ValueError):
    """Raised before search when receptor and peptide are not separated."""

    def __init__(self, validation: ReceptorPeptidePairValidation) -> None:
        self.validation = validation
        code = validation.error_code or "invalid_receptor_peptide_pair"
        super().__init__(
            "ExecuÃ§Ã£o bloqueada: o receptor ainda contÃ©m o peptÃ­deo ou "
            f"Ã¡tomos sobrepostos ({code}). {validation.message} "
            "Use um receptor sem a cadeia peptÃ­dica antes de rodar o docking."
        )


def validate_peptide_input_or_raise(
    cfg: Config,
    peptide_path: str,
) -> PeptidePDBValidation | None:
    """Run the scientific input preflight before allocating a docking run."""

    if peptide_path == "__dummy__":
        return None
    if not bool(getattr(cfg, "peptide_input_validation_enabled", True)):
        return None
    min_residues, max_residues = resolve_peptide_residue_limits(cfg)
    validation = validate_peptide_pdb(
        peptide_path,
        min_residues=min_residues,
        max_residues=max_residues,
    )
    if not validation.is_valid:
        raise PeptideInputValidationError(validation)
    return validation


def validate_receptor_peptide_pair_or_raise(
    cfg: Config,
    receptor_path: str,
    peptide_path: str,
) -> ReceptorPeptidePairValidation | None:
    """Validate that receptor and peptide files are separated before docking."""

    if receptor_path == "__dummy__" or peptide_path == "__dummy__":
        return None
    if not bool(getattr(cfg, "receptor_peptide_pair_validation_enabled", True)):
        return None
    validation = validate_receptor_peptide_pair(receptor_path, peptide_path)
    if not validation.is_valid:
        raise ReceptorPeptidePairValidationError(validation)
    return validation


def validate_docking_inputs_or_raise(
    cfg: Config,
    receptor_path: str,
    peptide_path: str,
) -> tuple[PeptidePDBValidation | None, ReceptorPeptidePairValidation | None]:
    """Run all structural input preflights before allocating a docking run."""

    peptide_validation = validate_peptide_input_or_raise(cfg, peptide_path)
    pair_validation = validate_receptor_peptide_pair_or_raise(
        cfg,
        receptor_path,
        peptide_path,
    )
    return peptide_validation, pair_validation


def _dummy_inputs() -> tuple[Any, Any, list[Pocket]]:
    receptor_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 5.0, 0.0],
        ],
        dtype=float,
    )
    peptide_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ],
        dtype=float,
    )
    pockets = [
        Pocket(id="dummy-0", center=np.array([0.0, 0.0, 0.0]), radius=5.0, coords=receptor_coords),
        Pocket(id="dummy-1", center=np.array([10.0, 0.0, 0.0]), radius=5.0, coords=receptor_coords),
        Pocket(id="dummy-2", center=np.array([0.0, 10.0, 0.0]), radius=5.0, coords=receptor_coords),
    ]
    receptor = {"dummy": True, "coords": receptor_coords}
    peptide = {"dummy": True, "coords": peptide_coords}
    return receptor, peptide, pockets


def _extract_coords(receptor: Any) -> np.ndarray:
    if isinstance(receptor, dict) and "coords" in receptor:
        return np.asarray(receptor["coords"], dtype=float)
    if isinstance(receptor, np.ndarray):
        return np.asarray(receptor, dtype=float)
    coords = getattr(receptor, "coords", None)
    if coords is None:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def _build_global_pocket(receptor: Any, cfg: Config) -> Pocket:
    coords = _extract_coords(receptor)
    if coords.size:
        center = coords.mean(axis=0)
        deltas = coords - center.reshape(1, 3)
        max_dist = float(np.max(np.linalg.norm(deltas, axis=1)))
    else:
        center = np.zeros(3, dtype=float)
        max_dist = 0.0
    pocket_margin = float(getattr(cfg, "pocket_margin", 2.0))
    radius = max_dist + pocket_margin
    return Pocket(
        id="global",
        center=center,
        radius=radius,
        coords=coords,
        meta={
            "coords": coords,
            "local_coords": coords,
            "receptor_coords_full": coords,
            "detection_mode": "global",
        },
    )


def _cfg_value(cfg: Optional[Any], key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _aplicar_reducao_condicionada_ao_peptideo(
    peptide: Any,
    pockets: list[Pocket],
    cfg: Config,
    tracer: AuditTracer,
    debug_logger: DebugLogger,
) -> list[Pocket]:
    """Aplica subpipeline opcional de redução guiada por peptídeo."""

    if not pockets:
        return pockets

    cfg_reducao = _cfg_value(cfg, "reducao", {}) or {}
    if not isinstance(cfg_reducao, dict):
        cfg_reducao = {}

    top_k_reducao = cfg_reducao.get("top_k")
    if top_k_reducao is None:
        top_k_reducao = len(pockets)

    score_minimo = cfg_reducao.get("score_minimo")
    apenas_admissiveis = bool(cfg_reducao.get("apenas_admissiveis", True))
    permitir_fallback = bool(cfg_reducao.get("permitir_fallback", True))
    quantidade_fallback = int(cfg_reducao.get("quantidade_fallback", 1) or 1)

    perfil_peptideo = construir_perfil_peptideo(peptide)
    debug_logger.log(
        {
            "type": "reducao_perfil_peptideo",
            "mensagem": "Perfil geométrico do peptídeo para redução condicionada.",
            "comprimento_efetivo": float(perfil_peptideo.comprimento_efetivo),
            "largura_efetiva": float(perfil_peptideo.largura_efetiva),
            "espessura_efetiva": float(perfil_peptideo.espessura_efetiva),
            "extensao_maxima": float(perfil_peptideo.extensao_maxima),
            "raio_giro": float(perfil_peptideo.raio_giro),
            "indice_flexibilidade": float(perfil_peptideo.indice_flexibilidade),
        }
    )

    geometrias_por_id = {str(pocket.id): descrever_geometria_bolsao(pocket) for pocket in pockets}
    resultados_admissibilidade = [
        avaliar_admissibilidade_bolsao(perfil_peptideo, geometrias_por_id[str(pocket.id)])
        for pocket in pockets
    ]
    for resultado in resultados_admissibilidade:
        geometria = geometrias_por_id.get(resultado.id_bolsao)
        debug_logger.log(
            {
                "type": "reducao_bolsao_avaliado",
                "mensagem": "Avaliação geométrica de admissibilidade do bolsão.",
                "id_bolsao": resultado.id_bolsao,
                "admissivel": bool(resultado.admissivel),
                "score_encaixe_geometrico": float(resultado.score_encaixe_geometrico),
                "motivos_reprovacao": list(resultado.motivos_reprovacao),
                "geometria": {
                    "comprimento_util": float(geometria.comprimento_util) if geometria is not None else None,
                    "largura_util": float(geometria.largura_util) if geometria is not None else None,
                    "profundidade_util": float(geometria.profundidade_util) if geometria is not None else None,
                    "continuidade_superficial": float(geometria.continuidade_superficial) if geometria is not None else None,
                    "exposicao_superficial": float(geometria.exposicao_superficial) if geometria is not None else None,
                    "volume_estimado": float(geometria.volume_estimado) if geometria is not None else None,
                },
            }
        )
        tracer.event(
            stage="pocket_filter",
            event_type="reducao_bolsao_avaliado",
            payload={
                "id_bolsao": resultado.id_bolsao,
                "admissivel": bool(resultado.admissivel),
                "score_encaixe_geometrico": float(resultado.score_encaixe_geometrico),
                "motivos_reprovacao": list(resultado.motivos_reprovacao),
            },
            pocket_id=str(resultado.id_bolsao),
            level="TRACE",
            decision=True,
        )

    resultados_pre_afinidade = [
        estimar_pre_afinidade_bolsao(
            perfil_peptideo,
            geometrias_por_id[resultado.id_bolsao],
            score_encaixe_geometrico=resultado.score_encaixe_geometrico,
        )
        for resultado in resultados_admissibilidade
        if resultado.admissivel
    ]
    for resultado_pre in resultados_pre_afinidade:
        debug_logger.log(
            {
                "type": "reducao_pre_afinidade",
                "mensagem": "Pré-afinidade barata calculada para bolsão admissível.",
                "id_bolsao": resultado_pre.id_bolsao,
                "score_pre_afinidade": float(resultado_pre.score_pre_afinidade),
                "score_contatos": float(resultado_pre.score_contatos),
                "penalidade_clash": float(resultado_pre.penalidade_clash),
                "score_ancoragem": float(resultado_pre.score_ancoragem),
            }
        )

    ranking_reducao = ranquear_bolsoes_candidatos(resultados_admissibilidade, resultados_pre_afinidade)
    for posicao, entrada in enumerate(ranking_reducao, start=1):
        debug_logger.log(
            {
                "type": "reducao_ranking",
                "mensagem": "Posição do bolsão no ranking da redução.",
                "id_bolsao": entrada.id_bolsao,
                "posicao": int(posicao),
                "admissivel": bool(entrada.admissivel),
                "score_encaixe_geometrico": float(entrada.score_encaixe_geometrico),
                "score_pre_afinidade": float(entrada.score_pre_afinidade),
                "score_final": float(entrada.score_final),
            }
        )

    selecionados_ranking = selecionar_bolsoes_para_busca(
        ranking_reducao,
        top_k=top_k_reducao,
        score_minimo=score_minimo,
        apenas_admissiveis=apenas_admissiveis,
        permitir_fallback=permitir_fallback,
        quantidade_fallback=quantidade_fallback,
    )

    mapa_pockets = {str(pocket.id): pocket for pocket in pockets}
    pockets_selecionados = [
        mapa_pockets[item.id_bolsao]
        for item in selecionados_ranking
        if item.id_bolsao in mapa_pockets
    ]

    elegiveis_sem_fallback = [
        item
        for item in ranking_reducao
        if (not apenas_admissiveis or bool(item.admissivel))
        and (score_minimo is None or float(item.score_final) >= float(score_minimo))
    ]
    fallback_acionado = bool(
        not elegiveis_sem_fallback
        and permitir_fallback
        and ranking_reducao
        and selecionados_ranking
    )

    tracer.event(
        stage="pocket_filter",
        event_type="reducao_condicionada_aplicada",
        payload={
            "n_pockets_entrada": int(len(pockets)),
            "n_pockets_saida": int(len(pockets_selecionados)),
            "top_k": int(top_k_reducao) if top_k_reducao is not None else None,
            "score_minimo": float(score_minimo) if score_minimo is not None else None,
            "apenas_admissiveis": bool(apenas_admissiveis),
            "permitir_fallback": bool(permitir_fallback),
            "quantidade_fallback": int(quantidade_fallback),
            "fallback_acionado": bool(fallback_acionado),
            "selecionados": [item.id_bolsao for item in selecionados_ranking],
            "n_pre_afinidade": int(len(resultados_pre_afinidade)),
        },
        level="BASIC",
        decision=True,
    )
    debug_logger.log(
        {
            "type": "reducao_condicionada",
            "mensagem": "Subpipeline de redução guiada por peptídeo ativado.",
            "n_pockets_entrada": int(len(pockets)),
            "n_pockets_saida": int(len(pockets_selecionados)),
            "regras_selecao": {
                "top_k": int(top_k_reducao) if top_k_reducao is not None else None,
                "score_minimo": float(score_minimo) if score_minimo is not None else None,
                "apenas_admissiveis": bool(apenas_admissiveis),
                "permitir_fallback": bool(permitir_fallback),
                "quantidade_fallback": int(quantidade_fallback),
            },
            "selecionados": [item.id_bolsao for item in selecionados_ranking],
            "fallback_acionado": bool(fallback_acionado),
            "n_pre_afinidade": int(len(resultados_pre_afinidade)),
        }
    )

    return pockets_selecionados


def _normalize_search_space_mode(search_space_mode: str | None, full_search: bool) -> str:
    raw = (search_space_mode or "").strip().lower()
    if raw in {"full", "reduced"}:
        return raw
    if raw == "global":
        return "full"
    if raw == "pockets":
        # Compatibilidade: alias legado "pockets" seguia `full_search`.
        return "full" if full_search else "reduced"
    return "full" if full_search else "reduced"


def _allocate_split_budget(total_generations: int, total_pop_size: int, n_pockets: int) -> list[tuple[int, int]]:
    if n_pockets <= 0:
        return []
    # The active population loop evaluates the initial population plus
    # ``generations`` evolutionary populations. Keep that exact convention in
    # the budget distributed across pockets.
    total_eval = max(1, (max(1, int(total_generations)) + 1) * int(total_pop_size))
    base = total_eval // n_pockets
    rem = total_eval % n_pockets
    budgets: list[tuple[int, int]] = []
    for idx in range(n_pockets):
        eval_budget = base + (1 if idx < rem else 0)
        pop_size = max(1, min(int(total_pop_size), int(eval_budget)))
        generations = max(1, (int(eval_budget) + pop_size - 1) // pop_size)
        budgets.append((generations, pop_size))
    return budgets


def _execute_single_run(
    cfg: Config,
    receptor: Any,
    peptide: Any,
    pockets: list[Pocket],
    out_dir: str,
    run_id: str,
    receptor_path: str,
    peptide_path: str,
    search_space_mode: str,
    total_pockets: int,
    selected_pockets: int,
    pocketing_time: float,
    scan_time: float,
    scan_params: dict[str, Any],
    scan_results: dict[str, dict[str, Any]],
    selected_pocket_ids: list[str],
    confidence_policy_metadata: dict[str, Any] | None = None,
    tracer: AuditTracer | None = None,
    pocket_id: str | None = None,
    include_global_overhead: bool = True,
):
    """Wrapper compatível para execução única delegada ao módulo de execução.

    Mantém assinatura histórica para evitar quebra externa enquanto a lógica
    concreta permanece encapsulada em `pipeline.execucao.execucao_busca`.
    """

    from dockingpp.pipeline.execucao import execucao_busca as execucao_busca_facade

    # Injecao explicita preserva o ponto historico de monkeypatch da fachada
    # sem criar uma dependencia reversa da implementacao para esse modulo.
    executor = execucao_busca_facade.ExecutorExecucaoUnicaPipeline(
        search_engine_resolver=execucao_busca_facade.resolve_search_engine,
    )
    return executor.executar(
        cfg=cfg,
        receptor=receptor,
        peptide=peptide,
        pockets=pockets,
        out_dir=out_dir,
        contexto=execucao_busca_facade.ContextoExecucaoUnica(
            run_id=run_id,
            receptor_path=receptor_path,
            peptide_path=peptide_path,
            search_space_mode=search_space_mode,
            total_pockets=total_pockets,
            selected_pockets=selected_pockets,
            pocketing_time=pocketing_time,
            scan_time=scan_time,
            scan_params=scan_params,
            scan_results=scan_results,
            selected_pocket_ids=selected_pocket_ids,
            confidence_policy_metadata=confidence_policy_metadata or {},
            tracer=tracer,
            pocket_id=pocket_id,
            include_global_overhead=include_global_overhead,
        ),
    )


def _build_run_docking_use_case() -> RunDocking:
    """Monta o caso de uso com os adaptadores concretos da aplicacao.

    A composicao le os simbolos deste modulo no momento da chamada. Isso
    preserva os pontos historicos de monkeypatch enquanto impede que o caso de
    uso central dependa da fachada ``pipeline.run``.
    """

    from dockingpp.composition import build_run_docking

    return build_run_docking(
        dummy_inputs=_dummy_inputs,
        load_receptor=load_receptor,
        load_peptide=load_peptide,
        extract_coords=_extract_coords,
        normalize_search_space_mode=_normalize_search_space_mode,
        cfg_value=_cfg_value,
        apply_conditional_reduction=_aplicar_reducao_condicionada_ao_peptideo,
        execute_single_run=_execute_single_run,
        allocate_split_budget=_allocate_split_budget,
        build_global_pocket=_build_global_pocket,
        config_class=Config,
    )


def _executar_pipeline_fluxo(cfg: Config, receptor_path: str, peptide_path: str, out_dir: str) -> RunResult:
    """Compatibilidade: delega o fluxo principal ao caso de uso central."""

    validate_docking_inputs_or_raise(cfg, receptor_path, peptide_path)
    return _build_run_docking_use_case().execute(
        cfg=cfg,
        receptor_path=receptor_path,
        peptide_path=peptide_path,
        out_dir=out_dir,
    )



def run_pipeline(cfg: Config, receptor_path: str, peptide_path: str, out_dir: str) -> RunResult:
    """Fachada pública de execução do pipeline.

    A orquestracao principal pertence agora a camada de aplicacao. Este modulo
    conserva os imports e pontos de injecao historicos.
    """
    # Mantém ponto de injeção do logger de debug para testes e execução local.
    cfg.debug_logger_cls = getattr(cfg, "debug_logger_cls", DebugLogger)

    if not getattr(cfg, "input_id", None):
        cfg.input_id = build_input_id(receptor_path, peptide_path)

    # A scientific id explicitly supplied by the GUI/protocol is authoritative,
    # but it must not contradict a PDB HEADER present in the persisted inputs.
    # This prevents a run stored under e.g. ``1AVF`` from silently using a
    # receptor whose HEADER says ``1A4W``.
    declared_complex_id = normalize_pdb_id(getattr(cfg, "complex_id", None))
    inferred_identity = resolve_complex_identity(receptor_path, peptide_path)
    if declared_complex_id is not None:
        inferred_complex_id = normalize_pdb_id(inferred_identity.get("complex_id"))
        inferred_source = str(inferred_identity.get("complex_id_source") or "")
        if inferred_source == "unresolved_header_conflict":
            raise ValueError(
                "Conflito de identidade nos arquivos PDB: "
                f"{inferred_identity.get('complex_id_conflict')}"
            )
        if inferred_complex_id is not None and inferred_complex_id != declared_complex_id:
            raise ValueError(
                "O PDB informado na interface nao corresponde ao HEADER dos arquivos: "
                f"declarado={declared_complex_id}, header={inferred_complex_id}. "
                "Revise receptor/peptideo antes de executar o experimento."
            )
        cfg.complex_id = declared_complex_id
        if not getattr(cfg, "complex_id_source", None):
            cfg.complex_id_source = "explicit_config"
        cfg.complex_id_conflict = None
    elif not getattr(cfg, "complex_id", None):
        cfg.complex_id = inferred_identity["complex_id"]
        cfg.complex_id_source = inferred_identity["complex_id_source"]
        cfg.complex_id_conflict = inferred_identity["complex_id_conflict"]
    elif not getattr(cfg, "complex_id_source", None):
        cfg.complex_id_source = "explicit_config"
    calibrator_path = getattr(cfg, "confidence_calibrator_path", None)
    if bool(getattr(cfg, "confidence_calibrator_required", False)) and not calibrator_path:
        raise ValueError(
            "confidence_calibrator_required=True exige confidence_calibrator_path"
        )
    if getattr(cfg, "confidence_calibrator", None) is None and calibrator_path:
        try:
            cfg.confidence_calibrator = load_confidence_calibrator(calibrator_path)
            cfg.confidence_calibrator_load_error = None
        except Exception as exc:  # noqa: BLE001 - strictness is configurable
            cfg.confidence_calibrator_load_error = f"{type(exc).__name__}: {exc}"
            if bool(getattr(cfg, "confidence_calibrator_required", False)):
                raise ValueError(
                    "Nao foi possivel carregar o calibrador de confianca: "
                    f"{cfg.confidence_calibrator_load_error}"
                ) from exc

    validate_docking_inputs_or_raise(cfg, receptor_path, peptide_path)
    return _build_run_docking_use_case().execute(
        cfg=cfg,
        receptor_path=receptor_path,
        peptide_path=peptide_path,
        out_dir=out_dir,
    )

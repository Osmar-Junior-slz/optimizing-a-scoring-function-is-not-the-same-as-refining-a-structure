"""Página de execução de docking com progresso e resultados."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import streamlit as st
import yaml

from dockingpp.data.io import load_config
from dockingpp.data.pdb_pair_validation import validate_receptor_peptide_pair
from dockingpp.data.pdb_validation import (
    PeptidePDBValidation,
    resolve_peptide_residue_limits,
    validate_peptide_pdb,
    validate_peptide_pdb_text,
)
from dockingpp.api import Config, run_pipeline
from dockingpp.data.pdb_identity import normalize_pdb_id
from dockingpp.gui.pages.base import BasePage
from dockingpp.gui.services.experiment_service import create_experiment_layout, update_experiment_index
from dockingpp.gui.services.io_service import ensure_dir, save_uploaded_file
from dockingpp.gui.services.progress_service import compute_progress, format_progress_text, read_last_metrics_generation
from dockingpp.gui.services.report_service import load_jsonl, metrics_series, summarize_metrics
from dockingpp.gui.state import AppState, DEFAULT_CONFIG_PATH, StateKeys, set_state
from dockingpp.gui.ui.components import download_json_button
from dockingpp.gui.utils.confidence_policy_labels import (
    build_confidence_policy_override,
    confidence_policy_mode_labels,
    confidence_policy_mode_to_label,
    resolve_confidence_policy_state,
)


PROTOCOL_ERROR_PENDING_KEY = "protocol_error_pending"


def validate_selected_peptide(
    *,
    cfg: Mapping[str, Any],
    prepared_path: Path | None,
    uploaded_file: Any,
) -> PeptidePDBValidation:
    """Validate the exact peptide source selected in the docking form."""

    min_residues, max_residues = resolve_peptide_residue_limits(cfg)
    limits = {
        "min_residues": min_residues,
        "max_residues": max_residues,
    }
    if prepared_path is not None:
        return validate_peptide_pdb(prepared_path, **limits)
    if uploaded_file is None:
        return validate_peptide_pdb_text("", source="<upload ausente>", **limits)
    return validate_peptide_pdb_text(
        uploaded_file.getvalue().decode("utf-8", errors="ignore"),
        source=str(getattr(uploaded_file, "name", "<upload>")),
        **limits,
    )


def persist_selected_input(
    *,
    prepared_path: Path | None,
    uploaded_file: Any,
    inputs_dir: Path,
    filename: str,
) -> Path:
    """Copy a prepared input or persist its upload into the experiment."""

    if prepared_path is not None:
        destination = inputs_dir / filename
        shutil.copy2(prepared_path, destination)
        return destination
    return save_uploaded_file(uploaded_file, inputs_dir, filename_hint=filename)


def persist_experiment_inputs(
    *,
    prepared_receptor_path: Path | None,
    receptor_upload: Any,
    prepared_peptide_path: Path | None,
    peptide_upload: Any,
    inputs_dir: Path,
) -> tuple[Path, Path]:
    """Persist both partners under stable, auditable experiment filenames."""

    receptor = persist_selected_input(
        prepared_path=prepared_receptor_path,
        uploaded_file=receptor_upload,
        inputs_dir=inputs_dir,
        filename="receptor.pdb",
    )
    peptide = persist_selected_input(
        prepared_path=prepared_peptide_path,
        uploaded_file=peptide_upload,
        inputs_dir=inputs_dir,
        filename="peptide.pdb",
    )
    return receptor, peptide


def render_prepared_input_controls() -> tuple[str, bool, str, bool]:
    """Render optional prepared receptor/peptide selectors."""

    receptor_path = st.session_state.get(StateKeys.PREPARED_RECEPTOR_PATH, "")
    use_receptor = False
    if receptor_path:
        st.info(f"Receptor preparado disponível: {receptor_path}")
        use_receptor = st.checkbox(
            "Usar receptor preparado", value=True, key="use_prepared_receptor"
        )
    peptide_path = st.session_state.get(StateKeys.PREPARED_PEPTIDE_PATH, "")
    use_peptide = False
    if peptide_path:
        st.info(f"Peptídeo preparado disponível: {peptide_path}")
        use_peptide = st.checkbox(
            "Usar peptídeo preparado", value=True, key="use_prepared_peptide"
        )
    return receptor_path, use_receptor, peptide_path, use_peptide


def peptide_preflight_allows_execution(
    *,
    cfg: Mapping[str, Any],
    prepared_path: Path | None,
    uploaded_file: Any,
) -> bool:
    """Render a clear preflight failure and stop invalid experiments."""

    if not bool(cfg.get("peptide_input_validation_enabled", True)):
        return True
    validation = validate_selected_peptide(
        cfg=cfg,
        prepared_path=prepared_path,
        uploaded_file=uploaded_file,
    )
    if validation.is_valid:
        return True
    st.error("Execução bloqueada: o arquivo informado não é um peptídeo padrão válido.")
    st.write(validation.message)
    st.json(validation.as_dict())
    return False


def receptor_peptide_pair_preflight_allows_execution(
    *,
    cfg: Mapping[str, Any],
    receptor_path: Path,
    peptide_path: Path,
) -> bool:
    """Block docking when receptor still contains the selected peptide."""

    if not bool(cfg.get("receptor_peptide_pair_validation_enabled", True)):
        return True
    validation = validate_receptor_peptide_pair(receptor_path, peptide_path)
    if validation.is_valid:
        return True
    st.error(
        "ExecuÃ§Ã£o bloqueada: o receptor ainda contÃ©m a cadeia do peptÃ­deo "
        "ou Ã¡tomos sobrepostos."
    )
    st.write(validation.message)
    st.json(validation.as_dict())
    return False


def write_resolved_config(out_dir: Path, cfg_dict: dict[str, Any]) -> None:
    """Salva a configuração resolvida no diretório de saída."""

    # PT-BR: garantimos o diretório antes de escrever a configuração final.
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.resolved.json"
    path.write_text(json.dumps(cfg_dict, indent=2, ensure_ascii=False), encoding="utf-8")


def _merge_mapping(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Mescla configurações aninhadas preservando blocos calibrados."""

    resolved = dict(base)
    for key, value in updates.items():
        if value is None:
            continue
        current = resolved.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            resolved[key] = _merge_mapping(dict(current), dict(value))
        else:
            resolved[key] = value
    return resolved


def apply_overrides(raw_cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Aplica sobreposições do usuário em uma cópia da configuração base."""

    # PT-BR: mesclamos dicionários para não apagar blocos internos como
    # confidence_policy.search_control quando a tela altera só enabled/mode.
    return _merge_mapping(raw_cfg, overrides)


def _merge_run_confidence_policy(
    resolved_cfg: dict[str, Any],
    run_policy: dict[str, Any],
) -> dict[str, Any]:
    """Preserva search_control calibrado ao alternar braços com/sem heurística."""

    base_policy = resolved_cfg.get("confidence_policy")
    if not isinstance(base_policy, Mapping):
        base_policy = {}
    return _merge_mapping(dict(base_policy), dict(run_policy or {}))


def normalize_out_dir(path: str) -> str:
    """Normaliza o caminho de saída garantindo resolução consistente."""

    if not path:
        return ""
    return str(Path(path).expanduser().resolve(strict=False))


def can_browse_for_directory() -> bool:
    """Informa se a biblioteca Tkinter está disponível para seleção de pasta."""

    try:
        import tkinter  # noqa: F401
        from tkinter import filedialog  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def browse_for_directory() -> str | None:
    """Abre um seletor de diretório nativo quando o Tkinter está disponível."""

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:  # noqa: BLE001
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        folder = filedialog.askdirectory()
        root.destroy()
    except Exception:  # noqa: BLE001
        return None
    return folder or None


def update_recent_out_dirs(selected: str, max_items: int = 8) -> None:
    """Atualiza a lista de diretórios de saída recentes."""

    if not selected:
        return
    normalized = normalize_out_dir(selected)
    recent = st.session_state.get(StateKeys.RECENT_OUT_DIRS, [])
    if not isinstance(recent, list):
        recent = []
    # PT-BR: movemos o diretório escolhido para o topo da lista.
    new_recent = [normalized] + [path for path in recent if path != normalized]
    st.session_state[StateKeys.RECENT_OUT_DIRS] = new_recent[:max_items]


def validate_out_dir(out_dir: str) -> tuple[bool, Path | None, str | None]:
    """Valida e cria o diretório de saída, retornando mensagem de erro se necessário."""

    if not out_dir or not out_dir.strip():
        return False, None, "Informe um diretório de saída."
    normalized = normalize_out_dir(out_dir)
    out_path = Path(normalized)
    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        return False, None, "Não foi possível criar o diretório de saída."
    try:
        with tempfile.NamedTemporaryFile(dir=out_path, prefix=".write_test_", delete=True) as handle:
            handle.write(b"ok")
            handle.flush()
    except Exception:  # noqa: BLE001
        return False, None, "Diretório de saída não é gravável."
    return True, out_path, None


def validate_protein_name(protein_name: str) -> tuple[bool, str | None]:
    """Valida o identificador de proteína/complexo usado para criar a pasta histórica."""

    if not protein_name or not protein_name.strip():
        return False, "Informe o nome da proteína/complexo/PDB."
    return True, None


def apply_experiment_complex_identity(
    resolved_cfg: dict[str, Any], protein_name: str
) -> dict[str, Any]:
    """Persist the GUI PDB id as audited scientific identity when canonical."""

    updated = dict(resolved_cfg)
    declared_complex_id = normalize_pdb_id(protein_name)
    if declared_complex_id is not None:
        updated["complex_id"] = declared_complex_id
        updated["complex_id_source"] = "gui_experiment_name"
        updated["complex_id_conflict"] = None
    return updated


def _search_space_mode_from_label(search_space_label: str) -> str:
    """Mapeia rotulo da UI para modo canonico de busca."""

    return "reduced" if search_space_label == "Bols\u00f5es" else "full"


def _build_compare_runs(out_path: Path, top_pockets: int) -> list[dict[str, Any]]:
    """Monta os dois runs do modo comparar usando modos canonicos."""

    return [
        {
            "label": "full",
            "out_dir": out_path / "full",
            "full_search": True,
            "top_pockets": None,
            "search_space_mode": "full",
        },
        {
            "label": "reduced",
            "out_dir": out_path / "reduced",
            "full_search": False,
            "top_pockets": int(top_pockets),
            "search_space_mode": "reduced",
        },
    ]


def _build_algorithm_runs(out_path: Path, search_space_mode: str) -> list[dict[str, Any]]:
    """Monta execuções sequenciais para comparar os três algoritmos."""

    return [
        {
            "label": "ABC+AG+VGOS",
            "engine": "hybrid",
            "out_dir": out_path / search_space_mode / "abc_ag_vgos",
        },
        {
            "label": "AG",
            "engine": "ag",
            "out_dir": out_path / search_space_mode / "ag",
        },
        {
            "label": "PSO",
            "engine": "pso",
            "out_dir": out_path / search_space_mode / "pso",
        },
    ]


def _build_full_protocol_runs(
    out_path: Path,
    *,
    top_pockets: int,
    confidence_policy_mode: str,
    round_index: int = 1,
    protocol_kind: str = "heuristic_reduced",
) -> list[dict[str, Any]]:
    """Monta o protocolo escolhido: heurística, full vs reduced ou completo."""

    round_name = f"rodada_{int(round_index):03d}"
    protocol_kind = str(protocol_kind or "heuristic_reduced")

    all_scenarios = [
        {
            "label": "Modo completo (sem heurística)",
            "mode_label": "Completo",
            "heuristic_label": "Sem heurística",
            "search_space_mode": "full",
            "full_search": True,
            "top_pockets": None,
            "confidence_policy": build_confidence_policy_override(
                enabled=False,
                selected_mode_or_label="disabled",
            ),
            "dir": out_path / "full" / round_name / "sem_heuristica",
        },
        {
            "label": "Modo completo (com heurística)",
            "mode_label": "Completo",
            "heuristic_label": "Com heurística",
            "search_space_mode": "full",
            "full_search": True,
            "top_pockets": None,
            "confidence_policy": build_confidence_policy_override(
                enabled=True,
                selected_mode_or_label=confidence_policy_mode,
            ),
            "dir": out_path / "full" / round_name / "com_heuristica",
        },
        {
            "label": "Modo reduzido (sem heurística)",
            "mode_label": "Reduzido",
            "heuristic_label": "Sem heurística",
            "search_space_mode": "reduced",
            "full_search": False,
            "top_pockets": int(top_pockets),
            "confidence_policy": build_confidence_policy_override(
                enabled=False,
                selected_mode_or_label="disabled",
            ),
            "dir": out_path / "reduced" / round_name / "sem_heuristica",
        },
        {
            "label": "Modo reduzido (com heurística)",
            "mode_label": "Reduzido",
            "heuristic_label": "Com heurística",
            "search_space_mode": "reduced",
            "full_search": False,
            "top_pockets": int(top_pockets),
            "confidence_policy": build_confidence_policy_override(
                enabled=True,
                selected_mode_or_label=confidence_policy_mode,
            ),
            "dir": out_path / "reduced" / round_name / "com_heuristica",
        },
    ]
    if protocol_kind == "heuristic_full":
        scenarios = [item for item in all_scenarios if item["search_space_mode"] == "full"]
    elif protocol_kind == "heuristic_reduced":
        scenarios = [item for item in all_scenarios if item["search_space_mode"] == "reduced"]
    elif protocol_kind == "full_vs_reduced":
        scenarios = [
            item
            for item in all_scenarios
            if item["search_space_mode"] in {"full", "reduced"} and item["heuristic_label"] == "Com heurística"
        ]
    else:
        scenarios = all_scenarios
    algorithms = [
        {"label": "ABC+AG+VGOS", "engine": "hybrid", "dir": "abc_ag_vgos"},
        {"label": "AG", "engine": "ag", "dir": "ag"},
        {"label": "PSO", "engine": "pso", "dir": "pso"},
    ]
    runs: list[dict[str, Any]] = []
    for scenario in scenarios:
        for algorithm in algorithms:
            runs.append(
                {
                    **scenario,
                    "algorithm_label": algorithm["label"],
                    "engine": algorithm["engine"],
                    "out_dir": scenario["dir"] / algorithm["dir"],
                }
            )
    return runs


def _algorithm_seed_offset(engine: str) -> int:
    offsets = {"hybrid": 1, "ag": 2, "pso": 3}
    return offsets.get(str(engine), 9)


def _protocol_label_to_kind(label: str) -> str:
    mapping = {
        "Heurística no modo reduzido": "heuristic_reduced",
        "Heurística no modo completo": "heuristic_full",
        "Heuristica no modo reduzido": "heuristic_reduced",
        "Heuristica no modo completo": "heuristic_full",
        "Completo vs reduzido": "full_vs_reduced",
        "Protocolo completo": "complete",
    }
    return mapping.get(str(label), "heuristic_reduced")


def _protocol_kind_to_question(kind: str) -> str:
    questions = {
        "heuristic_reduced": "A heurística reduz tempo e/ou melhora energia no modo reduzido?",
        "heuristic_full": "A heurística reduz tempo e/ou melhora energia no modo completo?",
        "full_vs_reduced": "O modo reduzido é competitivo contra o modo completo usando a heurística?",
        "complete": "Análise combinada: heurística no full, heurística no reduzido e full vs reduzido.",
    }
    return questions.get(str(kind), questions["heuristic_reduced"])


def _build_protocol_job_list(
    results_dir: Path,
    *,
    top_pockets: int,
    confidence_policy_mode: str,
    n_rounds: int,
    protocol_kind: str,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for round_index in range(1, int(n_rounds) + 1):
        round_runs = _build_full_protocol_runs(
            results_dir,
            top_pockets=top_pockets,
            confidence_policy_mode=str(confidence_policy_mode),
            round_index=round_index,
            protocol_kind=protocol_kind,
        )
        for run in round_runs:
            jobs.append({"round_index": int(round_index), "run": run})
    return jobs


def _write_experiment_status(experiment_dir: Path, payload: dict[str, Any]) -> None:
    """Salva status incremental para o usuario saber onde parou."""

    status_path = experiment_dir / "experiment_status.json"
    status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _first_present(payload: dict[str, Any], keys: list[str]) -> Any:
    """Retorna o primeiro valor não nulo para um conjunto de chaves."""

    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _derive_reduction_ratio(explicit_ratio: Any, pockets_total: Any, pockets_used: Any) -> float | None:
    """Calcula reduction_ratio como 1 - used/total quando necessário."""

    if explicit_ratio is not None:
        try:
            return float(explicit_ratio)
        except (TypeError, ValueError):
            return None

    try:
        total_val = float(pockets_total) if pockets_total is not None else None
        used_val = float(pockets_used) if pockets_used is not None else None
    except (TypeError, ValueError):
        return None

    if total_val is None or used_val is None or total_val <= 0.0:
        return None
    return min(1.0, max(0.0, 1.0 - (used_val / total_val)))


def _build_compare_quick_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Monta tabela rápida de comparação usando summary como fonte primária."""

    rows: list[dict[str, Any]] = []
    for label in ("full", "reduced"):
        payload = results.get(label, {})
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        timing = payload.get("timing") if isinstance(payload.get("timing"), dict) else {}

        if label == "reduced":
            cheap_keys = ["best_over_pockets_cheap", "triage_score", "best_score_cheap", "best_score", "best"]
            eval_keys = ["total_n_eval", "n_eval_total", "n_eval", "evals", "evaluations"]
        else:
            cheap_keys = ["triage_score", "best_score_cheap", "best_score", "best"]
            eval_keys = ["n_eval_total", "total_n_eval", "n_eval", "evals", "evaluations"]

        best_cheap = _first_present(summary, cheap_keys)
        if best_cheap is None:
            best_cheap = payload.get("best_score_cheap")

        n_eval_total = _first_present(summary, eval_keys)
        if n_eval_total is None:
            n_eval_total = _first_present(metrics, eval_keys)

        pockets_total = _first_present(summary, ["n_pockets_total", "pockets_total", "n_pockets_detected"])
        if pockets_total is None:
            pockets_total = _first_present(metrics, ["n_pockets_total", "pockets_total", "n_pockets_detected"])

        pockets_used = _first_present(summary, ["n_pockets_used", "pockets_used"])
        if pockets_used is None:
            pockets_used = _first_present(metrics, ["n_pockets_used", "pockets_used"])

        explicit_ratio = _first_present(summary, ["reduction_ratio", "ratio"])
        if explicit_ratio is None and (pockets_total is None or pockets_used is None):
            explicit_ratio = _first_present(metrics, ["reduction_ratio", "ratio"])
        reduction_ratio = _derive_reduction_ratio(explicit_ratio, pockets_total, pockets_used)

        runtime_total_s = _first_present(
            summary,
            ["runtime_total_s", "total_runtime_sec", "runtime_sec", "total_s", "elapsed_s", "elapsed_seconds", "elapsed"],
        )
        if runtime_total_s is None:
            runtime_total_s = timing.get("total_s")
        if runtime_total_s is None:
            runtime_total_s = payload.get("elapsed_seconds")

        rows.append(
            {
                "Modo": "Completo" if label == "full" else "Reduzido",
                "Melhor score (cheap)": best_cheap,
                "Avaliações": n_eval_total,
                "Bolsões totais": pockets_total,
                "Bolsões usados": pockets_used,
                "Razão de redução": reduction_ratio,
                "Tempo (s)": runtime_total_s,
            }
        )
    return rows

def run_pipeline_with_progress(
    cfg: Config,
    receptor_path: str,
    peptide_path: str,
    out_dir: Path,
    total_generations: int,
    progress_bar: st.delta_generator.DeltaGenerator,
    progress_text: st.delta_generator.DeltaGenerator,
    poll_interval: float = 0.3,
) -> tuple[Any, float]:
    """Executa o pipeline em thread paralela e atualiza o progresso da UI."""

    result_holder: dict[str, Any] = {}
    metrics_path = out_dir / "metrics.jsonl"

    def _run_pipeline() -> None:
        """Executa o pipeline e captura erro/tempo para a UI."""

        try:
            start_time = time.perf_counter()
            result_holder["result"] = run_pipeline(cfg, receptor_path, peptide_path, str(out_dir))
            result_holder["elapsed"] = time.perf_counter() - start_time
        except Exception as exc:  # noqa: BLE001
            result_holder["error"] = exc

    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()

    last_generation: int | None = None
    while thread.is_alive():
        last_generation = read_last_metrics_generation(metrics_path)
        progress_value = compute_progress(last_generation, total_generations)
        progress_bar.progress(progress_value)
        progress_text.write(format_progress_text(last_generation, total_generations, progress_value))
        time.sleep(poll_interval)

    thread.join()
    last_generation = read_last_metrics_generation(metrics_path) or last_generation
    progress_value = compute_progress(last_generation, total_generations)
    if total_generations > 0 and progress_value < 1.0:
        progress_value = 1.0
        # PT-BR: garante fechamento em 100% mesmo se a última geração não foi lida.
        last_generation = total_generations
    progress_bar.progress(progress_value)
    progress_text.write(format_progress_text(last_generation, total_generations, progress_value))

    if "error" in result_holder:
        raise result_holder["error"]
    return result_holder["result"], result_holder.get("elapsed", 0.0)


class _ProtocolExecutionMixin:
    """Fluxo de execução e retomada do protocolo experimental."""

    def _render_pending_protocol_error(self, pending_error: dict[str, Any]) -> None:
        """Mostra erro pausado e permite continuar ou cancelar."""

        payload = pending_error.get("payload") if isinstance(pending_error.get("payload"), dict) else {}
        failed = pending_error.get("failed") if isinstance(pending_error.get("failed"), dict) else {}
        st.error("O experimento foi pausado porque uma execução falhou.")
        st.write(f"Rodada: {failed.get('round_index', 'NA')}")
        st.write(f"Modo: {failed.get('mode_label', 'NA')}")
        st.write(f"Heurística: {failed.get('heuristic_label', 'NA')}")
        st.write(f"Algoritmo: {failed.get('algorithm_label', 'NA')}")
        st.write(f"Pasta: {failed.get('out_dir', 'NA')}")
        st.code(str(failed.get("error_message", "Erro sem mensagem.")), language="text")
        st.info("Escolha se deseja continuar ignorando esta execução ou cancelar o experimento agora.")

        col_continue, col_cancel = st.columns(2)
        with col_continue:
            if st.button("Continuar experimento", key="protocol_error_continue"):
                st.session_state.pop(PROTOCOL_ERROR_PENDING_KEY, None)
                self._resume_protocol_after_error(payload, failed, continue_after_error=True)
        with col_cancel:
            if st.button("Cancelar experimento", key="protocol_error_cancel"):
                experiment_dir = Path(str(payload.get("experiment_dir", "")))
                if str(experiment_dir):
                    _write_experiment_status(
                        experiment_dir,
                        {
                            "status": "cancelled",
                            "message": "Experimento cancelado pelo usuário após erro.",
                            "protocol_kind": payload.get("protocol_kind"),
                            "protocol_label": payload.get("protocol_label"),
                            "protocol_question": payload.get("protocol_question"),
                            "failed": failed,
                            "results": payload.get("results_by_protocol", []),
                        },
                    )
                    update_experiment_index(
                        protein_name=str(payload.get("protein_name", "proteina_sem_nome")),
                        experiment_dir=experiment_dir,
                        mode="full_protocol",
                        status="cancelled",
                        extra={"failed": failed},
                    )
                st.session_state.pop(PROTOCOL_ERROR_PENDING_KEY, None)
                st.warning("Experimento cancelado. Os resultados já concluídos foram preservados.")

    def _resume_protocol_after_error(
        self,
        payload: dict[str, Any],
        failed: dict[str, Any],
        *,
        continue_after_error: bool,
    ) -> None:
        """Retoma o protocolo depois que o usuário decide continuar."""

        results_by_protocol = list(payload.get("results_by_protocol", []))
        if continue_after_error:
            results_by_protocol.append(
                {
                    "Rodada": failed.get("round_index"),
                    "Seed": failed.get("seed"),
                    "Modo": failed.get("mode_label"),
                    "Heurística": failed.get("heuristic_label"),
                    "Algoritmo": failed.get("algorithm_label"),
                    "Status": "Erro ignorado pelo usuário",
                    "Erro": failed.get("error_message"),
                    "Pasta": failed.get("out_dir"),
                }
            )
        self._run_protocol_jobs_from_payload(
            payload,
            start_job_index=int(failed.get("job_index", -1)) + 1,
            results_by_protocol=results_by_protocol,
        )

    def _run_protocol_jobs_from_payload(
        self,
        payload: dict[str, Any],
        *,
        start_job_index: int,
        results_by_protocol: list[dict[str, Any]],
    ) -> None:
        """Executa ou retoma lista de jobs do protocolo completo."""

        experiment_dir = Path(str(payload["experiment_dir"]))
        results_dir = Path(str(payload["results_dir"]))
        receptor_path = str(payload["receptor_path"])
        peptide_path = str(payload["peptide_path"])
        resolved_cfg = dict(payload["resolved_cfg"])
        n_rounds = int(payload["n_rounds"])
        top_pockets = int(payload["top_pockets"])
        base_seed = int(payload["base_seed"])
        total_generations = int(payload["total_generations"])
        confidence_policy_mode = str(payload["confidence_policy_mode"])
        protocol_kind = str(payload.get("protocol_kind", "heuristic_reduced"))
        protein_name = str(payload["protein_name"])

        jobs = _build_protocol_job_list(
            results_dir,
            top_pockets=top_pockets,
            confidence_policy_mode=confidence_policy_mode,
            n_rounds=n_rounds,
            protocol_kind=protocol_kind,
        )
        total_jobs = len(jobs)
        st.write(f"Executando {payload.get('protocol_label', 'protocolo')}: {total_jobs} execuções.")
        if payload.get("protocol_question"):
            st.caption(str(payload["protocol_question"]))
        for job_index in range(int(start_job_index), total_jobs):
            job = jobs[job_index]
            round_index = int(job["round_index"])
            run = job["run"]
            run_cfg_dict = dict(resolved_cfg)
            run_cfg_dict["seed"] = int(base_seed + ((round_index - 1) * 100) + _algorithm_seed_offset(run["engine"]))
            run_cfg_dict["experimental_round"] = int(round_index)
            run_cfg_dict["n_rounds"] = int(n_rounds)
            run_cfg_dict["protocol_kind"] = protocol_kind
            run_cfg_dict["protocol_label"] = payload.get("protocol_label")
            run_cfg_dict["protocol_question"] = payload.get("protocol_question")
            run_cfg_dict["full_search"] = run["full_search"]
            run_cfg_dict["search_space_mode"] = run["search_space_mode"]
            run_cfg_dict["search_engine"] = run["engine"]
            run_cfg_dict["confidence_policy"] = _merge_run_confidence_policy(
                resolved_cfg,
                run["confidence_policy"],
            )
            if run["top_pockets"] is not None:
                run_cfg_dict["top_pockets"] = run["top_pockets"]

            run_out_dir = run["out_dir"]
            run_out_dir.mkdir(parents=True, exist_ok=True)
            _write_experiment_status(
                experiment_dir,
                {
                    "status": "running",
                    "current_job": job_index + 1,
                    "total_jobs": total_jobs,
                    "protocol_kind": protocol_kind,
                    "protocol_label": payload.get("protocol_label"),
                    "protocol_question": payload.get("protocol_question"),
                    "round_index": round_index,
                    "mode": run["mode_label"],
                    "heuristic": run["heuristic_label"],
                    "algorithm": run["algorithm_label"],
                    "out_dir": str(run_out_dir).replace("\\", "/"),
                    "results": results_by_protocol,
                },
            )
            st.markdown(
                f"**Execução {job_index + 1}/{total_jobs} - Rodada {round_index}/{n_rounds} - "
                f"{run['label']} - {run['algorithm_label']}**"
            )
            progress_bar = st.progress(0.0)
            progress_text = st.empty()
            try:
                result, elapsed = run_pipeline_with_progress(
                    Config(**run_cfg_dict),
                    receptor_path,
                    peptide_path,
                    run_out_dir,
                    total_generations,
                    progress_bar,
                    progress_text,
                )
                write_resolved_config(run_out_dir, run_cfg_dict)
                metrics_records = load_jsonl(run_out_dir / "metrics.jsonl")
                metrics = summarize_metrics(metrics_records)
                summary_path = run_out_dir / "summary.json"
                summary_data: dict[str, Any] = {}
                if summary_path.exists():
                    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
                results_by_protocol.append(
                    {
                        "Rodada": round_index,
                        "Seed": run_cfg_dict["seed"],
                        "Modo": run["mode_label"],
                        "Heurística": run["heuristic_label"],
                        "Algoritmo": run["algorithm_label"],
                        "Status": "Concluído",
                        "Melhor score (cheap)": result.best_pose.score_cheap,
                        "Evina/final_rescore": summary_data.get("final_rescore"),
                        "DockQ": summary_data.get("dockq"),
                        "Fnat": summary_data.get("fnat"),
                        "iRMSD": summary_data.get("irmsd"),
                        "LRMSD": summary_data.get("lrmsd"),
                        "Fallback executado": summary_data.get(
                            "fallback_executed", False
                        ),
                        "Avaliações": metrics.get("n_eval"),
                        "Tempo (s)": round(elapsed, 2),
                        "Pasta": str(run_out_dir).replace("\\", "/"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failed = {
                    "job_index": job_index,
                    "round_index": round_index,
                    "seed": run_cfg_dict["seed"],
                    "mode_label": run["mode_label"],
                    "heuristic_label": run["heuristic_label"],
                    "algorithm_label": run["algorithm_label"],
                    "out_dir": str(run_out_dir).replace("\\", "/"),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                payload["results_by_protocol"] = results_by_protocol
                st.session_state[PROTOCOL_ERROR_PENDING_KEY] = {"payload": payload, "failed": failed}
                _write_experiment_status(
                    experiment_dir,
                    {
                        "status": "paused_on_error",
                        "failed": failed,
                        "results": results_by_protocol,
                    },
                )
                update_experiment_index(
                    protein_name=protein_name,
                    experiment_dir=experiment_dir,
                    mode="full_protocol",
                    status="paused_on_error",
                    extra={"failed": failed},
                )
                st.rerun()

        _write_experiment_status(
            experiment_dir,
            {
                "status": "ok",
                "message": "Experimento finalizado.",
                "total_jobs": total_jobs,
                "protocol_kind": protocol_kind,
                "protocol_label": payload.get("protocol_label"),
                "protocol_question": payload.get("protocol_question"),
                "results": results_by_protocol,
            },
        )
        update_experiment_index(
            protein_name=protein_name,
            experiment_dir=experiment_dir,
            mode="full_protocol",
            status="ok",
            extra={
                "result_path": str(results_dir).replace("\\", "/"),
                "n_rounds": int(n_rounds),
                "protocol_kind": protocol_kind,
                "protocol_label": payload.get("protocol_label"),
            },
        )
        set_state(**{StateKeys.LAST_OUT_DIR: str(experiment_dir), StateKeys.REPORTS_ROOT: str(experiment_dir)})
        update_recent_out_dirs(str(experiment_dir))
        st.success("Experimento realizado com sucesso.")
        st.info(f"Experimento salvo em: {experiment_dir}")
        st.subheader("Resumo da execução")
        st.table(results_by_protocol)
        st.write("Artefatos")
        st.caption(
            "Os arquivos result.json/summary.json foram salvos em "
            "`results/<full|reduced>/rodada_XXX/...`."
        )


class DockingPage(_ProtocolExecutionMixin, BasePage):
    """Página principal de execução do docking com controles de entrada."""

    id = "Docking"
    title = "Docking"

    def render(self, state: AppState) -> None:
        """Renderiza a interface do usuário para execução do docking."""

        st.header("Execução de Docking")

        pending_error = st.session_state.get(PROTOCOL_ERROR_PENDING_KEY)
        if isinstance(pending_error, dict):
            self._render_pending_protocol_error(pending_error)
            return

        (
            prepared_path,
            use_prepared,
            prepared_peptide_path,
            use_prepared_peptide,
        ) = render_prepared_input_controls()

        receptor_upload = st.file_uploader("Receptor (.pdb)", type=["pdb"], key="receptor_upload")
        peptide_upload = st.file_uploader("Peptídeo (.pdb)", type=["pdb"], key="peptide_upload")

        st.subheader("Identificação do experimento")
        protein_name = st.text_input(
            "Nome da proteína/complexo/PDB",
            value=st.session_state.get(StateKeys.PROTEIN_NAME, ""),
            key=StateKeys.PROTEIN_NAME,
            placeholder="Ex.: 1A1N",
        )
        out_dir = str(Path("runs_experimentos") / (protein_name.strip() or "proteina_sem_nome") / "<data-hora>")
        st.caption(
            "Os resultados serão salvos automaticamente em "
            "`runs_experimentos/<proteína>/<YYYY-MM-DD_HH-MM-SS>/results/` "
            "e separados em `full/` e/ou `reduced/` conforme o protocolo escolhido."
        )

        st.subheader("Protocolo de experimento")
        protocol_options = [
            "Heurística no modo reduzido",
            "Heurística no modo completo",
            "Completo vs reduzido",
            "Protocolo completo",
        ]
        st.session_state.setdefault(StateKeys.EXPERIMENT_PROTOCOL, protocol_options[0])
        if st.session_state[StateKeys.EXPERIMENT_PROTOCOL] == "Heuristica no modo reduzido":
            st.session_state[StateKeys.EXPERIMENT_PROTOCOL] = "Heurística no modo reduzido"
        elif st.session_state[StateKeys.EXPERIMENT_PROTOCOL] == "Heuristica no modo completo":
            st.session_state[StateKeys.EXPERIMENT_PROTOCOL] = "Heurística no modo completo"
        protocol_label = st.selectbox(
            "O que deseja comparar?",
            protocol_options,
            key=StateKeys.EXPERIMENT_PROTOCOL,
        )
        protocol_kind = _protocol_label_to_kind(protocol_label)
        st.info(
            _protocol_kind_to_question(protocol_kind)
            + " Em todos os protocolos os algoritmos comparados são ABC+AG+VGOS, AG e PSO."
        )
        top_pockets_value = st.slider(
            "Melhores bolsões usados no modo reduzido",
            min_value=1,
            max_value=32,
            value=32,
            key=StateKeys.COMPARE_TOP_POCKETS,
            disabled=protocol_kind == "heuristic_full",
        )

        st.subheader("Configuração")
        config_choice = st.radio(
            "Configuração",
            options=["Padrão (configs/default.yaml)", "Enviar YAML"],
            index=0,
            key=StateKeys.CONFIG_CHOICE,
        )
        uploaded_config = None
        if config_choice == "Enviar YAML":
            uploaded_config = st.file_uploader("Configuração YAML", type=["yaml", "yml"], key=StateKeys.CONFIG_UPLOAD)
            if uploaded_config is not None:
                try:
                    config_text = uploaded_config.getvalue().decode("utf-8")
                    set_state(
                        **{
                            StateKeys.LOADED_CONFIG: yaml.safe_load(config_text) or {},
                            StateKeys.CONFIG_SOURCE_LABEL: f"YAML enviado: {uploaded_config.name}",
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error("Não foi possível ler o YAML enviado.")
                    st.exception(exc)
        else:
            set_state(
                **{
                    StateKeys.LOADED_CONFIG: load_config(str(DEFAULT_CONFIG_PATH)),
                    StateKeys.CONFIG_SOURCE_LABEL: "Padrão (configs/default.yaml)",
                }
            )

        st.caption("A configuração pode ser ajustada na aba Configurações.")

        st.subheader("Parâmetros de execução")
        base_cfg = st.session_state.get(StateKeys.LOADED_CONFIG, {})
        current_overrides = dict(st.session_state.get(StateKeys.CONFIG_OVERRIDES, {}))
        resolved_preview = apply_overrides(base_cfg, current_overrides)
        total_generations_preview = int(resolved_preview.get("generations", 0) or 0)
        pop_size_preview = int(resolved_preview.get("pop_size", 0) or 0)
        debug_enabled_preview = bool(resolved_preview.get("debug_log_enabled", False))
        confidence_policy_enabled_preview, confidence_policy_mode_preview = (
            resolve_confidence_policy_state(resolved_preview)
        )

        st.write(f"Total de gerações (configuração resolvida): {total_generations_preview}")
        st.write(f"Tamanho da população (configuração resolvida): {pop_size_preview}")

        st.session_state.setdefault(StateKeys.OVERRIDE_GENERATIONS, total_generations_preview or 1)
        st.session_state.setdefault(StateKeys.OVERRIDE_POP_SIZE, pop_size_preview or 1)
        st.session_state.setdefault(StateKeys.OVERRIDE_DEBUG_ENABLED, debug_enabled_preview)
        st.session_state.setdefault(StateKeys.OVERRIDE_DEBUG_PATH, None)
        st.session_state.setdefault(
            StateKeys.OVERRIDE_CONFIDENCE_POLICY_ENABLED,
            bool(confidence_policy_enabled_preview),
        )
        st.session_state.setdefault(
            StateKeys.OVERRIDE_CONFIDENCE_POLICY_MODE_LABEL,
            confidence_policy_mode_to_label(confidence_policy_mode_preview),
        )

        col_gen, col_pop = st.columns(2)
        with col_gen:
            generations_value = st.number_input(
                "Sobrescrever gerações",
                min_value=1,
                step=1,
                key=StateKeys.OVERRIDE_GENERATIONS,
            )
        with col_pop:
            pop_size_value = st.number_input(
                "Sobrescrever pop_size",
                min_value=1,
                step=1,
                key=StateKeys.OVERRIDE_POP_SIZE,
            )
        configured_n_rounds = max(
            1,
            int(resolved_preview.get("n_rounds", 5) or 5),
        )
        st.session_state.setdefault(
            StateKeys.OVERRIDE_N_ROUNDS,
            configured_n_rounds,
        )
        n_rounds_value = st.number_input(
            "Quantidade de rodadas estatísticas",
            min_value=1,
            max_value=100,
            step=1,
            key=StateKeys.OVERRIDE_N_ROUNDS,
            help="Cada rodada executa os algoritmos e condições do protocolo escolhido.",
        )
        jobs_preview = len(
            _build_protocol_job_list(
                Path("_preview"),
                top_pockets=int(top_pockets_value or 1),
                confidence_policy_mode="assist_selection",
                n_rounds=int(n_rounds_value or 1),
                protocol_kind=protocol_kind,
            )
        )
        st.caption(f"Este protocolo irá gerar {jobs_preview} execuções.")

        st.subheader("Heurística de confiança populacional")
        mode_label_options = confidence_policy_mode_labels()
        enabled_mode_label_options = [label for label in mode_label_options if label != confidence_policy_mode_to_label("disabled")]
        mode_label_value = str(
            st.session_state[StateKeys.OVERRIDE_CONFIDENCE_POLICY_MODE_LABEL] or ""
        )
        if mode_label_value not in enabled_mode_label_options:
            mode_label_value = confidence_policy_mode_to_label("assist_selection")
        confidence_policy_mode_label_value = st.selectbox(
            "Como a heurística deve atuar?",
            options=enabled_mode_label_options,
            key=StateKeys.OVERRIDE_CONFIDENCE_POLICY_MODE_LABEL,
            help="O braço sem heurística será executado automaticamente com a política desligada.",
        )

        st.subheader("Debug")
        debug_enabled_value = st.checkbox(
            "Gerar debug.jsonl",
            key=StateKeys.OVERRIDE_DEBUG_ENABLED,
        )
        if debug_enabled_value:
            debug_path_preview = Path(out_dir) / "debug" / "debug.jsonl"
            st.caption(f"Debug será salvo em: {debug_path_preview}")

        current_overrides["generations"] = int(generations_value)
        current_overrides["pop_size"] = int(pop_size_value)
        current_overrides["debug_log_enabled"] = bool(debug_enabled_value)
        current_overrides["debug_log_path"] = None
        current_overrides["confidence_policy"] = build_confidence_policy_override(
            enabled=True,
            selected_mode_or_label=confidence_policy_mode_label_value,
        )
        st.session_state[StateKeys.CONFIG_OVERRIDES] = current_overrides

        if st.button("Executar"):
            if config_choice == "Enviar YAML" and uploaded_config is None:
                st.error("Envie um arquivo de configuração YAML ou selecione a configuração padrão.")
                return
            valid_protein, protein_error = validate_protein_name(protein_name)
            if not valid_protein:
                st.error(protein_error or "Informe o nome da proteína/complexo/PDB.")
                return
            if use_prepared:
                prepared_file = Path(prepared_path)
                if not prepared_file.exists():
                    st.error(
                        "Arquivo do receptor preparado não foi encontrado no disco: "
                        f"{prepared_file}. Gere novamente ou faça upload."
                    )
                    return
            else:
                if receptor_upload is None:
                    st.error("Envie os arquivos PDB de receptor e peptídeo antes de executar.")
                    return
            prepared_peptide_file: Path | None = None
            if use_prepared_peptide:
                prepared_peptide_file = Path(prepared_peptide_path)
                if not prepared_peptide_file.exists():
                    st.error(
                        "O peptídeo preparado não foi encontrado no disco: "
                        f"{prepared_peptide_file}."
                    )
                    return
            elif peptide_upload is None:
                st.error("Envie os arquivos PDB de receptor e peptídeo antes de executar.")
                return

            preflight_cfg = apply_overrides(
                st.session_state.get(StateKeys.LOADED_CONFIG, {}),
                st.session_state.get(StateKeys.CONFIG_OVERRIDES, {}),
            )
            if not peptide_preflight_allows_execution(
                cfg=preflight_cfg,
                prepared_path=prepared_peptide_file,
                uploaded_file=peptide_upload,
            ):
                return

            layout = create_experiment_layout(protein_name)
            experiment_dir = layout["experiment_dir"]
            inputs_dir = layout["inputs_dir"]
            results_dir = layout["results_dir"]

            try:
                receptor_file, peptide_file = persist_experiment_inputs(
                    prepared_receptor_path=prepared_file if use_prepared else None,
                    receptor_upload=receptor_upload,
                    prepared_peptide_path=prepared_peptide_file,
                    peptide_upload=peptide_upload,
                    inputs_dir=inputs_dir,
                )
            except OSError as exc:
                st.error(f"Não foi possível salvar as entradas do experimento: {exc}")
                return
            receptor_path, peptide_path = str(receptor_file), str(peptide_file)
            if not receptor_peptide_pair_preflight_allows_execution(
                cfg=preflight_cfg,
                receptor_path=receptor_file,
                peptide_path=peptide_file,
            ):
                return

            if config_choice == "Enviar YAML" and uploaded_config is not None:
                config_path = save_uploaded_file(uploaded_config, inputs_dir)
                raw_cfg = load_config(str(config_path))
            else:
                raw_cfg = load_config(str(DEFAULT_CONFIG_PATH))

            resolved_cfg = apply_overrides(raw_cfg, st.session_state.get(StateKeys.CONFIG_OVERRIDES, {}))

            # Quando o usuário informa um PDB canônico (ex.: 1A1N), grave-o
            # explicitamente em TODOS os runs. Os arquivos persistidos como
            # receptor.pdb/peptide.pdb perdem o id no nome.
            resolved_cfg = apply_experiment_complex_identity(resolved_cfg, protein_name)

            total_generations = int(resolved_cfg.get("generations", 0) or 0)
            n_rounds = int(n_rounds_value or 1)
            base_seed = int(resolved_cfg.get("seed", 7) or 7)

            top_pockets = int(top_pockets_value or 1)
            confidence_policy_mode = build_confidence_policy_override(
                enabled=True,
                selected_mode_or_label=confidence_policy_mode_label_value,
            )["mode"]
            payload = {
                "protein_name": protein_name,
                "experiment_dir": str(experiment_dir),
                "results_dir": str(results_dir),
                "receptor_path": receptor_path,
                "peptide_path": peptide_path,
                "resolved_cfg": resolved_cfg,
                "n_rounds": int(n_rounds),
                "top_pockets": int(top_pockets),
                "base_seed": int(base_seed),
                "total_generations": int(total_generations),
                "confidence_policy_mode": str(confidence_policy_mode),
                "protocol_kind": str(protocol_kind),
                "protocol_label": str(protocol_label),
                "protocol_question": _protocol_kind_to_question(protocol_kind),
                "results_by_protocol": [],
            }
            self._run_protocol_jobs_from_payload(payload, start_job_index=0, results_by_protocol=[])

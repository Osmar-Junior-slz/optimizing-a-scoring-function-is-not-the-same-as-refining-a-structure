"""Dashboard cientifico para experimentos de docking proteina-peptideo."""

from __future__ import annotations

import json
import math
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from dockingpp.gui.pages.base import BasePage
from dockingpp.gui.services.dialog_service import choose_directory
from dockingpp.gui.state import AppState, StateKeys
from dockingpp.reporting.loaders import load_scientific_runs


class _DashboardShellMixin:
    """Seleção, filtros e avisos do dashboard."""

    def _render_header(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(
            f"""
            <div class="dash-topbar">
                <div class="dash-menu">☰</div>
                <div class="dash-title">
                    DASHBOARD DE ANALISE DE DOCKING PEPTIDICO HIBRIDO (ABC-GA-VGOS)
                    <span>| Autor: Osmar A. Ribeiro Jr.</span>
                </div>
                <div class="dash-clock">{now}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _select_experiment_folder(self, root: Path) -> Path:
        protein_dirs = _dashboard_protein_dirs(root)
        if not protein_dirs:
            return root

        protein_names = [path.name for path in protein_dirs]
        selected_protein = st.selectbox(
            "Proteina/complexo",
            protein_names,
            key="dashboard_selected_protein",
        )
        protein_dir = root / selected_protein
        run_dirs = sorted([path for path in protein_dir.iterdir() if path.is_dir()], reverse=True)
        if not run_dirs:
            return protein_dir

        run_names = [path.name for path in run_dirs]
        selected_run = st.selectbox(
            "Rodada historica",
            run_names,
            key="dashboard_selected_run",
        )
        selected = protein_dir / selected_run
        st.caption(f"Carregando experimento: {selected}")
        return selected

    def _render_filter_bar(self, runs: list[dict[str, Any]], selected_root: Path) -> list[dict[str, Any]]:
        complexes = _sorted_unique(run.get("complex_id") for run in runs)
        algorithms = _ordered_algorithms(run.get("algorithm") for run in runs)
        modes = _sorted_unique(run.get("mode") for run in runs)
        filter_suffix = _filter_key_suffix(selected_root)

        st.markdown('<div class="dash-panel filter-panel">', unsafe_allow_html=True)
        heuristics = _ordered_heuristics(_heuristic_label(run) for run in runs)

        c1, c2, c3, c4 = _columns([1.2, 1.2, 1.0, 1.15])
        with c1:
            selected_complexes = st.multiselect(
                "Complexo/PDB",
                complexes,
                default=complexes,
                key=f"dashboard_complexes_{filter_suffix}",
            )
        with c2:
            selected_algorithms = st.multiselect(
                "Algoritmo",
                algorithms,
                default=algorithms,
                key=f"dashboard_algorithms_{filter_suffix}",
            )
        with c3:
            selected_modes = st.multiselect(
                "Modo",
                modes,
                default=modes,
                key=f"dashboard_modes_{filter_suffix}",
            )
        with c4:
            selected_heuristics = st.multiselect(
                "Heurística",
                heuristics,
                default=heuristics,
                key=f"dashboard_heuristics_{filter_suffix}",
            )
        st.markdown("</div>", unsafe_allow_html=True)

        selected_complexes = selected_complexes or complexes
        selected_algorithms = selected_algorithms or algorithms
        selected_modes = selected_modes or modes
        selected_heuristics = selected_heuristics or heuristics

        return [
            run
            for run in runs
            if str(run.get("complex_id")) in selected_complexes
            and _algorithm_label(run.get("algorithm")) in selected_algorithms
            and str(run.get("mode")) in selected_modes
            and _heuristic_label(run) in selected_heuristics
        ]

    def _render_warnings(self, runs: list[dict[str, Any]]) -> None:
        if any(run.get("fallback_executed") for run in runs):
            st.warning("Execução reduzida caiu para busca completa.")
        missing = sorted({warning for run in runs for warning in run.get("warnings", [])})
        if missing:
            st.info("Campos ausentes em algumas execuções: " + "; ".join(missing[:5]))



class _DashboardStructuralMixin:
    """Painéis de evidência e métricas estruturais."""

    def _render_dashboard_grid(
        self,
        runs: list[dict[str, Any]],
        selected_root: Path,
        structural_rows: list[dict[str, Any]],
        structural_audit: dict[str, Any] | None,
    ) -> None:
        self._render_data_provenance_panel(
            runs,
            selected_root,
            structural_rows,
            structural_audit,
        )
        if not structural_rows:
            if _structural_audit_label(structural_audit) == "Não aplicável":
                st.info(
                    "Métricas estruturais não aplicáveis a estas execuções: "
                    "o parceiro de referência não atende à definição operacional "
                    "de peptídeo. Os valores ausentes não são zeros nem falhas."
                )
            else:
                st.warning(
                    "Metricas estruturais ainda nao foram encontradas nos summaries. "
                    "Para execucoes antigas, use scripts/evaluate_saved_poses.py."
                )
            return
        self._render_core_evidence_panel(structural_rows)
        left, right = _columns([0.52, 0.48])
        with left:
            self._render_hybrid_relevance_panel(structural_rows)
            self._render_heuristic_effect_panel(structural_rows)
        with right:
            self._render_structural_statistics_panel(structural_rows)
            self._render_structural_article_panel(structural_rows)

    def _render_data_provenance_panel(
        self,
        runs: list[dict[str, Any]],
        selected_root: Path,
        structural_rows: list[dict[str, Any]],
        structural_audit: dict[str, Any] | None,
    ) -> None:
        complexes = _sorted_unique(run.get("complex_id") for run in runs)
        run_dirs = sorted({str(run.get("run_dir_relative") or run.get("run_dir") or "") for run in runs})
        st.markdown('<section class="dash-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">FONTE DOS DADOS CARREGADOS</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = _columns([1.0, 0.8, 0.9, 1.3])
        with c1:
            st.metric("Proteina/PDB", ", ".join(complexes) if complexes else "NA")
        with c2:
            st.metric("Execucoes filtradas", len(runs))
        with c3:
            st.metric("Auditoria estrutural", _structural_audit_label(structural_audit))
            not_applicable = _structural_audit_count(
                structural_audit,
                "not_applicable_rows",
            )
            st.caption(
                f"Poses disponíveis: {len(structural_rows)}; "
                f"não aplicáveis: {not_applicable}"
            )
        with c4:
            st.caption(f"Pasta selecionada: {selected_root}")
            st.caption("Amostra de pastas de execucao: " + "; ".join(run_dirs[:3]))
        st.markdown("</section>", unsafe_allow_html=True)

    def _render_core_evidence_panel(self, rows: list[dict[str, Any]]) -> None:
        hybrid_summary = _structural_hybrid_summary(rows)
        heuristic_summary = _structural_heuristic_summary(rows)
        st.markdown('<section class="dash-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">RESPOSTA DIRETA PARA O ARTIGO</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = _columns([1, 1, 1, 1])
        with c1:
            st.metric("Melhor DockQ medio", hybrid_summary.get("best_algorithm", "NA"))
        with c2:
            st.metric("Hibrido vs PSO", hybrid_summary.get("hybrid_vs_pso", "NA"))
        with c3:
            st.metric("CAPRI media/alta", hybrid_summary.get("capri_success", "NA"))
        with c4:
            st.metric("Reducao media tempo", heuristic_summary.get("speedup_label", "NA"))
        st.caption(
            "Qualidade avaliada por DockQ CAPRI-peptideo. O score interno e exibido apenas "
            "como diagnostico e nao como prova de acuracia estrutural."
        )
        st.markdown("</section>", unsafe_allow_html=True)

    def _render_hybrid_relevance_panel(self, rows: list[dict[str, Any]]) -> None:
        st.markdown('<section class="dash-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">RELEVANCIA DO HIBRIDO ABC+AG+VGOS</div>', unsafe_allow_html=True)
        table = _structural_hybrid_table(rows)
        c1, c2 = _columns([1, 1])
        with c1:
            chart = _structural_algorithm_chart(rows)
            _render_image(chart)
        with c2:
            win_chart = _structural_wins_chart(rows)
            _render_image(win_chart)
        if table.empty:
            st.info("Dados insuficientes para comparar o hibrido.")
        else:
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.caption("DockQ: maior e melhor. Vitorias pareadas por proteina, rodada e condicao.")
        st.markdown("</section>", unsafe_allow_html=True)

    def _render_heuristic_effect_panel(self, rows: list[dict[str, Any]]) -> None:
        st.markdown('<section class="dash-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">EFEITO DA HEURISTICA EM CADA MOTOR</div>', unsafe_allow_html=True)
        table = _structural_heuristic_table(rows)
        _render_image(_structural_heuristic_chart(rows))
        if table.empty:
            st.info("Dados insuficientes para comparar com/sem heuristica.")
        else:
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.caption("Reducao de tempo positiva e melhor; delta DockQ proximo de zero preserva qualidade.")
        st.markdown("</section>", unsafe_allow_html=True)

    def _render_structural_statistics_panel(self, rows: list[dict[str, Any]]) -> None:
        st.markdown('<section class="dash-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">METRICAS ESTRUTURAIS CAPRI</div>', unsafe_allow_html=True)
        st.dataframe(_structural_statistics_table(rows), use_container_width=True, hide_index=True)
        st.caption("IC95 por t-Student; Fnat usa 4 A e a interface do iRMSD usa C-beta/C-alfa a 8 A.")
        st.markdown("</section>", unsafe_allow_html=True)

    def _render_structural_article_panel(self, rows: list[dict[str, Any]]) -> None:
        st.markdown('<section class="dash-panel tall-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">LEITURA CIENTIFICA ATUAL</div>', unsafe_allow_html=True)
        summary = _structural_hybrid_summary(rows)
        table = _structural_hybrid_table(rows)
        status = summary.get("hybrid_status", "inconclusivo")
        if status == "sustentado":
            st.success("O hibrido possui o maior DockQ medio neste recorte.")
        else:
            st.warning("A superioridade estrutural do hibrido ainda nao esta sustentada neste recorte.")
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption("Conclusao limitada ao conjunto filtrado; a unidade independente final sera o complexo.")
        st.markdown("</section>", unsafe_allow_html=True)



class _DashboardAnalysisMixin:
    """Painéis analíticos e comparações científicas."""

    def _render_essential_charts(self, runs: list[dict[str, Any]]) -> None:
        st.markdown('<section class="dash-panel essential-dashboard">', unsafe_allow_html=True)
        runtime_chart = _runtime_heuristic_pair_chart(runs)
        energy_chart = _energy_heuristic_pair_chart(runs)
        if runtime_chart is None and energy_chart is None:
            st.info("Para gerar os gráficos, rode os mesmos algoritmos com e sem heurística.")
        else:
            _render_image(runtime_chart)
            _render_image(energy_chart)
        st.markdown("</section>", unsafe_allow_html=True)

    def _render_complex_panel(self, runs: list[dict[str, Any]]) -> None:
        selected_complex = _active_complex(runs)
        st.markdown('<section class="dash-panel tall-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">COMPLEXOS PDB ANALISADOS</div>', unsafe_allow_html=True)
        st.text_input("ID", key="dashboard_complex_search", placeholder="Buscar...")
        rows = []
        for complex_id in _sorted_unique(run.get("complex_id") for run in runs):
            active = "active-row" if complex_id == selected_complex else ""
            rows.append(f'<div class="complex-row {active}">{complex_id}</div>')
        st.markdown("".join(rows) or '<div class="muted">Nenhum complexo identificado.</div>', unsafe_allow_html=True)
        st.markdown("</section>", unsafe_allow_html=True)

    def _render_comparison_panel(self, runs: list[dict[str, Any]]) -> None:
        st.markdown('<section class="dash-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">COMPARATIVO DE DESEMPENHO DE ALGORITMOS</div>', unsafe_allow_html=True)
        c1, c2 = _columns([1, 1])
        with c1:
            _render_image(_bar_chart(runs, "energy_evina", "MEDIA DE ENERGIA (E_vina)", "Energia (kcal/mol)"))
        with c2:
            _render_image(_bar_chart(runs, "runtime_sec", "TEMPO MEDIO DE EXECUCAO", "Tempo (segundos)"))
        st.markdown("</section>", unsafe_allow_html=True)

    def _render_statistics_panel(self, runs: list[dict[str, Any]]) -> None:
        st.markdown('<section class="dash-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">ESTATISTICA DAS RODADAS</div>', unsafe_allow_html=True)
        question = _infer_experimental_question(runs)
        st.caption(question)
        stats_df = _statistical_summary_table(runs)
        if stats_df.empty:
            st.info("Dados insuficientes para estatística das rodadas.")
        else:
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            rounds = sorted({run.get("round_index") for run in runs if run.get("round_index") is not None})
            st.caption(
                f"Rodadas identificadas: {len(rounds) if rounds else 1}. "
                "Energia Evina: menor é melhor. IC95 calculado por t-Student quando n >= 2."
            )
            anova = _anova_energy_by_algorithm(runs)
            if anova:
                st.caption(
                    "ANOVA energia por algoritmo: "
                    f"F({anova['df_between']}, {anova['df_within']})={anova['F']:.4g}, "
                    f"p={anova['p']:.4g}, eta2={anova['eta2']:.4g}."
                )
        st.markdown("</section>", unsafe_allow_html=True)

    def _render_heuristic_comparison_panel(self, runs: list[dict[str, Any]]) -> None:
        st.markdown('<section class="dash-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">COMPARAÇÃO COM HEURÍSTICA vs SEM HEURÍSTICA</div>', unsafe_allow_html=True)
        without_runs = [run for run in runs if _heuristic_label(run) == "Sem heurística"]
        with_runs = [run for run in runs if _heuristic_label(run) == "Com heurística"]
        c1, c2 = _columns([1, 1])
        with c1:
            st.markdown("**Sem heurística**")
            if without_runs:
                _render_image(_bar_chart(without_runs, "energy_evina", "ENERGIA POR ALGORITMO", "E_vina"))
            else:
                st.info("Este experimento não possui execução gravada sem heurística. Rode uma execução com confidence_policy desligada/off para comparar.")
        with c2:
            st.markdown("**Com heurística**")
            if with_runs:
                _render_image(_bar_chart(with_runs, "energy_evina", "ENERGIA POR ALGORITMO", "E_vina"))
            else:
                st.info("Este experimento não possui execução gravada com heurística.")
        st.markdown("</section>", unsafe_allow_html=True)

    def _render_paired_effect_panel(self, runs: list[dict[str, Any]]) -> None:
        st.markdown('<section class="dash-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">EFEITO PAREADO DA HEURISTICA</div>', unsafe_allow_html=True)
        chart = _runtime_heuristic_pair_chart(runs)
        if chart is None:
            st.info("Para calcular o efeito pareado, rode os mesmos algoritmos com e sem heuristica.")
        else:
            _render_image(chart)
        st.markdown("</section>", unsafe_allow_html=True)

    def _render_distribution_panel(self, runs: list[dict[str, Any]]) -> None:
        st.markdown('<section class="dash-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">DISTRIBUIÇÃO DA ENERGIA FINAL E ESTABILIDADE</div>', unsafe_allow_html=True)
        _render_image(_boxplot(runs))
        st.markdown("</section>", unsafe_allow_html=True)

    def _render_method_panel(self, runs: list[dict[str, Any]]) -> None:
        st.markdown('<section class="dash-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">METODOLOGIA HÍBRIDA & FUNÇÃO GEOMÉTRICA</div>', unsafe_allow_html=True)
        c1, c2, c3 = _columns([1, 1, 0.8])
        with c1:
            _render_image(_convergence_chart(runs))
        with c2:
            _render_image(_radar_chart(runs))
        with c3:
            _render_image(_donut_chart(runs))
        st.markdown("</section>", unsafe_allow_html=True)

    def _render_top_poses_panel(self, runs: list[dict[str, Any]]) -> None:
        st.markdown('<section class="dash-panel tall-panel">', unsafe_allow_html=True)
        selected_complex = _active_complex(runs)
        st.markdown(
            f'<div class="panel-title">TOP POSES DO COMPLEXO ATUAL (Ex: {selected_complex})</div>',
            unsafe_allow_html=True,
        )
        table = _top_pose_table(runs)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption("Ranking baseado em Evina/final_rescore_energy. Menor energia e melhor.")
        st.markdown("</section>", unsafe_allow_html=True)

    def _render_kpi_panel(self, runs: list[dict[str, Any]]) -> None:
        best_energy_run = _min_by(runs, "energy_evina")
        fastest_run = _min_by(runs, "runtime_sec")
        best_algorithm = _best_algorithm_by_energy(runs)
        modes = sorted({str(run.get("mode") or "unknown") for run in runs})

        st.markdown('<section class="dash-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">RESUMO EXECUTIVO</div>', unsafe_allow_html=True)
        c1, c2 = _columns([1, 1])
        with c1:
            st.metric("Melhor Evina", _fmt_metric(best_energy_run.get("energy_evina") if best_energy_run else None))
            st.metric("Execuções", len(runs))
        with c2:
            st.metric("Menor tempo", _fmt_metric(fastest_run.get("runtime_sec") if fastest_run else None))
            st.metric("Melhor algoritmo", best_algorithm or "NA")
        st.metric("Modo executado", ", ".join(modes[:3]) if modes else "NA")
        st.markdown("</section>", unsafe_allow_html=True)

        st.markdown('<section class="dash-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">TABELA CONSOLIDADA ESTATISTICA</div>', unsafe_allow_html=True)
        stats = _statistical_summary_table(runs)
        if stats.empty:
            st.info("Dados insuficientes para estatistica consolidada.")
        else:
            st.dataframe(stats, use_container_width=True, hide_index=True)
        st.markdown("</section>", unsafe_allow_html=True)

    def _render_article_statistics_panel(self, runs: list[dict[str, Any]]) -> None:
        st.markdown('<section class="dash-panel tall-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">RESUMO PARA ARTIGO</div>', unsafe_allow_html=True)
        table = _article_ready_summary(runs)
        if table.empty:
            st.info("Dados insuficientes para resumo estatistico.")
        else:
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.caption("N >= 5 por grupo e um bom inicio; N >= 10 fortalece IC95 e testes.")
        st.markdown("</section>", unsafe_allow_html=True)



class ScientificDashboardPage(
    _DashboardShellMixin,
    _DashboardStructuralMixin,
    _DashboardAnalysisMixin,
    BasePage,
):
    """Página de dashboard científico com filtros, KPIs e gráficos."""

    id = "Dashboard cient\u00edfico"
    title = "Dashboard cient\u00edfico"

    def render(self, state: AppState) -> None:
        self._apply_dashboard_style()

        if st.session_state.get(StateKeys.REPORTS_ROOT_PENDING):
            st.session_state[StateKeys.REPORTS_ROOT] = st.session_state.pop(StateKeys.REPORTS_ROOT_PENDING)

        default_root = "runs_experimentos" if Path("runs_experimentos").exists() else "runs"
        current_root = str(st.session_state.get(StateKeys.REPORTS_ROOT, "") or "").strip()
        if not current_root or (current_root == "runs" and default_root == "runs_experimentos"):
            st.session_state[StateKeys.REPORTS_ROOT] = default_root

        folder_path = st.text_input("Pasta raiz dos experimentos", key=StateKeys.REPORTS_ROOT)
        if st.button("Selecionar pasta...", key="dashboard_choose_folder"):
            chosen = choose_directory()
            if chosen is not None:
                st.session_state[StateKeys.REPORTS_ROOT_PENDING] = chosen
                st.rerun()
        if not folder_path:
            st.info("Informe uma pasta com resultados para carregar o dashboard.")
            return

        root = Path(folder_path).expanduser()
        if not root.exists():
            st.warning("Pasta informada nao existe.")
            return

        selected_root = self._select_experiment_folder(root)
        runs = load_scientific_runs(selected_root)
        if not runs:
            st.warning("Nenhum JSON/JSONL valido foi encontrado para o dashboard.")
            return

        self._render_header()
        filtered_runs = self._render_filter_bar(runs, selected_root)
        if not filtered_runs:
            st.info("Nenhuma execucao encontrada com os filtros selecionados.")
            return
        self._render_warnings(filtered_runs)
        structural_rows, structural_audit = _load_structural_evidence(
            root,
            selected_root,
            filtered_runs,
        )
        self._render_dashboard_grid(
            filtered_runs,
            selected_root,
            structural_rows,
            structural_audit,
        )

    def _apply_dashboard_style(self) -> None:
        st.markdown(
            """
            <style>
            .stApp { background: #ffffff; color: #1f2933; }
            .block-container { max-width: 100%; padding: 0.35rem 0.65rem 1rem; }
            [data-testid="stSidebar"] { background: #f7f9fb; }
            h1, h2, h3, label, p, span { color: #1f2933; }
            .dash-topbar {
                align-items: center; background: #ffffff; border-bottom: 1px solid #d7dee6;
                display: flex; gap: 14px; height: 38px; margin: -0.35rem -0.65rem 0.45rem;
                padding: 0 14px;
            }
            .dash-menu { color: #243447; font-size: 22px; width: 28px; }
            .dash-title { color: #17212b; flex: 1; font-size: 20px; font-weight: 700; letter-spacing: 0; }
            .dash-title span { color: #415466; font-weight: 500; }
            .dash-clock { color: #516170; font-size: 14px; }
            .dash-panel {
                background: #ffffff;
                border: 1px solid #d7dee6; border-radius: 6px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
                margin-bottom: 8px; padding: 8px 10px;
            }
            .filter-panel { padding-bottom: 2px; }
            .tall-panel { min-height: 210px; }
            .panel-title {
                border-bottom: 1px solid #d7dee6; color: #17212b; font-size: 14px; font-weight: 800;
                margin: -2px -2px 8px; padding: 0 0 6px; text-transform: uppercase;
            }
            .complex-row {
                border-bottom: 1px solid #e2e8f0; color: #243447; font-size: 14px; padding: 5px 8px;
            }
            .active-row { background: #e8f2ff; color: #0f4d85; font-weight: 700; }
            .muted { color: #5f6f7d; font-size: 12px; line-height: 1.35; }
            div[data-testid="stMetric"] {
                background: #f8fafc; border: 1px solid #d7dee6; border-radius: 5px; padding: 8px 10px;
            }
            div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #17212b; }
            div[data-testid="stDataFrame"] { background: #ffffff; border: 1px solid #d7dee6; border-radius: 5px; }
            .stTextInput input, .stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"] {
                background: #ffffff; border-color: #cbd5df; color: #17212b;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


def _columns(spec: list[float]) -> list[Any]:
    try:
        return st.columns(spec)
    except TypeError:
        return st.columns(len(spec))


def _dashboard_protein_dirs(root: Path) -> list[Path]:
    """Return audited protein folders, excluding diagnostics and incomplete cases."""

    if not root.exists():
        return []
    audit_candidates = sorted(root.glob("_structural_metrics*/audit.json"))
    if audit_candidates:
        try:
            audit = json.loads(audit_candidates[-1].read_text(encoding="utf-8"))
            selected = audit.get("selected_proteins", []) if isinstance(audit, dict) else []
            audited = [root / str(name) for name in selected]
            audited = [path for path in audited if path.is_dir()]
            if audited:
                return sorted(audited)
        except (OSError, json.JSONDecodeError):
            pass

    canonical = []
    for path in root.iterdir():
        if not path.is_dir() or path.name.startswith("_"):
            continue
        experiments = [child for child in path.iterdir() if child.is_dir()]
        if any(
            (experiment / "inputs" / "receptor.pdb").is_file()
            and (experiment / "inputs" / "peptide.pdb").is_file()
            and (experiment / "results").is_dir()
            for experiment in experiments
        ):
            canonical.append(path)
    return sorted(canonical)


def _filter_key_suffix(selected_root: Path) -> str:
    raw = str(selected_root).replace("\\", "_").replace("/", "_").replace(":", "_")
    return "".join(char if char.isalnum() or char == "_" else "_" for char in raw)[-120:]


def _ensure_multiselect_default(key: str, options: list[str]) -> None:
    current = st.session_state.get(key)
    if not isinstance(current, list):
        return
    valid = [value for value in current if value in options]
    if valid:
        if valid != current:
            st.session_state[key] = valid
        return
    if options:
        st.session_state[key] = list(options)


def _render_image(path: Path | None) -> None:
    if path is None or not path.exists():
        st.info("Dados insuficientes para gerar este gráfico.")
        return
    st.image(str(path), use_container_width=True)


def _chart_path(name: str) -> Path:
    temp_dir = Path(tempfile.gettempdir()) / "dockingpp_scientific_dashboard"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / name


def _bar_chart(runs: list[dict[str, Any]], key: str, title: str, ylabel: str) -> Path | None:
    grouped = _group_numeric_by_algorithm(runs, key)
    if not grouped:
        return None
    labels = [label for label in _ordered_algorithms(grouped.keys()) if label in grouped]
    means = [sum(grouped[label]) / len(grouped[label]) for label in labels]
    errors = [_ci95_half_width(grouped[label]) for label in labels]

    path = _chart_path(f"{key}_bar.png")
    plt = _matplotlib()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(4.5, 2.35), dpi=140)
    _style_axis(ax)
    colors = ["#67c778", "#ff914d", "#88cff2", "#c3a3ff"]
    ax.bar(labels, means, yerr=errors, color=colors[: len(labels)], capsize=4, edgecolor="#d7dee6")
    ax.set_title(title, color="#17212b", fontsize=9, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, color="#243447", fontsize=8)
    ax.tick_params(axis="x", labelrotation=0, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(pad=0.8)
    fig.savefig(path, facecolor="#ffffff")
    plt.close(fig)
    return path


def _runtime_heuristic_pair_chart(runs: list[dict[str, Any]]) -> Path | None:
    rows = _runtime_heuristic_pair_rows(runs)
    if not rows:
        return None

    path = _chart_path("heuristic_runtime_pair.png")
    plt = _matplotlib()
    if plt is None:
        return None

    labels = [str(row["label"]) for row in rows]
    without_values = [float(row["tempo_sem"]) for row in rows]
    with_values = [float(row["tempo_com"]) for row in rows]
    max_value = max(without_values + with_values) if without_values or with_values else 0.0
    complex_id = _active_complex(runs)
    n_rounds = max(int(row["n_pareado"]) for row in rows)

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=140)
    _style_axis(ax)
    x_positions = list(range(len(rows)))
    width = 0.36
    ax.bar(
        [pos - width / 2.0 for pos in x_positions],
        without_values,
        width=width,
        label="Sem heuristica",
        color="#6b7280",
    )
    ax.bar(
        [pos + width / 2.0 for pos in x_positions],
        with_values,
        width=width,
        label="Com heuristica",
        color="#2563eb",
    )
    for pos, row in zip(x_positions, rows):
        gain = float(row["ganho_tempo_pct"])
        color = "#16a34a" if gain >= 0 else "#dc2626"
        suffix = "mais rapido" if gain >= 0 else "mais lento"
        y = max(float(row["tempo_sem"]), float(row["tempo_com"])) + max(max_value * 0.035, 0.08)
        ax.text(pos, y, f"{abs(gain):.1f}% {suffix}", ha="center", color=color, fontsize=9)

    ax.set_title(f"{complex_id} - tempo medio em {n_rounds} rodadas", color="#111827", fontsize=14, pad=8)
    ax.set_ylabel("Tempo medio de execucao (s)", color="#111827")
    ax.set_xticks(x_positions, labels)
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, max_value * 1.18 if max_value > 0 else 1.0)
    fig.tight_layout(pad=1.1)
    fig.savefig(path, facecolor="#ffffff")
    plt.close(fig)
    return path


def _energy_heuristic_pair_chart(runs: list[dict[str, Any]]) -> Path | None:
    rows = _energy_heuristic_pair_rows(runs)
    if not rows:
        return None

    path = _chart_path("heuristic_energy_pair.png")
    plt = _matplotlib()
    if plt is None:
        return None

    labels = [str(row["label"]) for row in rows]
    without_values = [float(row["energia_sem"]) for row in rows]
    with_values = [float(row["energia_com"]) for row in rows]
    all_values = without_values + with_values
    min_value = min(all_values)
    max_value = max(all_values)
    span = max(max_value - min_value, abs(max_value) * 0.08, 1.0)
    complex_id = _active_complex(runs)
    n_rounds = max(int(row["n_pareado"]) for row in rows)

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=140)
    _style_axis(ax)
    x_positions = list(range(len(rows)))
    width = 0.36
    ax.bar(
        [pos - width / 2.0 for pos in x_positions],
        without_values,
        width=width,
        label="Sem heuristica",
        color="#6b7280",
    )
    ax.bar(
        [pos + width / 2.0 for pos in x_positions],
        with_values,
        width=width,
        label="Com heuristica",
        color="#2563eb",
    )
    for pos, row in zip(x_positions, rows):
        delta = float(row["delta_energia"])
        color = "#16a34a" if delta <= 0 else "#dc2626"
        y = max(float(row["energia_sem"]), float(row["energia_com"])) + span * 0.045
        ax.text(pos, y, f"Delta {delta:.2f}", ha="center", color=color, fontsize=9)

    ax.set_title(
        f"{complex_id} - energia final media em {n_rounds} rodadas (menor e melhor)",
        color="#111827",
        fontsize=14,
        pad=8,
    )
    ax.set_ylabel("Energia final Evina media", color="#111827")
    ax.set_xticks(x_positions, labels)
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(min(0.0, min_value - span * 0.12), max(0.0, max_value + span * 0.18))
    fig.tight_layout(pad=1.1)
    fig.savefig(path, facecolor="#ffffff")
    plt.close(fig)
    return path


def _runtime_heuristic_pair_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for run in runs:
        algorithm = _algorithm_label(run.get("algorithm"))
        mode = str(run.get("mode") or "unknown")
        arm = _heuristic_arm_key(_heuristic_label(run))
        if arm is None:
            continue
        grouped.setdefault((algorithm, mode), {}).setdefault(arm, []).append(run)

    modes = {mode for _, mode in grouped}
    rows: list[dict[str, Any]] = []
    for (algorithm, mode), arms in sorted(
        grouped.items(),
        key=lambda item: (_runtime_chart_algorithm_order_key(item[0][0]), item[0][1]),
    ):
        without_runs = arms.get("sem", [])
        with_runs = arms.get("com", [])
        if not without_runs or not with_runs:
            continue
        pairs = _paired_by_round(without_runs, with_runs)
        if pairs:
            without_times = [float(pair[0]["runtime_sec"]) for pair in pairs]
            with_times = [float(pair[1]["runtime_sec"]) for pair in pairs]
            speedups = [
                ((sem - com) / sem) * 100.0
                for sem, com in zip(without_times, with_times)
                if sem > 0
            ]
            n_paired = len(pairs)
        else:
            without_times = _numeric_list([run.get("runtime_sec") for run in without_runs])
            with_times = _numeric_list([run.get("runtime_sec") for run in with_runs])
            speedups = []
            n_paired = min(len(without_times), len(with_times))
        without_mean = _mean_float(without_times)
        with_mean = _mean_float(with_times)
        if without_mean is None or with_mean is None:
            continue
        if speedups:
            gain_pct = float(sum(speedups) / len(speedups))
        else:
            gain_pct = ((without_mean - with_mean) / without_mean) * 100.0 if without_mean > 0 else 0.0
        label = algorithm if len(modes) <= 1 else f"{algorithm}\n{mode}"
        rows.append(
            {
                "label": label,
                "algoritmo": algorithm,
                "modo": mode,
                "tempo_sem": float(without_mean),
                "tempo_com": float(with_mean),
                "ganho_tempo_pct": float(gain_pct),
                "n_pareado": int(n_paired or max(len(without_times), len(with_times))),
            }
        )
    rendered_modes = {str(row["modo"]) for row in rows}
    if len(rendered_modes) <= 1:
        for row in rows:
            row["label"] = row["algoritmo"]
    return rows


def _energy_heuristic_pair_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for run in runs:
        algorithm = _algorithm_label(run.get("algorithm"))
        mode = str(run.get("mode") or "unknown")
        arm = _heuristic_arm_key(_heuristic_label(run))
        if arm is None:
            continue
        grouped.setdefault((algorithm, mode), {}).setdefault(arm, []).append(run)

    modes = {mode for _, mode in grouped}
    rows: list[dict[str, Any]] = []
    for (algorithm, mode), arms in sorted(
        grouped.items(),
        key=lambda item: (_runtime_chart_algorithm_order_key(item[0][0]), item[0][1]),
    ):
        without_runs = arms.get("sem", [])
        with_runs = arms.get("com", [])
        if not without_runs or not with_runs:
            continue
        pairs = _paired_metric_values_by_round(without_runs, with_runs, _energy_value_for_chart)
        if pairs:
            without_energies = [float(pair[0]) for pair in pairs]
            with_energies = [float(pair[1]) for pair in pairs]
            n_paired = len(pairs)
        else:
            without_energies = _metric_values(without_runs, _energy_value_for_chart)
            with_energies = _metric_values(with_runs, _energy_value_for_chart)
            n_paired = min(len(without_energies), len(with_energies))
        without_mean = _mean_float(without_energies)
        with_mean = _mean_float(with_energies)
        if without_mean is None or with_mean is None:
            continue
        label = algorithm if len(modes) <= 1 else f"{algorithm}\n{mode}"
        rows.append(
            {
                "label": label,
                "algoritmo": algorithm,
                "modo": mode,
                "energia_sem": float(without_mean),
                "energia_com": float(with_mean),
                "delta_energia": float(with_mean - without_mean),
                "n_pareado": int(n_paired or max(len(without_energies), len(with_energies))),
            }
        )
    rendered_modes = {str(row["modo"]) for row in rows}
    if len(rendered_modes) <= 1:
        for row in rows:
            row["label"] = row["algoritmo"]
    return rows


def _structural_source_key_from_run_path(
    *, protein: str, experiment: str, path_hint: str
) -> str | None:
    """Map current and legacy result paths to the batch evaluator source key."""

    parts = tuple(
        part
        for part in str(path_hint).replace("\\", "/").split("/")
        if part
    )
    if len(parts) < 5 or parts[0] != "results":
        return None
    first, second, heuristic, algorithm = parts[1:5]
    if first.lower().startswith("rodada_"):
        round_name, mode = first, second
    elif second.lower().startswith("rodada_"):
        mode, round_name = first, second
    else:
        return None
    if mode.lower() not in {"full", "reduced"}:
        return None
    return "/".join(
        (
            str(protein),
            str(experiment),
            round_name,
            mode.lower(),
            heuristic,
            algorithm,
        )
    )


def _load_structural_evidence(
    runs_root: Path,
    selected_root: Path,
    filtered_runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Load audited structural rows corresponding to the current filters."""

    candidates = sorted(runs_root.glob("_structural_metrics*/structural_metrics.csv"))
    if not candidates:
        return _structural_rows_from_summaries(selected_root, filtered_runs)
    csv_path = candidates[-1]
    try:
        frame = pd.read_csv(csv_path)
    except Exception:  # noqa: BLE001
        return _structural_rows_from_summaries(selected_root, filtered_runs)

    try:
        relative_selected = selected_root.resolve().relative_to(runs_root.resolve())
    except (OSError, ValueError):
        return [], None
    if len(relative_selected.parts) < 2:
        return [], None
    protein, experiment = relative_selected.parts[:2]
    frame = frame[
        (frame["protein"].astype(str) == str(protein))
        & (frame["experiment"].astype(str) == str(experiment))
    ]

    allowed_keys: set[str] = set()
    for run in filtered_runs:
        path_hint = str(run.get("run_dir_relative") or "").replace("\\", "/")
        source_key = _structural_source_key_from_run_path(
            protein=str(protein),
            experiment=str(experiment),
            path_hint=path_hint,
        )
        if source_key is not None:
            allowed_keys.add(source_key)
    if allowed_keys:
        frame = frame[frame["source_key"].astype(str).isin(allowed_keys)]

    audit_path = csv_path.parent / "audit.json"
    audit = None
    if audit_path.is_file():
        try:
            loaded = json.loads(audit_path.read_text(encoding="utf-8"))
            audit = loaded if isinstance(loaded, dict) else None
        except (OSError, json.JSONDecodeError):
            audit = None
    status_column = next(
        (
            column
            for column in ("structural_metrics_status", "status")
            if column in frame.columns
        ),
        None,
    )
    not_applicable_rows: list[dict[str, Any]] = []
    if status_column is not None:
        status = frame[status_column].astype(str).str.strip().str.lower()
        not_applicable_mask = status == "not_applicable"
        not_applicable_rows = frame[not_applicable_mask].to_dict(orient="records")
        frame = frame[~not_applicable_mask]

    rows = frame.to_dict(orient="records")
    if not_applicable_rows:
        audit = dict(audit or {})
        audit["not_applicable_rows"] = len(not_applicable_rows)
        audit["not_applicable"] = not_applicable_rows
        audit["successful_rows"] = len(rows)
        audit["passed"] = (
            bool(rows)
            and not bool(audit.get("failed_rows"))
            and not not_applicable_rows
        )
    if rows:
        return rows, audit
    if not_applicable_rows:
        return [], audit
    return _structural_rows_from_summaries(selected_root, filtered_runs)


def _structural_rows_from_summaries(
    selected_root: Path,
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build structural evidence directly from automatic run summaries."""

    rows: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    not_applicable: list[dict[str, Any]] = []
    for run in runs:
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        status = str(summary.get("structural_metrics_status") or "").strip().lower()
        if status == "not_applicable":
            not_applicable.append(
                {
                    "run_dir": run.get("run_dir"),
                    "status": "not_applicable",
                    "error_code": summary.get("structural_metrics_error_code"),
                    "partner_kind": summary.get(
                        "structural_metrics_partner_kind"
                    ),
                    "reason": summary.get("structural_metrics_error"),
                }
            )
            continue
        dockq = summary.get("dockq")
        if not isinstance(dockq, (int, float)):
            if summary.get("structural_metrics_enabled"):
                unavailable.append(
                    {
                        "run_dir": run.get("run_dir"),
                        "error": summary.get("structural_metrics_error"),
                    }
                )
            continue
        heuristic_arm = _heuristic_arm_key(_heuristic_label(run))
        rows.append(
            {
                "protein": str(
                    run.get("complex_id")
                    or summary.get("complex_id")
                    or selected_root.parent.name
                ),
                "experiment": selected_root.name,
                "round": str(
                    run.get("round_index")
                    or summary.get("experimental_round")
                    or "1"
                ),
                "mode": str(
                    summary.get("search_space_mode")
                    or run.get("search_space_mode")
                    or run.get("mode")
                    or "unknown"
                ),
                "heuristic": (
                    "com_heuristica"
                    if heuristic_arm == "com"
                    else "sem_heuristica"
                ),
                "algorithm": run.get("algorithm"),
                "dockq": float(dockq),
                "dockq_category": summary.get("dockq_category"),
                "fnat": summary.get("fnat"),
                "irmsd": summary.get("irmsd"),
                "lrmsd": summary.get("lrmsd"),
                "runtime_sec": run.get("runtime_sec"),
                "source_key": str(run.get("run_dir_relative") or run.get("run_dir")),
                "structural_metrics_output_path": summary.get(
                    "structural_metrics_output_path"
                ),
                "structural_metrics_source": "summary.json",
                "structural_metrics_status": summary.get(
                    "structural_metrics_status",
                    "available",
                ),
            }
        )
    audit = {
        "source": "automatic_summary_metrics",
        "successful_rows": len(rows),
        "failed_rows": len(unavailable),
        "not_applicable_rows": len(not_applicable),
        "errors": unavailable,
        "not_applicable": not_applicable,
        "passed": bool(rows) and not unavailable and not not_applicable,
    }
    return rows, audit


def _structural_audit_count(
    audit: dict[str, Any] | None,
    key: str,
) -> int:
    if not audit:
        return 0
    value = audit.get(key, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not pd.isna(value):
        return max(int(value), 0)
    return 0


def _structural_audit_label(audit: dict[str, Any] | None) -> str:
    if audit is None:
        return "Pendente"
    successful = _structural_audit_count(audit, "successful_rows")
    failed = _structural_audit_count(audit, "failed_rows")
    not_applicable = _structural_audit_count(audit, "not_applicable_rows")
    if not_applicable > 0 and successful == 0 and failed == 0:
        return "Não aplicável"
    if not_applicable > 0 and failed == 0:
        return "Parcial"
    if bool(audit.get("passed")):
        return "Aprovada"
    return "Falhou"


def _structural_numeric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if not isinstance(value, (int, float)) or pd.isna(value):
        return None
    return float(value)


def _structural_winner_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if _structural_numeric(row, "dockq") is None:
            continue
        key = tuple(
            str(row.get(field) or "")
            for field in ("protein", "experiment", "round", "mode", "heuristic")
        )
        grouped.setdefault(key, []).append(row)

    wins: dict[str, int] = {}
    for group in grouped.values():
        algorithms = {_algorithm_label(row.get("algorithm")) for row in group}
        if not {"ABC+AG+VGOS", "AG", "PSO"}.issubset(algorithms):
            continue
        winner = max(group, key=lambda row: _structural_numeric(row, "dockq") or -1.0)
        label = _algorithm_label(winner.get("algorithm"))
        wins[label] = wins.get(label, 0) + 1
    return wins


def _structural_hybrid_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_algorithm_label(row.get("algorithm")), []).append(row)
    wins = _structural_winner_counts(rows)
    output = []
    for algorithm in _ordered_algorithms(grouped):
        group = grouped.get(algorithm, [])
        dockq = [value for row in group if (value := _structural_numeric(row, "dockq")) is not None]
        if not dockq:
            continue
        lrmsd = [value for row in group if (value := _structural_numeric(row, "lrmsd")) is not None]
        irmsd = [value for row in group if (value := _structural_numeric(row, "irmsd")) is not None]
        fnat = [value for row in group if (value := _structural_numeric(row, "fnat")) is not None]
        output.append(
            {
                "algoritmo": algorithm,
                "n": len(dockq),
                "DockQ medio": round(_mean_float(dockq) or 0.0, 4),
                "DockQ DP": round(_std(dockq), 4),
                "DockQ IC95": _fmt_ci95(dockq),
                "L-RMSD medio": round(_mean_float(lrmsd) or 0.0, 4),
                "iRMSD medio": round(_mean_float(irmsd) or 0.0, 4),
                "Fnat medio": round(_mean_float(fnat) or 0.0, 4),
                "CAPRI media/alta (%)": round(100.0 * sum(value >= 0.49 for value in dockq) / len(dockq), 1),
                "vitorias pareadas": wins.get(algorithm, 0),
            }
        )
    return pd.DataFrame(output)


def _structural_hybrid_summary(rows: list[dict[str, Any]]) -> dict[str, str]:
    table = _structural_hybrid_table(rows)
    if table.empty:
        return {
            "best_algorithm": "NA",
            "hybrid_vs_pso": "NA",
            "capri_success": "NA",
            "hybrid_status": "inconclusivo",
        }
    best = table.sort_values("DockQ medio", ascending=False).iloc[0]
    values = {
        str(row["algoritmo"]): float(row["DockQ medio"])
        for _, row in table.iterrows()
    }
    difference = values.get("ABC+AG+VGOS", float("nan")) - values.get("PSO", float("nan"))
    comparison = "NA"
    if math.isfinite(difference):
        comparison = f"{'Sim' if difference > 0 else 'Nao'} ({difference:+.3f})"
    successes = sum(
        1 for row in rows if (_structural_numeric(row, "dockq") or 0.0) >= 0.49
    )
    return {
        "best_algorithm": str(best["algoritmo"]),
        "hybrid_vs_pso": comparison,
        "capri_success": f"{successes}/{len(rows)}",
        "hybrid_status": (
            "sustentado" if str(best["algoritmo"]) == "ABC+AG+VGOS" else "nao_sustentado"
        ),
    }


def _structural_heuristic_pairs(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = tuple(
            str(row.get(field) or "")
            for field in ("protein", "experiment", "round", "mode", "algorithm")
        )
        grouped.setdefault(key, {})[str(row.get("heuristic") or "")] = row
    pairs = []
    for arms in grouped.values():
        without = arms.get("sem_heuristica")
        with_heuristic = arms.get("com_heuristica")
        if without is not None and with_heuristic is not None:
            pairs.append((without, with_heuristic))
    return pairs


def _structural_heuristic_summary(rows: list[dict[str, Any]]) -> dict[str, str]:
    pairs = _structural_heuristic_pairs(rows)
    paired_times = [
        (_structural_numeric(without, "runtime_sec"), _structural_numeric(with_row, "runtime_sec"))
        for without, with_row in pairs
    ]
    paired_times = [pair for pair in paired_times if pair[0] is not None and pair[1] is not None]
    if not paired_times:
        return {"speedup_label": "NA"}
    mean_without = _mean_float([float(pair[0]) for pair in paired_times]) or 0.0
    mean_with = _mean_float([float(pair[1]) for pair in paired_times]) or 0.0
    speedup = 100.0 * (mean_without - mean_with) / mean_without if mean_without else 0.0
    return {"speedup_label": f"{speedup:.1f}%"}


def _structural_heuristic_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for without, with_row in _structural_heuristic_pairs(rows):
        grouped.setdefault(_algorithm_label(without.get("algorithm")), []).append((without, with_row))
    output = []
    for algorithm in _ordered_algorithms(grouped):
        pairs = grouped.get(algorithm, [])
        if not pairs:
            continue
        complete_pairs = [
            (
                _structural_numeric(pair[0], "runtime_sec"),
                _structural_numeric(pair[1], "runtime_sec"),
                _structural_numeric(pair[0], "dockq"),
                _structural_numeric(pair[1], "dockq"),
                pair,
            )
            for pair in pairs
        ]
        complete_pairs = [item for item in complete_pairs if all(value is not None for value in item[:4])]
        if not complete_pairs:
            continue
        mean_without = _mean_float([float(item[0]) for item in complete_pairs]) or 0.0
        mean_with = _mean_float([float(item[1]) for item in complete_pairs]) or 0.0
        dockq_delta = [float(item[3]) - float(item[2]) for item in complete_pairs]
        output.append(
            {
                "algoritmo": algorithm,
                "n pareado": len(complete_pairs),
                "tempo sem (s)": round(mean_without, 4),
                "tempo com (s)": round(mean_with, 4),
                "reducao tempo (%)": round(100.0 * (mean_without - mean_with) / mean_without, 2),
                "delta DockQ": round(_mean_float(dockq_delta) or 0.0, 4),
                "poses identicas": sum(
                    item[4][0].get("pose_sha256") == item[4][1].get("pose_sha256")
                    for item in complete_pairs
                ),
            }
        )
    return pd.DataFrame(output)


def _structural_statistics_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (_algorithm_label(row.get("algorithm")), str(row.get("heuristic") or "")),
            [],
        ).append(row)
    output = []
    for (algorithm, heuristic), group in sorted(
        grouped.items(), key=lambda item: (_algorithm_order_key(item[0][0]), item[0][1])
    ):
        values = {
            key: [
                value for row in group if (value := _structural_numeric(row, key)) is not None
            ]
            for key in ("dockq", "lrmsd", "irmsd", "fnat")
        }
        output.append(
            {
                "algoritmo": algorithm,
                "heuristica": "Com" if heuristic == "com_heuristica" else "Sem",
                "n": len(group),
                "DockQ media": round(_mean_float(values["dockq"]) or 0.0, 4),
                "DockQ DP": round(_std(values["dockq"]), 4),
                "DockQ IC95": _fmt_ci95(values["dockq"]),
                "L-RMSD media": round(_mean_float(values["lrmsd"]) or 0.0, 4),
                "iRMSD media": round(_mean_float(values["irmsd"]) or 0.0, 4),
                "Fnat media": round(_mean_float(values["fnat"]) or 0.0, 4),
            }
        )
    return pd.DataFrame(output)


def _structural_algorithm_chart(rows: list[dict[str, Any]]) -> Path | None:
    table = _structural_hybrid_table(rows)
    if table.empty:
        return None
    path = _chart_path("structural_dockq_by_algorithm.png")
    plt = _matplotlib()
    if plt is None:
        return None
    labels = table["algoritmo"].astype(str).tolist()
    values = table["DockQ medio"].astype(float).tolist()
    errors = []
    for label in labels:
        group = [
            value
            for row in rows
            if _algorithm_label(row.get("algorithm")) == label
            and (value := _structural_numeric(row, "dockq")) is not None
        ]
        errors.append(_ci95_half_width(group))
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=140)
    _style_axis(ax)
    colors = ["#1976d2" if label == "ABC+AG+VGOS" else "#6b7280" for label in labels]
    bars = ax.bar(labels, values, yerr=errors, capsize=4, color=colors)
    _label_bars(ax, bars, values, "{:.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_title("DockQ medio por algoritmo", fontsize=10, fontweight="bold")
    ax.set_ylabel("DockQ (maior e melhor)", fontsize=8)
    fig.tight_layout(pad=0.8)
    fig.savefig(path, facecolor="#ffffff")
    plt.close(fig)
    return path


def _structural_wins_chart(rows: list[dict[str, Any]]) -> Path | None:
    wins = _structural_winner_counts(rows)
    if not wins:
        return None
    path = _chart_path("structural_dockq_wins.png")
    plt = _matplotlib()
    if plt is None:
        return None
    labels = [label for label in _ordered_algorithms(wins) if label in wins]
    values = [wins[label] for label in labels]
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=140)
    _style_axis(ax)
    bars = ax.bar(labels, values, color=["#1976d2" if label == "ABC+AG+VGOS" else "#6b7280" for label in labels])
    _label_bars(ax, bars, values, "{:.0f}")
    ax.set_title("Vitorias pareadas por DockQ", fontsize=10, fontweight="bold")
    ax.set_ylabel("Vitorias", fontsize=8)
    fig.tight_layout(pad=0.8)
    fig.savefig(path, facecolor="#ffffff")
    plt.close(fig)
    return path


def _structural_heuristic_chart(rows: list[dict[str, Any]]) -> Path | None:
    table = _structural_heuristic_table(rows)
    if table.empty:
        return None
    path = _chart_path("structural_runtime_heuristic.png")
    plt = _matplotlib()
    if plt is None:
        return None
    labels = table["algoritmo"].astype(str).tolist()
    without = table["tempo sem (s)"].astype(float).tolist()
    with_values = table["tempo com (s)"].astype(float).tolist()
    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=140)
    _style_axis(ax)
    ax.bar([value - width / 2 for value in x], without, width, label="Sem heuristica", color="#6b7280")
    ax.bar([value + width / 2 for value in x], with_values, width, label="Com heuristica", color="#1976d2")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Tempo (s)", fontsize=8)
    ax.set_title("Tempo pareado por condicao", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout(pad=0.8)
    fig.savefig(path, facecolor="#ffffff")
    plt.close(fig)
    return path


def _final_rescore_value(run: dict[str, Any]) -> float | None:
    score = run.get("final_rescore_score")
    if isinstance(score, (int, float)):
        return float(score)
    energy = run.get("energy_evina")
    if isinstance(energy, (int, float)):
        return -float(energy)
    return None


def _hybrid_relevance_summary(runs: list[dict[str, Any]]) -> dict[str, str]:
    table = _hybrid_relevance_table(runs)
    wins = _winner_counts_by_algorithm(runs)
    total_wins = sum(wins.values())
    hybrid_wins = wins.get("ABC+AG+VGOS", 0)
    best_algorithm = "NA"
    if not table.empty:
        best_row = table.sort_values("final_rescore_medio", ascending=False).iloc[0]
        best_algorithm = str(best_row.get("algoritmo") or "NA")
    return {
        "hybrid_wins_label": f"{hybrid_wins}/{total_wins}" if total_wins else "NA",
        "best_algorithm": best_algorithm,
    }


def _overall_heuristic_effect_summary(runs: list[dict[str, Any]]) -> dict[str, str]:
    rows = _heuristic_effect_rows(runs)
    if not rows:
        return {"faster_label": "NA", "speedup_label": "NA"}
    faster = sum(1 for row in rows if row["delta_tempo_s"] < 0)
    total = len(rows)
    speedups = [row["reducao_tempo_pct"] for row in rows if isinstance(row["reducao_tempo_pct"], (int, float))]
    mean_speedup = _mean_float([float(value) for value in speedups])
    return {
        "faster_label": f"{faster}/{total}",
        "speedup_label": "NA" if mean_speedup is None else f"{mean_speedup:.1f}%",
    }


def _hybrid_relevance_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[str, list[float]] = {}
    runtime_grouped: dict[str, list[float]] = {}
    for run in runs:
        algorithm = _algorithm_label(run.get("algorithm"))
        score = _final_rescore_value(run)
        runtime = run.get("runtime_sec")
        if score is not None:
            grouped.setdefault(algorithm, []).append(score)
        if isinstance(runtime, (int, float)):
            runtime_grouped.setdefault(algorithm, []).append(float(runtime))

    wins = _winner_counts_by_algorithm(runs)
    rows = []
    for algorithm in _ordered_algorithms(grouped.keys()):
        scores = grouped.get(algorithm, [])
        runtimes = runtime_grouped.get(algorithm, [])
        if not scores:
            continue
        rows.append(
            {
                "algoritmo": algorithm,
                "n": len(scores),
                "final_rescore_medio": _round_or_na(_mean_float(scores), 4),
                "final_rescore_std": _round_or_na(_std(scores), 4),
                "final_rescore_ic95": _fmt_ci95(scores),
                "tempo_medio_s": _round_or_na(_mean_float(runtimes), 4),
                "vitorias_pareadas": wins.get(algorithm, 0),
            }
        )
    return pd.DataFrame(rows)


def _winner_counts_by_algorithm(runs: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for run in runs:
        score = _final_rescore_value(run)
        round_index = run.get("round_index")
        if score is None or round_index is None:
            continue
        key = (
            str(run.get("complex_id") or "NA"),
            str(run.get("mode") or "unknown"),
            str(round_index),
            _heuristic_label(run),
        )
        grouped.setdefault(key, []).append(run)

    wins: dict[str, int] = {}
    for group in grouped.values():
        present = {_algorithm_label(run.get("algorithm")) for run in group}
        if not {"ABC+AG+VGOS", "AG", "PSO"}.issubset(present):
            continue
        winner = max(group, key=lambda run: _final_rescore_value(run) or float("-inf"))
        algorithm = _algorithm_label(winner.get("algorithm"))
        wins[algorithm] = wins.get(algorithm, 0) + 1
    return wins


def _algorithm_final_rescore_chart(runs: list[dict[str, Any]]) -> Path | None:
    grouped: dict[str, list[float]] = {}
    for run in runs:
        score = _final_rescore_value(run)
        if score is not None:
            grouped.setdefault(_algorithm_label(run.get("algorithm")), []).append(score)
    if not grouped:
        return None
    labels = [label for label in _ordered_algorithms(grouped.keys()) if label in grouped]
    values = [_mean_float(grouped[label]) or 0.0 for label in labels]
    errors = [_ci95_half_width(grouped[label]) for label in labels]
    path = _chart_path("hybrid_final_rescore.png")
    plt = _matplotlib()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=140)
    _style_axis(ax)
    colors = ["#2563eb" if label == "ABC+AG+VGOS" else "#94a3b8" for label in labels]
    bars = ax.bar(labels, values, yerr=errors, color=colors, capsize=4, edgecolor="#d7dee6")
    _label_bars(ax, bars, values, "{:.2f}")
    ax.set_title("final_rescore medio por algoritmo", color="#17212b", fontsize=10, fontweight="bold")
    ax.set_ylabel("final_rescore (maior e melhor)", color="#243447", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(pad=0.8)
    fig.savefig(path, facecolor="#ffffff")
    plt.close(fig)
    return path


def _algorithm_wins_chart(runs: list[dict[str, Any]]) -> Path | None:
    wins = _winner_counts_by_algorithm(runs)
    if not wins:
        return None
    labels = [label for label in _ordered_algorithms(wins.keys()) if label in wins]
    values = [wins[label] for label in labels]
    path = _chart_path("hybrid_paired_wins.png")
    plt = _matplotlib()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=140)
    _style_axis(ax)
    colors = ["#2563eb" if label == "ABC+AG+VGOS" else "#94a3b8" for label in labels]
    bars = ax.bar(labels, values, color=colors, edgecolor="#d7dee6")
    _label_bars(ax, bars, values, "{:.0f}")
    ax.set_title("vitorias pareadas", color="#17212b", fontsize=10, fontweight="bold")
    ax.set_ylabel("numero de vitorias", color="#243447", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(pad=0.8)
    fig.savefig(path, facecolor="#ffffff")
    plt.close(fig)
    return path


def _heuristic_effect_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for run in runs:
        algorithm = _algorithm_label(run.get("algorithm"))
        mode = str(run.get("mode") or "unknown")
        arm = _heuristic_arm_key(_heuristic_label(run))
        if arm is None:
            continue
        grouped.setdefault((algorithm, mode), {}).setdefault(arm, []).append(run)

    rows: list[dict[str, Any]] = []
    for (algorithm, mode), arms in sorted(
        grouped.items(),
        key=lambda item: (_runtime_chart_algorithm_order_key(item[0][0]), item[0][1]),
    ):
        without_runs = arms.get("sem", [])
        with_runs = arms.get("com", [])
        if not without_runs or not with_runs:
            continue
        pairs = _paired_by_round(without_runs, with_runs)
        if not pairs:
            continue
        runtime_without = [float(pair[0]["runtime_sec"]) for pair in pairs]
        runtime_with = [float(pair[1]["runtime_sec"]) for pair in pairs]
        score_pairs = [
            (_final_rescore_value(pair[0]), _final_rescore_value(pair[1]))
            for pair in pairs
        ]
        valid_score_pairs = [(sem, com) for sem, com in score_pairs if sem is not None and com is not None]
        mean_runtime_without = _mean_float(runtime_without)
        mean_runtime_with = _mean_float(runtime_with)
        mean_score_without = _mean_float([float(pair[0]) for pair in valid_score_pairs])
        mean_score_with = _mean_float([float(pair[1]) for pair in valid_score_pairs])
        if mean_runtime_without is None or mean_runtime_with is None:
            continue
        delta_time = mean_runtime_with - mean_runtime_without
        speedup = ((mean_runtime_without - mean_runtime_with) / mean_runtime_without) * 100.0 if mean_runtime_without > 0 else 0.0
        delta_score = None if mean_score_without is None or mean_score_with is None else mean_score_with - mean_score_without
        rows.append(
            {
                "algoritmo": algorithm,
                "modo": mode,
                "n_pareado": len(pairs),
                "tempo_sem_s": mean_runtime_without,
                "tempo_com_s": mean_runtime_with,
                "delta_tempo_s": delta_time,
                "reducao_tempo_pct": speedup,
                "final_sem": mean_score_without,
                "final_com": mean_score_with,
                "delta_final_rescore": delta_score,
            }
        )
    return rows


def _heuristic_effect_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in _heuristic_effect_rows(runs):
        rows.append(
            {
                "algoritmo": row["algoritmo"],
                "modo": row["modo"],
                "n_pareado": row["n_pareado"],
                "tempo_sem_s": round(float(row["tempo_sem_s"]), 4),
                "tempo_com_s": round(float(row["tempo_com_s"]), 4),
                "reducao_tempo_pct": round(float(row["reducao_tempo_pct"]), 2),
                "final_sem": _round_or_na(row["final_sem"], 4),
                "final_com": _round_or_na(row["final_com"], 4),
                "delta_final_rescore": _round_or_na(row["delta_final_rescore"], 4),
            }
        )
    return pd.DataFrame(rows)


def _paired_metric_values_by_round(
    without_runs: list[dict[str, Any]],
    with_runs: list[dict[str, Any]],
    value_getter: Any,
) -> list[tuple[float, float]]:
    without_by_round = {}
    for run in without_runs:
        round_index = run.get("round_index")
        value = value_getter(run)
        if round_index is not None and value is not None:
            without_by_round[str(round_index)] = float(value)

    with_by_round = {}
    for run in with_runs:
        round_index = run.get("round_index")
        value = value_getter(run)
        if round_index is not None and value is not None:
            with_by_round[str(round_index)] = float(value)

    common_rounds = sorted(set(without_by_round) & set(with_by_round), key=_round_sort_key)
    return [(without_by_round[round_id], with_by_round[round_id]) for round_id in common_rounds]


def _metric_values(runs: list[dict[str, Any]], value_getter: Any) -> list[float]:
    values = []
    for run in runs:
        value = value_getter(run)
        if value is not None:
            values.append(float(value))
    return values


def _energy_value_for_chart(run: dict[str, Any]) -> float | None:
    score = run.get("final_rescore_score")
    if isinstance(score, (int, float)):
        return float(score)
    energy = run.get("energy_evina")
    if isinstance(energy, (int, float)):
        return abs(float(energy))
    return None


def _paired_by_round(
    without_runs: list[dict[str, Any]],
    with_runs: list[dict[str, Any]],
    key: str = "runtime_sec",
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    without_by_round = {
        str(run.get("round_index")): run
        for run in without_runs
        if run.get("round_index") is not None and isinstance(run.get(key), (int, float))
    }
    with_by_round = {
        str(run.get("round_index")): run
        for run in with_runs
        if run.get("round_index") is not None and isinstance(run.get(key), (int, float))
    }
    common_rounds = sorted(set(without_by_round) & set(with_by_round), key=_round_sort_key)
    return [(without_by_round[round_id], with_by_round[round_id]) for round_id in common_rounds]


def _heuristic_arm_key(label: Any) -> str | None:
    lowered = str(label or "").strip().lower()
    if lowered.startswith("sem"):
        return "sem"
    if lowered.startswith("com"):
        return "com"
    return None


def _round_sort_key(value: Any) -> tuple[int, str]:
    try:
        return int(value), str(value)
    except (TypeError, ValueError):
        return 10**9, str(value)


def _runtime_chart_algorithm_order_key(label: str) -> tuple[int, str]:
    order = {"ABC+AG+VGOS": 0, "AG": 1, "PSO": 2}
    return order.get(label, 99), label


def _boxplot(runs: list[dict[str, Any]]) -> Path | None:
    grouped = _group_numeric_by_algorithm(runs, "energy_evina")
    grouped = {key: value for key, value in grouped.items() if value}
    if not grouped:
        return None
    path = _chart_path("energy_distribution.png")
    plt = _matplotlib()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 2.05), dpi=140)
    _style_axis(ax)
    labels = [label for label in _ordered_algorithms(grouped.keys()) if label in grouped]
    data = [grouped[label] for label in labels]
    box = ax.boxplot(data, patch_artist=True, labels=labels, widths=0.45)
    for patch, color in zip(box["boxes"], ["#67c778", "#ff914d", "#88cff2", "#c3a3ff"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for element in ("whiskers", "caps", "medians"):
        for item in box[element]:
            item.set_color("#243447")
    ax.set_title("DISTRIBUICAO DA ENERGIA FINAL", color="#17212b", fontsize=9, fontweight="bold")
    ax.set_ylabel("E_vina", color="#243447", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(pad=0.8)
    fig.savefig(path, facecolor="#ffffff")
    plt.close(fig)
    return path


def _convergence_chart(runs: list[dict[str, Any]]) -> Path | None:
    run = next((item for item in runs if item.get("series")), None)
    if run is None:
        return None
    series = run.get("series") or {}
    xs = _numeric_list(series.get("iter"))
    ys = _numeric_list(series.get("best_expensive")) or _numeric_list(series.get("best_cheap"))
    if not xs or not ys:
        return None
    count = min(len(xs), len(ys))
    path = _chart_path("convergence_dark.png")
    plt = _matplotlib()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(3.2, 2.2), dpi=140)
    _style_axis(ax)
    ax.plot(xs[:count], ys[:count], color="#1f77b4", linewidth=1.6)
    ax.set_title("CONVERGENCIA DA ENERGIA", color="#17212b", fontsize=8, fontweight="bold")
    ax.set_xlabel("Geracoes", color="#243447", fontsize=7)
    ax.set_ylabel("Energia", color="#243447", fontsize=7)
    ax.tick_params(axis="both", labelsize=7)
    fig.tight_layout(pad=0.8)
    fig.savefig(path, facecolor="#ffffff")
    plt.close(fig)
    return path


def _radar_chart(runs: list[dict[str, Any]]) -> Path | None:
    values = _radar_values(runs)
    path = _chart_path("radar_dark.png")
    plt = _matplotlib()
    if plt is None:
        return None
    import math

    labels = ["P_col", "P_exp", "B_cont", "P_sxp"]
    angles = [idx / float(len(labels)) * 2 * math.pi for idx in range(len(labels))]
    plot_values = values + values[:1]
    plot_angles = angles + angles[:1]

    fig = plt.figure(figsize=(3.2, 2.2), dpi=140)
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#ffffff")
    ax.plot(plot_angles, plot_values, color="#1f77b4", linewidth=1.4)
    ax.fill(plot_angles, plot_values, color="#1f77b4", alpha=0.18)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, color="#243447", fontsize=7)
    ax.set_yticklabels([])
    ax.grid(color="#d7dee6", alpha=0.9)
    ax.spines["polar"].set_color("#cbd5df")
    ax.set_title("PENALIDADE/BONUS", color="#17212b", fontsize=8, fontweight="bold", pad=9)
    fig.tight_layout(pad=0.6)
    fig.savefig(path, facecolor="#ffffff")
    plt.close(fig)
    return path


def _donut_chart(runs: list[dict[str, Any]]) -> Path | None:
    efficiency = _kdtree_efficiency(runs)
    if efficiency is None:
        efficiency = 0.0
    path = _chart_path("kdtree_donut.png")
    plt = _matplotlib()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(2.6, 2.2), dpi=140)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.pie(
        [efficiency, max(0.0, 100.0 - efficiency)],
        colors=["#4aa3df", "#e2e8f0"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.28, "edgecolor": "#ffffff"},
    )
    label = f"{efficiency:.0f}%" if efficiency else "N/A"
    ax.text(0, 0, label, color="#17212b", ha="center", va="center", fontsize=15, fontweight="bold")
    ax.set_title("EFICIENCIA KD-tree", color="#17212b", fontsize=8, fontweight="bold")
    fig.tight_layout(pad=0.6)
    fig.savefig(path, facecolor="#ffffff")
    plt.close(fig)
    return path


def _matplotlib() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    return plt


def _style_axis(ax: Any) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, axis="y", color="#d7dee6", alpha=0.75, linewidth=0.7)
    ax.spines["bottom"].set_color("#9aa8b5")
    ax.spines["left"].set_color("#9aa8b5")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#243447")


def _label_bars(ax: Any, bars: Any, values: list[float], fmt: str) -> None:
    if not values:
        return
    max_value = max(values)
    offset = max(abs(max_value) * 0.025, 0.05)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#243447",
        )


def _group_numeric_by_algorithm(runs: list[dict[str, Any]], key: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for run in runs:
        algorithm = _algorithm_label(run.get("algorithm"))
        value = run.get(key)
        if isinstance(value, (int, float)):
            grouped.setdefault(algorithm, []).append(float(value))
    return grouped


def _heuristic_summary_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    grouped_energy = _group_numeric_by_algorithm(runs, "energy_evina")
    grouped_time = _group_numeric_by_algorithm(runs, "runtime_sec")
    rows = []
    for algorithm in _ordered_algorithms(set(grouped_energy) | set(grouped_time)):
        energies = grouped_energy.get(algorithm, [])
        runtimes = grouped_time.get(algorithm, [])
        rows.append(
            {
                "algoritmo": algorithm,
                "evina_media": round(sum(energies) / len(energies), 4) if energies else "Sem dados",
                "evina_ic95": _fmt_ci95(energies) if energies else "Sem dados",
                "tempo_medio_s": round(sum(runtimes) / len(runtimes), 4) if runtimes else "Sem dados",
                "tempo_ic95_s": _fmt_ci95(runtimes) if runtimes else "Sem dados",
                "n": max(len(energies), len(runtimes)),
            }
        )
    return pd.DataFrame(rows)


def _paired_heuristic_effect_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for run in runs:
        key = (_algorithm_label(run.get("algorithm")), str(run.get("mode") or "unknown"))
        grouped.setdefault(key, {}).setdefault(_heuristic_label(run), []).append(run)

    rows = []
    for (algorithm, mode), arms in sorted(grouped.items(), key=lambda item: (_algorithm_order_key(item[0][0]), item[0][1])):
        without_runs = arms.get("Sem heurÃ­stica") or arms.get("Sem heurística") or []
        with_runs = arms.get("Com heurÃ­stica") or arms.get("Com heurística") or []
        if not without_runs or not with_runs:
            continue
        without_score = _mean_float(_numeric_list([run.get("energy_evina") for run in without_runs]))
        with_score = _mean_float(_numeric_list([run.get("energy_evina") for run in with_runs]))
        without_time = _mean_float(_numeric_list([run.get("runtime_sec") for run in without_runs]))
        with_time = _mean_float(_numeric_list([run.get("runtime_sec") for run in with_runs]))
        delta_score = None if without_score is None or with_score is None else with_score - without_score
        delta_time = None if without_time is None or with_time is None else with_time - without_time
        rows.append(
            {
                "algoritmo": algorithm,
                "modo": mode,
                "n_sem": len(without_runs),
                "n_com": len(with_runs),
                "score_sem": _round_or_na(without_score, 4),
                "score_com": _round_or_na(with_score, 4),
                "delta_score": _round_or_na(delta_score, 4),
                "tempo_sem_s": _round_or_na(without_time, 4),
                "tempo_com_s": _round_or_na(with_time, 4),
                "delta_tempo_s": _round_or_na(delta_time, 4),
                "ganhou_tempo": bool(delta_time is not None and delta_time < 0),
                "score_nao_piorou": bool(delta_score is not None and delta_score <= 1e-9),
            }
        )
    return pd.DataFrame(rows)


def _article_ready_summary(runs: list[dict[str, Any]]) -> pd.DataFrame:
    stats = _statistical_summary_table(runs)
    if stats.empty:
        return stats
    rows = []
    for _, row in stats.iterrows():
        n = int(row.get("n", 0) or 0)
        rows.append(
            {
                "algoritmo": row.get("algoritmo"),
                "heuristica": row.get("heurÃ­stica", row.get("heurística")),
                "n": n,
                "score_media": row.get("evina_media"),
                "score_std": row.get("evina_std"),
                "score_ic95": row.get("evina_ic95"),
                "tempo_media_s": row.get("tempo_media_s"),
                "tempo_std_s": row.get("tempo_std_s"),
                "tempo_ic95_s": row.get("tempo_ic95_s"),
                "pronto_ic95": "sim" if n >= 2 else "nao",
                "forca_estatistica": "boa" if n >= 10 else ("inicial" if n >= 5 else "baixa"),
            }
        )
    return pd.DataFrame(rows)


def _statistical_summary_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for run in runs:
        key = (
            _algorithm_label(run.get("algorithm")),
            str(run.get("mode") or "unknown"),
            _heuristic_label(run),
        )
        grouped.setdefault(key, []).append(run)

    rows = []
    for (algorithm, mode, heuristic), group in sorted(
        grouped.items(),
        key=lambda item: (_algorithm_order_key(item[0][0]), item[0][1], item[0][2]),
    ):
        energies = _numeric_list([run.get("energy_evina") for run in group])
        runtimes = _numeric_list([run.get("runtime_sec") for run in group])
        evals = _numeric_list([run.get("n_eval") for run in group])
        rows.append(
            {
                "algoritmo": algorithm,
                "modo": mode,
                "heurística": heuristic,
                "n": len(group),
                "evina_media": _round_or_na(_mean_float(energies), 4),
                "evina_std": _round_or_na(_std(energies), 4),
                "evina_ic95": _fmt_ci95(energies),
                "tempo_media_s": _round_or_na(_mean_float(runtimes), 4),
                "tempo_std_s": _round_or_na(_std(runtimes), 4),
                "tempo_ic95_s": _fmt_ci95(runtimes),
                "avaliacoes_media": _round_or_na(_mean_float(evals), 2),
            }
        )
    return pd.DataFrame(rows)


def _anova_energy_by_algorithm(runs: list[dict[str, Any]]) -> dict[str, float] | None:
    grouped = _group_numeric_by_algorithm(runs, "energy_evina")
    groups = [values for values in grouped.values() if len(values) >= 2]
    if len(groups) < 2:
        return None
    try:
        from scipy import stats
    except Exception:  # noqa: BLE001
        return None
    try:
        result = stats.f_oneway(*groups)
    except Exception:  # noqa: BLE001
        return None
    k = len(groups)
    n_total = sum(len(group) for group in groups)
    if n_total <= k:
        return None
    f_value = float(result.statistic)
    p_value = float(result.pvalue)
    denominator = ((k - 1) * f_value) + (n_total - k)
    eta2 = (((k - 1) * f_value) / denominator) if denominator > 0 else 0.0
    return {
        "F": f_value,
        "p": p_value,
        "eta2": float(eta2),
        "df_between": float(k - 1),
        "df_within": float(n_total - k),
    }


def _infer_experimental_question(runs: list[dict[str, Any]]) -> str:
    for run in runs:
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        question = summary.get("protocol_question")
        if question:
            return str(question)
        config_subset = summary.get("config_resolved_subset") if isinstance(summary.get("config_resolved_subset"), dict) else {}
        question = config_subset.get("protocol_question")
        if question:
            return str(question)

    modes = {str(run.get("mode")) for run in runs if run.get("mode") not in (None, "")}
    heuristics = {_heuristic_label(run) for run in runs}
    if modes == {"full"} and {"Sem heurística", "Com heurística"}.issubset(heuristics):
        return "Pergunta: a heurística reduz tempo e/ou melhora energia no modo completo?"
    if modes == {"reduced"} and {"Sem heurística", "Com heurística"}.issubset(heuristics):
        return "Pergunta: a heurística reduz tempo e/ou melhora energia no modo reduzido?"
    if {"full", "reduced"}.issubset(modes) and len(heuristics) == 1:
        return "Pergunta: o modo reduzido é competitivo contra o modo completo?"
    return "Pergunta: análise combinada de algoritmo, heurística e modo de busca."


def _top_pose_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    ranked = sorted(
        [run for run in runs if isinstance(run.get("energy_evina"), (int, float))],
        key=lambda item: float(item["energy_evina"]),
    )
    for idx, run in enumerate(ranked[:15], start=1):
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        rows.append(
            {
                "Pose ID": idx,
                "Algoritmo": _algorithm_label(run.get("algorithm")),
                "E_vina": _fmt_metric(run.get("energy_evina")),
                "Modo": run.get("mode"),
                "Heurística": _heuristic_label(run),
                "Rodada": run.get("round_index") or "NA",
                "Tempo (s)": _fmt_metric(run.get("runtime_sec")),
            }
        )
    if not rows:
        rows.append({"Pose ID": "NA", "Algoritmo": "NA", "E_vina": "NA", "Modo": "NA", "Heurística": "NA", "Rodada": "NA", "Tempo (s)": "NA"})
    return pd.DataFrame(rows)


def _runs_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        rows.append(
            {
                "complexo": run.get("complex_id"),
                "algoritmo": _algorithm_label(run.get("algorithm")),
                "heurística": _heuristic_label(run),
                "modo": run.get("mode"),
                "evina": run.get("energy_evina"),
                "tempo_s": run.get("runtime_sec"),
                "avaliacoes": run.get("n_eval"),
                "rodada": run.get("round_index"),
                "fallback": run.get("fallback_executed"),
            }
        )
    return pd.DataFrame(rows)


def _radar_values(runs: list[dict[str, Any]]) -> list[float]:
    summaries = [run.get("summary") for run in runs if isinstance(run.get("summary"), dict)]
    keys = [
        ["p_col", "P_col", "collision_penalty"],
        ["p_exp", "P_exp", "exposure_penalty"],
        ["b_cont", "B_cont", "contact_bonus"],
        ["p_sxp", "P_sxp", "surface_exposure_penalty"],
    ]
    values: list[float] = []
    for aliases in keys:
        collected = []
        for summary in summaries:
            value = _first_present(summary, aliases)
            if isinstance(value, (int, float)):
                collected.append(abs(float(value)))
        values.append(min(1.0, (sum(collected) / len(collected)) if collected else 0.35))
    return values


def _kdtree_efficiency(runs: list[dict[str, Any]]) -> float | None:
    values = []
    for run in runs:
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        for key in ("kdtree_efficiency", "kd_tree_efficiency", "geom_kdtree_efficiency"):
            value = summary.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value) * (100.0 if value <= 1 else 1.0))
    if not values:
        return None
    return max(0.0, min(100.0, sum(values) / len(values)))


def _active_complex(runs: list[dict[str, Any]]) -> str:
    complexes = _sorted_unique(run.get("complex_id") for run in runs)
    return complexes[0] if complexes else "NA"


def _algorithm_label(value: Any) -> str:
    raw = str(value or "").strip()
    lowered = raw.lower().replace("-", "_").replace("+", "_")
    compact = lowered.replace("_", "").replace(" ", "")
    if compact in {
        "hybrid",
        "abcagvgos",
        "abcagvgossearch",
        "abcagvgosengine",
        "abcgavgos",
        "abcgavgossearch",
        "abcgavgosengine",
    }:
        return "ABC+AG+VGOS"
    if lowered in {"pso", "particle_swarm", "particle_swarm_optimization"}:
        return "PSO"
    if lowered in {"ag", "ga", "genetic", "genetic_algorithm", "algoritmo_genetico"}:
        return "AG"
    return raw or "NA"


def _ordered_algorithms(values: Iterable[Any]) -> list[str]:
    labels = {_algorithm_label(value) for value in values if value not in (None, "")}
    labels.update({"ABC+AG+VGOS", "PSO", "AG"})
    order = {"ABC+AG+VGOS": 0, "PSO": 1, "AG": 2}
    return sorted(labels, key=lambda item: (order.get(item, 99), item))


def _ordered_heuristics(values: Iterable[Any]) -> list[str]:
    labels = {str(value) for value in values if value not in (None, "")}
    labels.update({"Sem heurística", "Com heurística"})
    order = {"Sem heurística": 0, "Com heurística": 1}
    return sorted(labels, key=lambda item: (order.get(item, 99), item))


def _heuristic_label(run: dict[str, Any]) -> str:
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    config_subset = summary.get("config_resolved_subset") if isinstance(summary.get("config_resolved_subset"), dict) else {}
    policy = summary.get("confidence_policy")
    if not isinstance(policy, dict):
        policy = config_subset.get("confidence_policy") if isinstance(config_subset.get("confidence_policy"), dict) else {}
    mode = str(
        policy.get("mode")
        or summary.get("confidence_policy_mode")
        or summary.get("confidence_mode")
        or summary.get("arm")
        or summary.get("policy_arm")
        or ""
    ).strip().lower()
    path_hint = str(run.get("run_dir_relative") or run.get("run_dir") or "").lower()
    if not mode:
        for candidate in ("assist_fallback", "assist_selection", "assist_all", "observe", "off", "disabled"):
            if candidate in path_hint:
                mode = candidate
                break
    used = summary.get("confidence_policy_used")
    if used is None:
        used = policy.get("enabled")
    has_population = bool(summary.get("population_confidence_global") or summary.get("population_confidence_by_pocket"))
    if used is False or mode in {"off", "disabled", "none", "desligada"}:
        return "Sem heurística"
    if used is True or has_population or mode in {"observe", "assist_selection", "assist_fallback", "assist_all"}:
        return "Com heurística"
    return "Sem heurística"


def _sorted_unique(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if value not in (None, "")})


def _min_by(runs: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    valid = [run for run in runs if isinstance(run.get(key), (int, float))]
    if not valid:
        return None
    return min(valid, key=lambda run: float(run[key]))


def _best_algorithm_by_energy(runs: list[dict[str, Any]]) -> str | None:
    grouped = _group_numeric_by_algorithm(runs, "energy_evina")
    if not grouped:
        return None
    averages = {algorithm: sum(values) / len(values) for algorithm, values in grouped.items()}
    return min(averages, key=averages.get)


def _fmt_metric(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "NA"
    return f"{float(value):.4g}"


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _mean_float(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _ci95_half_width(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    std = _std(values)
    tcrit = _tcrit_95(len(values) - 1)
    return float(tcrit * (std / math.sqrt(len(values))))


def _ci95(values: list[float]) -> tuple[float, float] | None:
    mean = _mean_float(values)
    if mean is None:
        return None
    half = _ci95_half_width(values)
    return mean - half, mean + half


def _fmt_ci95(values: list[float]) -> str:
    interval = _ci95(values)
    if interval is None:
        return "NA"
    if len(values) < 2:
        return "NA"
    return f"[{interval[0]:.4g}, {interval[1]:.4g}]"


def _tcrit_95(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        25: 2.060,
        30: 2.042,
        40: 2.021,
        60: 2.000,
        120: 1.980,
    }
    if df <= 0:
        return 0.0
    if df in table:
        return table[df]
    larger = [key for key in table if key >= df]
    if larger:
        return table[min(larger)]
    return 1.96


def _round_or_na(value: float | None, digits: int) -> float | str:
    if value is None:
        return "NA"
    return round(float(value), digits)


def _algorithm_order_key(label: str) -> tuple[int, str]:
    order = {"ABC+AG+VGOS": 0, "PSO": 1, "AG": 2}
    return order.get(label, 99), label


def _numeric_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    numeric = []
    for value in values:
        if isinstance(value, (int, float)):
            numeric.append(float(value))
    return numeric


def _first_present(payload: dict[str, Any], aliases: list[str]) -> Any:
    for alias in aliases:
        if payload.get(alias) is not None:
            return payload.get(alias)
    return None

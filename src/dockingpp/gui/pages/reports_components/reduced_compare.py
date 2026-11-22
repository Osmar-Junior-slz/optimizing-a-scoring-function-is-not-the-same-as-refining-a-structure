"""Relatórios de execução única, reduced aggregate e comparação."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import tempfile

import pandas as pd
import streamlit as st

from dockingpp.gui.services.report_service import ReportBundle

from .dependencies import legacy_reports_module


class ReportsReducedCompareMixin:
    """Normaliza e apresenta resultados single, reduced e compare."""

    def _resumo_compare_exato(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Build compare KPIs from summary/result fields only."""

        return {
            "n_eval_total": self._summary_value(summary, ["total_n_eval", "n_eval_total", "n_eval", "evals", "evaluations"]),
            "n_filtered": self._summary_value(summary, ["n_filtered"]),
            "kept_ratio_final": self._summary_value(summary, ["kept_ratio_final"]),
            "runtime_total_s": self._summary_value(
                summary,
                ["total_runtime_sec", "runtime_sec", "runtime_total_s", "total_s", "elapsed_s", "elapsed_seconds", "elapsed"],
            ),
            "expensive_ran": self._summary_value(summary, ["expensive_ran_count", "rescored_candidates_count"]),
            "best_cheap": self._summary_value(
                summary,
                ["best_over_pockets_cheap", "triage_score", "best_score_cheap", "best_score", "best"],
            ),
            "best_expensive": self._summary_value(
                summary,
                ["best_over_pockets_expensive", "final_rescore", "rescore_proxy", "best_score_expensive", "best_expensive"],
            ),
        }


    @staticmethod
    def _load_paired_comparison_csv(csv_path: Path | None) -> list[dict[str, Any]]:
        """Carrega paired_comparison.csv quando disponível."""

        if csv_path is None or not csv_path.exists():
            return []
        try:
            frame = pd.read_csv(csv_path)
        except (OSError, pd.errors.EmptyDataError):
            return []
        return frame.to_dict(orient="records")

    def _load_fallback_metrics_series(
        self,
        base_dir: Path | None,
        reduced_summary: dict[str, Any],
    ) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
        """Carrega métricas do fallback_full quando reduced agregou via fallback."""

        if not bool(reduced_summary.get("fallback_to_full")):
            return None, None
        fallback_outdir = reduced_summary.get("fallback_full_outdir")
        if not fallback_outdir:
            return None, None
        fallback_dir = self._resolve_report_path(base_dir, str(fallback_outdir))
        if fallback_dir is None or not fallback_dir.exists():
            return None, None
        metrics_ts_path = fallback_dir / "metrics.timeseries.jsonl"
        metrics_path = fallback_dir / "metrics.jsonl"
        ts_records = legacy_reports_module().load_jsonl(metrics_ts_path) if metrics_ts_path.exists() else None
        records = legacy_reports_module().load_jsonl(metrics_path) if metrics_path.exists() else None
        return records, ts_records

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int(value: Any) -> int | None:
        parsed = ReportsReducedCompareMixin._to_float(value)
        if parsed is None:
            return None
        return int(parsed)

    @staticmethod
    def _build_pocket_rank_map(summary_reduced: dict[str, Any]) -> dict[str, int]:
        collections = legacy_reports_module().resolve_reduced_pocket_collections(summary_reduced)
        final_pockets = collections.get("executed_pockets", [])
        if not isinstance(final_pockets, list):
            return {}
        return {str(pocket_id): idx + 1 for idx, pocket_id in enumerate(final_pockets)}

    @staticmethod
    def _reduced_pocket_counts(summary_reduced: dict[str, Any]) -> dict[str, int]:
        collections = legacy_reports_module().resolve_reduced_pocket_collections(summary_reduced)
        rejected_raw = summary_reduced.get("rejected_pockets")
        rejected_count = len(rejected_raw) if isinstance(rejected_raw, list) else 0
        return {
            "candidate": len(collections.get("candidate_pockets", [])),
            "selected_after_scan": len(collections.get("selected_after_scan_pockets", [])),
            "accepted": len(collections.get("accepted_pockets", [])),
            "executed": len(collections.get("executed_pockets", [])),
            "rejected": rejected_count,
        }

    def _normalize_reduced_pocket_rows(
        self,
        raw_rows: list[dict[str, Any]],
        rank_map: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        rank_map = rank_map or {}
        for idx, raw in enumerate(raw_rows):
            if not isinstance(raw, dict):
                continue
            pocket_id_value = self._summary_value(
                raw,
                ["pocket_id", "best_pose_pocket_id", "best_pocket_id", "pocket", "id", "pocket_index"],
            )
            if pocket_id_value is None:
                continue
            pocket_id = str(pocket_id_value)

            pocket_rank_raw = self._summary_value(raw, ["pocket_rank", "rank"])
            pocket_rank = self._to_int(pocket_rank_raw)
            if pocket_rank is None:
                pocket_index = self._to_int(raw.get("pocket_index"))
                if pocket_index is not None:
                    pocket_rank = pocket_index + 1
            if pocket_rank is None and pocket_id in rank_map:
                pocket_rank = rank_map[pocket_id]
            if pocket_rank is None:
                pocket_rank = idx + 1

            best_score_cheap_raw = self._summary_value(raw, ["triage_score", "best_score_cheap", "best_score", "best_cheap", "best"])
            best_score_expensive_raw = self._summary_value(raw, ["final_rescore", "rescore_proxy", "best_score_expensive", "best_expensive"])
            runtime_raw = self._summary_value(
                raw,
                ["runtime_sec", "runtime_total_s", "total_runtime_sec", "total_s", "elapsed_s", "elapsed_seconds", "elapsed"],
            )
            n_eval_raw = self._summary_value(raw, ["n_eval_total", "n_eval", "evals", "evaluations"])
            search_time_raw = self._summary_value(raw, ["search_time_sec"])

            n_eval_total = self._to_int(n_eval_raw)
            normalized.append(
                {
                    "pocket_id": pocket_id,
                    "pocket_rank": pocket_rank,
                    "best_score_cheap": self._to_float(best_score_cheap_raw),
                    "best_score_expensive": self._to_float(best_score_expensive_raw),
                    "runtime_sec": self._to_float(runtime_raw),
                    "n_eval": n_eval_total,
                    "n_eval_total": n_eval_total,
                    "search_time_sec": self._to_float(search_time_raw),
                }
            )
        normalized.sort(key=lambda item: (item.get("pocket_rank") is None, item.get("pocket_rank"), item.get("pocket_id")))
        return normalized

    @staticmethod
    def _has_pocket_detail(rows: list[dict[str, Any]]) -> bool:
        for row in rows:
            if row.get("pocket_id") is None:
                continue
            if any(row.get(key) is not None for key in ("best_score_cheap", "best_score_expensive", "runtime_sec", "n_eval_total", "n_eval")):
                return True
        return False

    def _load_reduced_pocket_rows_from_subdirs(
        self,
        reduced_run_dir: Path | None,
        rank_map: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        if reduced_run_dir is None or not reduced_run_dir.exists():
            return []
        rank_map = rank_map or {}
        rows: list[dict[str, Any]] = []
        for child in sorted(reduced_run_dir.iterdir()):
            if not child.is_dir():
                continue
            summary_payload = self._load_run_payload(child / "summary.json")
            result_payload = self._load_run_payload(child / "result.json")
            merged = self._merge_summary_result(summary_payload, result_payload)
            if not merged:
                continue
            mode = str(merged.get("mode") or "").lower()
            search_mode = str(merged.get("search_space_mode") or "").lower()
            if mode and mode != "single":
                continue
            if search_mode and search_mode not in {"reduced", "pockets"}:
                continue
            pocket_id = self._summary_value(merged, ["pocket_id", "best_pose_pocket_id", "best_pocket_id"])
            if pocket_id is None:
                pocket_id = child.name
            merged["pocket_id"] = str(pocket_id)
            if rank_map.get(str(pocket_id)) is not None:
                merged["pocket_rank"] = rank_map[str(pocket_id)]
            rows.append(merged)
        return self._normalize_reduced_pocket_rows(rows, rank_map)

    def _resolve_reduced_pocket_rows_for_compare(
        self,
        summary_reduced: dict[str, Any],
        reduced_run_dir: Path | None,
    ) -> tuple[list[dict[str, Any]], str]:
        rank_map = self._build_pocket_rank_map(summary_reduced)
        per_pocket_summary = summary_reduced.get("per_pocket_results")
        if isinstance(per_pocket_summary, list):
            rows_from_summary = self._normalize_reduced_pocket_rows(per_pocket_summary, rank_map)
            if self._has_pocket_detail(rows_from_summary):
                return rows_from_summary, "summary"

        if self._is_reduced_aggregate_fallback(summary_reduced):
            return [], "fallback_full"

        rows_from_subdirs = self._load_reduced_pocket_rows_from_subdirs(reduced_run_dir, rank_map)
        if self._has_pocket_detail(rows_from_subdirs):
            return rows_from_subdirs, "subdirs"
        return [], "none"

    def _render_compare_reduced_pockets(
        self,
        summary_reduced: dict[str, Any],
        temp_dir: Path,
        reduced_run_dir: Path | None,
    ) -> None:
        st.subheader("Detalhes por bolsão no modo Reduzido")
        st.caption("A compara\u00e7\u00e3o acima \u00e9 agregada; abaixo est\u00e3o os resultados individuais por bols\u00e3o.")

        rows, source = self._resolve_reduced_pocket_rows_for_compare(summary_reduced, reduced_run_dir)
        if not rows:
            if source == "fallback_full":
                st.info("Sem bolsões viáveis para detalhar (fallback para full).")
            else:
                st.info("Sem dados por bols\u00e3o no summary agregado e sem subpastas v\u00e1lidas no run reduced.")
            return

        frame = pd.DataFrame(rows)
        preferred_cols = [
            "pocket_id",
            "pocket_rank",
            "best_score_cheap",
            "best_score_expensive",
            "runtime_sec",
            "n_eval",
        ]
        visible_cols = [col for col in preferred_cols if col in frame.columns]
        if visible_cols:
            st.dataframe(frame[visible_cols], use_container_width=True)
        else:
            st.dataframe(frame, use_container_width=True)

        if source == "summary":
            st.caption("Fonte dos bols\u00f5es: summary_reduced.per_pocket_results")
        elif source == "subdirs":
            st.caption("Fonte dos bols\u00f5es: subpastas do run reduced (fallback de leitura)")

        pocket_plot_payload = {
            "mode": "reduced_aggregate",
            "best_pocket_id": summary_reduced.get("best_pocket_id"),
            "per_pocket_results": rows,
        }
        pocket_png = temp_dir / "pocket_rank_effect_reduced_compare.png"
        if legacy_reports_module().plot_pocket_rank_effect(pocket_plot_payload, pocket_png) is None and pocket_png.exists():
            self._render_png(pocket_png, "Efeito por bols\u00e3o (Reduced)")
        elif pocket_png.exists():
            self._render_png(pocket_png, "Efeito por bols\u00e3o (Reduced)")
        else:
            st.info("Sem dados suficientes para gr\u00e1fico por bols\u00e3o no Reduced.")


    def _render_kpis(self, summary_data: dict[str, Any], series: dict[str, Any] | None, key_suffix: str) -> None:
        st.subheader("Resumo do Gap")
        resumo = self._resumo_gap(series or {}, summary_data)
        st.table(pd.DataFrame([resumo]))

        st.subheader("Resumo da execução")
        summary_rows = self._build_summary_rows(summary_data, resumo)
        st.table(summary_rows)
        self._render_score_semantics_caption(summary_data)
        summary_df = pd.DataFrame(summary_rows)
        self._download_csv_button("Baixar tabela resumo (CSV)", summary_df, "resumo_execucao.csv", key_suffix)

    def _render_single_report(
        self,
        bundle: ReportBundle,
        metrics_records: list[dict[str, Any]] | None,
        summary_data: dict[str, Any],
        metrics_timeseries: list[dict[str, Any]] | None = None,
        key_suffix: str = "single",
    ) -> None:
        """Renderiza relatório de execução única com gráficos e resumos."""

        series_records = self._select_series_records(metrics_records, metrics_timeseries)
        series = legacy_reports_module().extract_series(series_records or [])
        self._render_kpis(summary_data, series, key_suffix)

        if not self._has_time_series(series_records):
            st.info("Sem série temporal; exibindo KPIs.")
            return

        self._render_warning_missing(series)
        st.subheader("Gráficos")
        temp_dir = Path(tempfile.mkdtemp(prefix="reports_"))

        reduction_png = temp_dir / "search_space_reduction.png"
        if legacy_reports_module().plot_search_space_reduction({**summary_data, **self._resumo_gap(series, summary_data)}, reduction_png):
            self._render_png(reduction_png, "Redução do espaço de busca")

        convergence_png = temp_dir / "convergence.png"
        if legacy_reports_module().plot_convergence(series, convergence_png):
            self._render_png(convergence_png, "Convergência: best_score_cheap vs n_eval_cumulative")


    def _render_reduced_aggregate_report(
        self,
        bundle: ReportBundle,
        metrics_records: list[dict[str, Any]] | None,
        summary_data: dict[str, Any],
    ) -> None:
        """Renderiza relatório agregado do reduced por bolsão."""

        st.subheader("Reduced agregado por bolsão")
        st.json(
            {
                "best_pocket_id": summary_data.get("best_pocket_id"),
                "best_over_pockets_cheap": summary_data.get("best_over_pockets_cheap"),
                "best_over_pockets_expensive": summary_data.get("best_over_pockets_expensive"),
                "total_n_eval": summary_data.get("total_n_eval"),
                "total_runtime_sec": summary_data.get("total_runtime_sec"),
                "budget_policy": summary_data.get("budget_policy"),
                "budget_delta": summary_data.get("budget_delta"),
                "fallback_to_full": summary_data.get("fallback_to_full"),
                "fallback_reason": summary_data.get("fallback_reason"),
                "executed_mode": summary_data.get("executed_mode"),
                "n_pockets_total": summary_data.get("n_pockets_total"),
                "n_pockets_used": summary_data.get("n_pockets_used"),
            }
        )
        if bool(summary_data.get("fallback_to_full")):
            st.warning("Reduced sem bolsões viáveis; execução caiu em fallback para full")
            st.caption(f"fallback_reason={summary_data.get('fallback_reason')} | n_pockets_used={summary_data.get('n_pockets_used')}")
        counts = self._reduced_pocket_counts(summary_data)
        st.caption(
            "colecoes_de_bolsoes: "
            f"candidatos={counts['candidate']} | "
            f"apos_scan={counts['selected_after_scan']} | "
            f"aceitos={counts['accepted']} | "
            f"executados={counts['executed']} | "
            f"rejeitados={counts['rejected']}"
        )
        per_pocket = summary_data.get("per_pocket_results", [])
        if isinstance(per_pocket, list) and per_pocket:
            frame = pd.DataFrame(per_pocket)
            st.subheader("Resultados por bolsão")
            st.dataframe(frame)
        else:
            st.info("Sem resultados por bolsão no summary agregado.")

        rejected = summary_data.get("rejected_pockets", [])
        if isinstance(rejected, list) and rejected:
            st.subheader("Bolsões rejeitados")
            st.dataframe(pd.DataFrame(rejected))

        temp_dir = Path(tempfile.mkdtemp(prefix="reports_reduced_agg_"))
        pocket_png = temp_dir / "pocket_rank_effect.png"
        if per_pocket:
            if legacy_reports_module().plot_pocket_rank_effect(summary_data, pocket_png) is None and pocket_png.exists():
                self._render_png(pocket_png, "Custo/score por bolsão")
            elif pocket_png.exists():
                self._render_png(pocket_png, "Custo/score por bolsão")

    def _render_compare_report(
        self,
        bundle: ReportBundle,
        metrics_full: list[dict[str, Any]] | None,
        metrics_reduced: list[dict[str, Any]] | None,
        summary_full: dict[str, Any],
        summary_reduced: dict[str, Any],
        metrics_full_timeseries: list[dict[str, Any]] | None = None,
        metrics_reduced_timeseries: list[dict[str, Any]] | None = None,
        key_suffix: str = "compare",
        reduced_run_dir: Path | None = None,
    ) -> None:
        """Renderiza relatório comparativo entre modos completo e reduzido."""

        st.subheader("Comparação: Completo vs Reduzido")
        if self._is_reduced_aggregate_fallback(summary_reduced):
            st.warning("Reduced sem bolsões viáveis; execução caiu em fallback para full")
            st.caption(
                f"fallback_reason={summary_reduced.get('fallback_reason')} | "
                f"n_pockets_used={summary_reduced.get('n_pockets_used')}"
            )
            rejected = summary_reduced.get("rejected_pockets")
            if isinstance(rejected, list) and rejected:
                st.caption(f"rejected_pockets={len(rejected)}")

        rows = legacy_reports_module().build_compare_table(bundle.main_json)
        if rows:
            st.subheader("Comparação avançada")
            st.table(rows)
            compare_df = pd.DataFrame(rows)
            self._download_csv_button(
                "Baixar tabela resumo (CSV)",
                compare_df,
                "comparacao_full_reduced.csv",
                f"{key_suffix}_full_reduced",
            )

        series_full_records = self._select_series_records(metrics_full, metrics_full_timeseries)
        series_reduced_records = self._select_series_records(metrics_reduced, metrics_reduced_timeseries)
        series_full = legacy_reports_module().extract_series(series_full_records or [])
        series_reduced = legacy_reports_module().extract_series(series_reduced_records or [])
        resumo_full = self._resumo_compare_exato(summary_full)
        resumo_reduced = self._resumo_compare_exato(summary_reduced)
        if self._is_reduced_aggregate_fallback(summary_reduced):
            reduced_block = self._extract_compare_block(bundle.main_json, "reduced")
            resumo_reduced = self._resolve_compare_fallback_metrics(resumo_reduced, reduced_block)

        st.subheader("Tabela comparativa (KPIs)")
        st.table(pd.DataFrame([{"Modo": "Completo", **resumo_full}, {"Modo": "Reduzido", **resumo_reduced}]))

        st.subheader("Gráficos (Full vs Reduced)")
        temp_dir = Path(tempfile.mkdtemp(prefix="reports_compare_"))

        cost_png = temp_dir / "cost_comparison.png"
        if legacy_reports_module().plot_cost_comparison(resumo_full, resumo_reduced, cost_png):
            self._render_png(cost_png, "Comparação de custo: runtime_sec e n_eval_total (Reduced pode estar em fallback→full)")
        else:
            st.info("Sem dados suficientes para gráfico de custo.")

        reduction_full_png = temp_dir / "reduction_full.png"
        reduction_reduced_png = temp_dir / "reduction_reduced.png"
        has_full_reduction = legacy_reports_module().plot_search_space_reduction({**summary_full, **resumo_full}, reduction_full_png)
        if self._is_reduced_aggregate_fallback(summary_reduced):
            has_reduced_reduction = False
        else:
            has_reduced_reduction = legacy_reports_module().plot_search_space_reduction({**summary_reduced, **resumo_reduced}, reduction_reduced_png)
        if has_full_reduction or has_reduced_reduction:
            col1, col2 = st.columns(2)
            with col1:
                if has_full_reduction:
                    self._render_png(reduction_full_png, "Redução do espaço de busca (Completo)")
            with col2:
                if has_reduced_reduction:
                    self._render_png(reduction_reduced_png, "Redução do espaço de busca (Reduzido)")
        else:
            st.info("Sem dados suficientes para redução do espaço de busca.")

        convergence_png = temp_dir / "convergence_compare.png"
        reduced_series_for_plot = series_reduced
        if self._is_reduced_aggregate_fallback(summary_reduced) and not metrics_reduced and not metrics_reduced_timeseries:
            st.info("Fallback no reduced sem dados de série temporal; curva Reduced ocultada.")
            reduced_series_for_plot = {}
        if legacy_reports_module().plot_convergence({"full": series_full, "reduced": reduced_series_for_plot}, convergence_png):
            self._render_png(convergence_png, "Convergência: best_score_cheap vs n_eval_cumulative")
        else:
            st.info("Sem dados suficientes para convergência.")

        paired_rows: list[dict[str, Any]] = []
        paired_obj = bundle.aux_jsons.get("paired_comparison")
        if isinstance(paired_obj, dict):
            if all(k in paired_obj for k in ("speedup_runtime", "speedup_eval", "delta_score_cheap")):
                paired_rows = [paired_obj]
            elif isinstance(paired_obj.get("rows"), list):
                paired_rows = [row for row in paired_obj.get("rows", []) if isinstance(row, dict)]
        paired_png = temp_dir / "paired_comparison.png"
        if legacy_reports_module().plot_paired_comparison(paired_rows, paired_png):
            self._render_png(paired_png, "Comparação pareada: speedup_runtime, speedup_eval, delta_score_cheap")
        elif paired_rows:
            st.info("Dados pareados insuficientes para plot.")

        self._render_compare_reduced_pockets(summary_reduced, temp_dir, reduced_run_dir)


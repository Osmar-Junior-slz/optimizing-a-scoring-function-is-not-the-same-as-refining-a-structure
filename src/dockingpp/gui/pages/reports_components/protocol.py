"""Relatórios do protocolo científico e navegação por execuções."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dockingpp.gui.services.report_service import ReportBundle
from dockingpp.reporting.loaders import ReportRun

from .dependencies import legacy_reports_module


class ReportsProtocolMixin:
    """Carrega execuções persistidas e apresenta protocolos científicos."""

    @staticmethod
    def _load_run_payload(path: Path | None) -> dict[str, Any]:
        if path and path.exists():
            try:
                return legacy_reports_module().load_any_json(path)
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _render_run_block(
        self,
        label: str,
        run: ReportRun,
    ) -> None:
        st.subheader(label)
        summary_payload = self._load_run_payload(run.summary_path)
        result_payload = self._load_run_payload(run.result_path)
        merged = self._merge_summary_result(summary_payload, result_payload)
        metrics_records = legacy_reports_module().load_jsonl(run.metrics_path) if run.metrics_path else None
        metrics_timeseries = legacy_reports_module().load_jsonl(run.metrics_timeseries_path) if run.metrics_timeseries_path else None
        self._render_single_report(
            ReportBundle(kind="single", main_json=merged, metrics=metrics_records, aux_jsons={}),
            metrics_records,
            merged,
            metrics_timeseries,
            key_suffix=f"{run.kind}_{run.run_id or run.run_dir.name}",
        )

    def _render_algorithm_protocol_report(self, selected_folder: Path) -> None:
        run_table = selected_folder / "run_table.csv"
        if not run_table.exists():
            st.warning("A pasta selecionada nao contem run_table.csv.")
            return

        st.subheader("Protocolo AG vs PSO vs ABC+AG+VGOS")
        try:
            visual = legacy_reports_module().build_algorithm_protocol_visual_report(selected_folder)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Não foi possível gerar visualizações automaticamente: {exc}")
            visual = {"generated": False, "figures": []}

        try:
            frame = pd.read_csv(run_table)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Não foi possível carregar run_table.csv: {exc}")
            return

        ok_frame = frame[frame["status"] == "ok"] if "status" in frame.columns else frame
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Execuções", int(len(ok_frame)))
        with c2:
            st.metric("Algoritmos", int(ok_frame["engine"].nunique()) if "engine" in ok_frame.columns else 0)
        with c3:
            st.metric(
                "Bracos de confianca",
                int(ok_frame["confidence_arm"].nunique()) if "confidence_arm" in ok_frame.columns else 0,
            )

        figures = [Path(path) for path in visual.get("figures", [])]
        if not figures:
            figures = sorted((selected_folder / "figures").glob("*.png"))
        for figure in figures:
            if figure.exists():
                st.image(str(figure), caption=figure.name, use_container_width=True)

        summary_json = selected_folder / "visual_summary.json"
        if summary_json.exists():
            try:
                summary_payload = json.loads(summary_json.read_text(encoding="utf-8"))
                group_stats = summary_payload.get("group_stats")
                if isinstance(group_stats, list) and group_stats:
                    st.subheader("Tabela consolidada")
                    st.dataframe(pd.DataFrame(group_stats), use_container_width=True)
            except Exception:
                pass

        visible_cols = [
            "case_id",
            "engine",
            "seed",
            "confidence_arm",
            "final_rescore",
            "runtime_sec",
            "fallback_recommended",
            "fallback_executed",
            "confidence_policy_effect",
        ]
        available_cols = [col for col in visible_cols if col in frame.columns]
        if available_cols:
            st.subheader("Dados por execucao")
            st.dataframe(frame[available_cols], use_container_width=True)
        self._download_csv_button("Baixar run_table.csv", frame, "run_table.csv", "algorithm_protocol")

        dashboard = selected_folder / "dashboard.html"
        if dashboard.exists():
            st.download_button(
                "Baixar dashboard HTML",
                dashboard.read_bytes(),
                file_name="dashboard.html",
                mime="text/html",
                key="download_algorithm_dashboard",
            )

    def _render_folder_reports(self, selected_folder: Path) -> None:
        algorithm_protocol_csv = selected_folder / "run_table.csv"
        audit_files = self._discover_audit_batch_files(selected_folder)
        runs = legacy_reports_module().find_report_runs(selected_folder)
        if not runs and not audit_files and not algorithm_protocol_csv.exists():
            st.warning("Nenhuma execução encontrada na pasta.")
            return

        view_options = ["Separado", "Comparar"]
        if algorithm_protocol_csv.exists():
            view_options = ["Protocolo AG/PSO/Hibrido", *view_options]
        if audit_files:
            view_options = ["Auditoria Exata (Batch)", *view_options]
        view_mode = st.selectbox("Modo de visualização", view_options)

        if view_mode == "Protocolo AG/PSO/Hibrido":
            self._render_algorithm_protocol_report(selected_folder)
            return

        if view_mode == "Auditoria Exata (Batch)":
            if not self._render_batch_audit_report(selected_folder):
                st.warning("Não foi possível carregar artefatos de auditoria nesta pasta.")
            return

        if not runs:
            st.warning("Nenhuma execução Full/Reduced encontrada para os modos Separado/Comparar.")
            return

        full_runs = [run for run in runs if run.kind == "full"]
        reduced_runs = [run for run in runs if run.kind == "reduced"]

        def _select_run(label: str, options: list[ReportRun]) -> ReportRun | None:
            if not options:
                st.warning(f"Nenhum run {label.lower()} encontrado.")
                return None
            labels = [run.label() for run in options]
            selected = st.selectbox(f"Run {label}", labels, key=f"run_{label}")
            return options[labels.index(selected)]

        if view_mode == "Separado":
            if full_runs or reduced_runs:
                selected_full = _select_run("Full", full_runs) if full_runs else None
                selected_reduced = _select_run("Reduced", reduced_runs) if reduced_runs else None
                if selected_full:
                    self._render_run_block("Full", selected_full)
                if selected_reduced:
                    self._render_run_block("Reduced", selected_reduced)
                return

            unknown_runs = [run for run in runs if run.kind == "unknown"]
            selected_unknown = _select_run("Run", unknown_runs)
            if selected_unknown:
                self._render_run_block("Run", selected_unknown)
            return

        if view_mode == "Comparar":
            selected_full, selected_reduced = legacy_reports_module().pair_full_reduced(runs)
            if full_runs:
                selected_full = _select_run("Full", full_runs) or selected_full
            if reduced_runs:
                selected_reduced = _select_run("Reduced", reduced_runs) or selected_reduced
            if not selected_full or not selected_reduced:
                st.warning("É necessário selecionar runs Full e Reduced para comparar.")
                return

            summary_full = self._merge_summary_result(
                self._load_run_payload(selected_full.summary_path),
                self._load_run_payload(selected_full.result_path),
            )
            summary_reduced = self._merge_summary_result(
                self._load_run_payload(selected_reduced.summary_path),
                self._load_run_payload(selected_reduced.result_path),
            )
            metrics_full = legacy_reports_module().load_jsonl(selected_full.metrics_path) if selected_full.metrics_path else None
            metrics_reduced = legacy_reports_module().load_jsonl(selected_reduced.metrics_path) if selected_reduced.metrics_path else None
            metrics_full_timeseries = (
                legacy_reports_module().load_jsonl(selected_full.metrics_timeseries_path) if selected_full.metrics_timeseries_path else None
            )
            metrics_reduced_timeseries = (
                legacy_reports_module().load_jsonl(selected_reduced.metrics_timeseries_path) if selected_reduced.metrics_timeseries_path else None
            )
            paired_rows: list[dict[str, Any]] = []
            for candidate in [
                selected_full.run_dir / "paired_comparison.csv",
                selected_reduced.run_dir / "paired_comparison.csv",
                selected_full.run_dir.parent / "paired_comparison.csv",
                selected_folder / "paired_comparison.csv",
            ]:
                paired_rows = self._load_paired_comparison_csv(candidate)
                if paired_rows:
                    break

            compare_payload = {
                "full": dict(summary_full),
                "reduced": dict(summary_reduced),
            }
            self._render_compare_report(
                ReportBundle(
                    kind="compare",
                    main_json=compare_payload,
                    metrics=None,
                    aux_jsons={"paired_comparison": {"rows": paired_rows}},
                ),
                metrics_full,
                metrics_reduced,
                summary_full,
                summary_reduced,
                metrics_full_timeseries,
                metrics_reduced_timeseries,
                reduced_run_dir=selected_reduced.run_dir,
            )
            return

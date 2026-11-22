"""Página de relatórios para visualização de métricas e resultados."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import tempfile

import pandas as pd
import streamlit as st

from dockingpp.gui.pages.base import BasePage
from dockingpp.gui.services.dialog_service import choose_directory
from dockingpp.gui.services.report_service import (
    ReportBundle,
    build_compare_table,
    infer_json_kind,
)
from dockingpp.gui.state import AppState, StateKeys
from dockingpp.gui.ui.components import download_json_button
from dockingpp.reporting.loaders import (
    ReportRun,
    extract_series,
    find_matching_jsonl,
    find_report_runs,
    load_any_json,
    load_jsonl,
    pair_full_reduced,
    resolve_reduced_pocket_collections,
)
from dockingpp.reporting.algorithm_protocol import build_algorithm_protocol_visual_report
from dockingpp.reporting.plots import (
    plot_convergence,
    plot_cost_comparison,
    plot_paired_comparison,
    plot_pocket_rank_effect,
    plot_search_space_reduction,
)
from dockingpp.gui.pages.reports_components import (
    ReportsAuditMixin,
    ReportsDataMixin,
    ReportsProtocolMixin,
    ReportsReducedCompareMixin,
)


class ReportsPage(
    ReportsDataMixin,
    ReportsAuditMixin,
    ReportsReducedCompareMixin,
    ReportsProtocolMixin,
    BasePage,
):
    """Página de relatórios com gráficos de convergência e comparações."""

    id = "Relatórios"
    title = "Relatórios"

    def render(self, state: AppState) -> None:
        """Renderiza a página completa de relatórios."""

        st.header("Relatórios")
        if st.session_state.get(StateKeys.REPORTS_ROOT_PENDING):
            st.session_state[StateKeys.REPORTS_ROOT] = st.session_state.pop(StateKeys.REPORTS_ROOT_PENDING)
        if StateKeys.REPORTS_ROOT not in st.session_state:
            st.session_state[StateKeys.REPORTS_ROOT] = "runs"

        source = st.radio(
            "Fonte do relatório",
            ["Selecionar pasta", "Selecionar arquivo JSON"],
            horizontal=True,
        )

        main_json: dict[str, Any] | None = None
        aux_jsons: dict[str, dict[str, Any]] = {}
        metrics_records: list[dict[str, Any]] | None = None
        metrics_full: list[dict[str, Any]] | None = None
        metrics_reduced: list[dict[str, Any]] | None = None
        main_json_path: Path | None = None
        metrics_path: Path | None = None
        metrics_full_path: Path | None = None
        metrics_reduced_path: Path | None = None
        main_json_upload: st.runtime.uploaded_file_manager.UploadedFile | None = None
        metrics_upload: st.runtime.uploaded_file_manager.UploadedFile | None = None
        metrics_full_upload: st.runtime.uploaded_file_manager.UploadedFile | None = None
        metrics_reduced_upload: st.runtime.uploaded_file_manager.UploadedFile | None = None
        base_dir: Path | None = None

        if source == "Selecionar pasta":
            folder_path = st.text_input("Pasta do relatório", key=StateKeys.REPORTS_ROOT)
            if st.button("Selecionar pasta..."):
                chosen = choose_directory()
                if chosen is not None:
                    st.session_state[StateKeys.REPORTS_ROOT_PENDING] = chosen
                    st.rerun()
            if not folder_path:
                st.info("Informe uma pasta com arquivos JSON para continuar.")
                return

            selected_folder = Path(folder_path).expanduser()
            base_dir = selected_folder
            if not selected_folder.exists():
                st.warning("Pasta informada não existe.")
                return
            self._render_folder_reports(selected_folder)
            return
        else:
            main_json_upload = st.file_uploader("JSON principal", type=["json"])
            if not main_json_upload:
                st.info("Selecione um arquivo JSON para continuar.")
                return
            main_json = self._load_uploaded_json(main_json_upload)
            kind = infer_json_kind(main_json)

            if kind in {"single", "reduced_aggregate"}:
                metrics_upload = st.file_uploader("Arquivo de métricas (.jsonl)", type=["jsonl"])
                if metrics_upload:
                    metrics_records = self._load_uploaded_jsonl(metrics_upload)
            elif kind == "compare":
                metrics_full_upload = st.file_uploader(
                    "Métricas do modo completo (.jsonl)",
                    type=["jsonl"],
                    key="metrics_full_upload",
                )
                metrics_reduced_upload = st.file_uploader(
                    "Métricas do modo reduzido (.jsonl)",
                    type=["jsonl"],
                    key="metrics_reduced_upload",
                )
                if metrics_full_upload:
                    metrics_full = self._load_uploaded_jsonl(metrics_full_upload)
                if metrics_reduced_upload:
                    metrics_reduced = self._load_uploaded_jsonl(metrics_reduced_upload)

        if main_json is None:
            return

        kind = infer_json_kind(main_json)
        summary_data = main_json
        summary_full = main_json
        summary_reduced = main_json

        if kind in {"single", "reduced_aggregate"}:
            summary_path = self._resolve_report_path(
                self._resolve_report_path(base_dir, main_json.get("outdir")),
                main_json.get("summary_path"),
            )
            summary_data = self._load_summary_data(summary_path, main_json)
            if metrics_records is None:
                report_metrics_path = self._resolve_report_path(
                    self._resolve_report_path(base_dir, summary_data.get("outdir")),
                    summary_data.get("metrics_path"),
                )
                if report_metrics_path and report_metrics_path.exists():
                    metrics_path = report_metrics_path
                    metrics_records = load_jsonl(metrics_path)
                elif main_json_path:
                    report_metrics_path = find_matching_jsonl(main_json_path)
                    if report_metrics_path and report_metrics_path.exists():
                        metrics_path = report_metrics_path
                        metrics_records = load_jsonl(metrics_path)
        elif kind == "compare":
            full_block = self._extract_compare_block(main_json, "full") or {}
            reduced_block = self._extract_compare_block(main_json, "reduced") or {}
            full_summary_path = self._resolve_report_path(
                self._resolve_report_path(base_dir, full_block.get("outdir")),
                full_block.get("summary_path"),
            )
            reduced_summary_path = self._resolve_report_path(
                self._resolve_report_path(base_dir, reduced_block.get("outdir")),
                reduced_block.get("summary_path"),
            )
            summary_full = self._load_summary_data(full_summary_path, full_block or main_json)
            summary_reduced = self._load_summary_data(reduced_summary_path, reduced_block or main_json)
            if metrics_full is None:
                report_full_path = self._resolve_report_path(
                    self._resolve_report_path(base_dir, full_block.get("outdir")),
                    full_block.get("metrics_path"),
                )
                if report_full_path and report_full_path.exists():
                    metrics_full_path = report_full_path
                    metrics_full = load_jsonl(metrics_full_path)
            if metrics_reduced is None:
                report_reduced_path = self._resolve_report_path(
                    self._resolve_report_path(base_dir, reduced_block.get("outdir")),
                    reduced_block.get("metrics_path"),
                )
                if report_reduced_path and report_reduced_path.exists():
                    metrics_reduced_path = report_reduced_path
                    metrics_reduced = load_jsonl(metrics_reduced_path)
                else:
                    fallback_records, fallback_timeseries = self._load_fallback_metrics_series(base_dir, summary_reduced)
                    if fallback_records is not None:
                        metrics_reduced = fallback_records
                    if fallback_timeseries is not None:
                        metrics_reduced = fallback_timeseries

        bundle = ReportBundle(
            kind=kind,
            main_json=main_json,
            metrics=metrics_records,
            aux_jsons=aux_jsons,
        )

        if bundle.kind == "single":
            self._render_single_report(bundle, metrics_records, summary_data)
        elif bundle.kind == "reduced_aggregate":
            self._render_reduced_aggregate_report(bundle, metrics_records, summary_data)
        elif bundle.kind == "compare":
            self._render_compare_report(
                bundle,
                metrics_full,
                metrics_reduced,
                summary_full,
                summary_reduced,
            )
        else:
            st.warning("Não foi possível identificar o tipo do JSON. Exibindo conteúdo bruto.")
            st.json(bundle.main_json)

        st.subheader("Downloads")
        if main_json_path:
            download_json_button("Baixar JSON principal", main_json_path, filename=main_json_path.name, warn_missing=False)
        elif main_json_upload:
            self._download_json_payload("Baixar JSON principal", main_json, main_json_upload.name)

        if bundle.kind in {"single", "reduced_aggregate"}:
            if metrics_path:
                st.download_button(
                    "Baixar métricas (.jsonl)",
                    data=metrics_path.read_text(encoding="utf-8"),
                    file_name=metrics_path.name,
                    mime="application/jsonl",
                )
            elif metrics_upload and metrics_records is not None:
                self._download_jsonl_payload("Baixar métricas (.jsonl)", metrics_records, metrics_upload.name)
        elif bundle.kind == "compare":
            if metrics_full_path:
                st.download_button(
                    "Baixar métricas completo (.jsonl)",
                    data=metrics_full_path.read_text(encoding="utf-8"),
                    file_name=metrics_full_path.name,
                    mime="application/jsonl",
                )
            elif metrics_full_upload and metrics_full is not None:
                self._download_jsonl_payload(
                    "Baixar métricas completo (.jsonl)",
                    metrics_full,
                    metrics_full_upload.name,
                )

            if metrics_reduced_path:
                st.download_button(
                    "Baixar métricas reduzido (.jsonl)",
                    data=metrics_reduced_path.read_text(encoding="utf-8"),
                    file_name=metrics_reduced_path.name,
                    mime="application/jsonl",
                )
            elif metrics_reduced_upload and metrics_reduced is not None:
                self._download_jsonl_payload(
                    "Baixar métricas reduzido (.jsonl)",
                    metrics_reduced,
                    metrics_reduced_upload.name,
                )

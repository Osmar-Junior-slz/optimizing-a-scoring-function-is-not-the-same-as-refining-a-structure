"""Descoberta e renderização dos artefatos de auditoria batch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .dependencies import legacy_reports_module


class ReportsAuditMixin:
    """Agrupa o fluxo específico dos relatórios de auditoria exata."""

    @staticmethod
    def _discover_audit_batch_files(selected_folder: Path) -> dict[str, Path]:
        """Locate batch-level audit artifacts inside selected folder."""

        candidates = {
            "audit_json": selected_folder / "audit_cases_exact.json",
            "audit_jsonl": selected_folder / "audit_cases_exact.jsonl",
            "audit_csv": selected_folder / "audit_cases_exact.csv",
            "audit_manifest": selected_folder / "audit_manifest.json",
            "batch_report": selected_folder / "batch_report.json",
        }
        return {name: path for name, path in candidates.items() if path.exists()}

    @staticmethod
    def _load_audit_batch_rows(artifact_paths: dict[str, Path]) -> tuple[list[dict[str, Any]], str]:
        """Load audit rows preferring strict audit payloads."""

        audit_json = artifact_paths.get("audit_json")
        if audit_json:
            payload = legacy_reports_module().load_any_json(audit_json)
            rows = payload.get("rows")
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)], "audit_cases_exact.json"

        batch_report = artifact_paths.get("batch_report")
        if batch_report:
            payload = legacy_reports_module().load_any_json(batch_report)
            rows = payload.get("rows")
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)], "batch_report.json"

        audit_csv = artifact_paths.get("audit_csv")
        if audit_csv:
            try:
                frame = pd.read_csv(audit_csv)
            except (OSError, pd.errors.EmptyDataError):
                return [], "audit_cases_exact.csv"
            return frame.to_dict(orient="records"), "audit_cases_exact.csv"

        return [], "none"

    def _render_batch_audit_report(self, selected_folder: Path) -> bool:
        """Render strict audit view from batch artifacts without metric inference."""

        artifact_paths = self._discover_audit_batch_files(selected_folder)
        if not artifact_paths:
            return False

        rows, source_name = self._load_audit_batch_rows(artifact_paths)
        st.subheader("Auditoria Exata (Batch)")
        st.caption(
            "Valores com prioridade para `audit_cases_exact.*`, sem inferencia por series de metricas."
        )
        if not rows:
            st.warning("Artefatos de auditoria encontrados, mas sem linhas validas para exibir.")
            return True

        frame = pd.DataFrame(rows)
        preferred_cols = [
            "case_id",
            "status",
            "search_space_mode",
            "mode",
            "fallback_to_full",
            "runtime_sec",
            "n_pockets_total",
            "n_pockets_used",
            "triage_score",
            "final_rescore",
            "selection_score_name",
            "final_score_available",
            "rescoring_applied",
            "rescoring_candidates_total",
            "rescored_candidates_count",
            "ranking_changed_after_rescoring",
            "scientific_readiness",
            "scientific_score_available",
            "summary_path",
        ]
        visible_cols = [col for col in preferred_cols if col in frame.columns]
        if visible_cols:
            st.dataframe(frame[visible_cols], use_container_width=True)
        else:
            st.dataframe(frame, use_container_width=True)
        st.caption(f"fonte_principal={source_name}")

        manifest_path = artifact_paths.get("audit_manifest")
        if manifest_path:
            try:
                manifest = legacy_reports_module().load_any_json(manifest_path)
            except (OSError, json.JSONDecodeError):
                manifest = {}
            if manifest:
                st.subheader("Manifesto de Auditoria")
                st.json(
                    {
                        "generator": manifest.get("generator"),
                        "version": manifest.get("version"),
                        "generated_at_utc": manifest.get("generated_at_utc"),
                        "policy": manifest.get("policy"),
                        "counts": manifest.get("counts"),
                        "outputs": manifest.get("outputs"),
                    }
                )

        st.subheader("Downloads (Auditaveis)")
        for key, label in (
            ("audit_json", "Baixar audit_cases_exact.json"),
            ("audit_jsonl", "Baixar audit_cases_exact.jsonl"),
            ("audit_csv", "Baixar audit_cases_exact.csv"),
            ("audit_manifest", "Baixar audit_manifest.json"),
            ("batch_report", "Baixar batch_report.json"),
        ):
            path = artifact_paths.get(key)
            if path and path.exists():
                st.download_button(
                    label,
                    data=path.read_text(encoding="utf-8"),
                    file_name=path.name,
                    mime="application/json" if path.suffix == ".json" else "text/plain",
                    key=f"audit_download_{key}_{selected_folder.name}",
                )
        return True

"""Entrada, normalização e utilitários comuns dos relatórios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .dependencies import legacy_reports_module


class ReportsDataMixin:
    """Fornece conversão, descoberta e normalização de dados de relatório."""

    @staticmethod
    def _download_csv_button(label: str, df: pd.DataFrame, filename: str, key_suffix: str) -> None:
        """Gera um botão de download para CSV a partir de um DataFrame."""

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label,
            csv_data,
            file_name=filename,
            mime="text/csv",
            key=f"download_csv_{key_suffix}",
        )

    @staticmethod
    def _load_uploaded_json(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> dict[str, Any]:
        """Carrega JSON enviado via upload na interface."""

        return json.loads(uploaded_file.getvalue().decode("utf-8"))

    @staticmethod
    def _load_uploaded_jsonl(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> list[dict[str, Any]]:
        """Carrega JSONL enviado via upload na interface."""

        content = uploaded_file.getvalue().decode("utf-8").splitlines()
        records: list[dict[str, Any]] = []
        for line in content:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    @staticmethod
    def _download_json_payload(label: str, payload: dict[str, Any], filename: str) -> None:
        """Gera um botão de download para JSON em memória."""

        st.download_button(
            label,
            json.dumps(payload, indent=2, ensure_ascii=False),
            file_name=filename,
            mime="application/json",
        )

    @staticmethod
    def _download_jsonl_payload(label: str, records: list[dict[str, Any]], filename: str) -> None:
        """Gera um botão de download para JSONL em memória."""

        content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        st.download_button(
            label,
            content,
            file_name=filename,
            mime="application/jsonl",
        )

    @staticmethod
    def _summary_value(data: dict[str, Any], keys: list[str]) -> Any:
        """Seleciona o primeiro valor disponível dentre chaves candidatas."""

        return next((data.get(key) for key in keys if data.get(key) is not None), None)

    @staticmethod
    def _resolve_report_path(base_dir: Path | None, raw_path: str | None) -> Path | None:
        """Resolve caminhos relativos usando o diretório base do relatório."""

        if not raw_path:
            return None
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate
        if base_dir:
            return base_dir / candidate
        return candidate

    def _build_summary_rows(
        self,
        summary_data: dict[str, Any],
        fallback_scores: dict[str, float | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Monta linhas para a tabela resumo a partir do summary.json."""

        fallback_scores = fallback_scores or {}
        return [
            {
                "Campo": "pockets_detected",
                "Valor": self._summary_value(
                    summary_data,
                    ["pockets_detected", "n_pockets_total", "pockets_total", "n_pockets_detected"],
                ),
            },
            {
                "Campo": "pockets_used",
                "Valor": self._summary_value(
                    summary_data,
                    ["pockets_used", "n_pockets_used"],
                ),
            },
            {
                "Campo": "best_cheap",
                "Valor": self._summary_value(
                    summary_data,
                    ["triage_score", "best_score_cheap", "best_score", "best"],
                )
                or fallback_scores.get("best_cheap"),
            },
            {
                "Campo": "best_expensive",
                "Valor": self._summary_value(
                    summary_data,
                    ["final_rescore", "rescore_proxy", "best_score_expensive"],
                )
                or fallback_scores.get("best_expensive"),
            },
            {
                "Campo": "total_s",
                "Valor": self._summary_value(
                    summary_data,
                    ["total_runtime_sec", "runtime_sec", "total_s", "elapsed_s", "elapsed_seconds", "elapsed"],
                ),
            },
            {
                "Campo": "expensive_ran_count",
                "Valor": self._summary_value(
                    summary_data,
                    ["expensive_ran_count"],
                ),
            },
        ]

    @staticmethod
    def _render_score_semantics_caption(summary_data: dict[str, Any]) -> None:
        """Mostra uma legenda curta e honesta sobre o papel dos scores."""

        if not isinstance(summary_data, dict):
            return
        if summary_data.get("score_semantics_version") is None:
            return
        primary = summary_data.get("selection_score_name") or summary_data.get("optimization_score_name") or "score_cheap"
        final_available = bool(summary_data.get("final_score_available", False))
        final_text = "nao disponivel nesta execucao" if not final_available else "disponivel"
        st.caption(
            f"score principal atual: triage proxy (`{primary}`) | "
            "rescoring final seletivo (`final_rescore`) | "
            f"score final cientifico: {final_text}"
        )

    @staticmethod
    def _guess_metrics_index(options: list[str], tokens: list[str]) -> int:
        """Sugere índice padrão para arquivos de métricas baseado em tokens."""

        for idx, name in enumerate(options, start=1):
            lowered = name.lower()
            if any(token in lowered for token in tokens):
                return idx
        return 0

    @staticmethod
    def _extract_compare_block(report_data: dict[str, Any], label: str) -> dict[str, Any] | None:
        """Busca o bloco de comparação correspondente ao rótulo fornecido."""

        for container in (report_data, report_data.get("runs"), report_data.get("comparison")):
            if isinstance(container, dict) and label in container and isinstance(container[label], dict):
                return container[label]
        return None

    def _load_summary_data(self, summary_path: Path | None, fallback: dict[str, Any]) -> dict[str, Any]:
        """Carrega summary.json, caindo no fallback em caso de erro."""

        if summary_path and summary_path.exists():
            try:
                return legacy_reports_module().load_any_json(summary_path)
            except (OSError, json.JSONDecodeError):
                return fallback
        return fallback

    @staticmethod
    def _buscar_jsonl_na_pasta(base_dir: Path) -> list[Path]:
        """Lista JSONL disponíveis na pasta e subpastas."""

        candidatos = sorted(base_dir.glob("*.jsonl"))
        candidatos += sorted(base_dir.rglob("metrics.jsonl"))
        vistos: set[Path] = set()
        unicos: list[Path] = []
        for candidato in candidatos:
            if candidato in vistos:
                continue
            vistos.add(candidato)
            unicos.append(candidato)
        return unicos

    @staticmethod
    def _ultimo_valor(lista: list[Any]) -> Any:
        """Retorna o último valor não nulo da lista."""

        for valor in reversed(lista):
            if valor is not None:
                return valor
        return None

    @staticmethod
    def _resumo_gap(series: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
        """Monta resumo do gap com base em séries e summary."""

        n_eval = ReportsDataMixin._ultimo_valor(series.get("n_eval_total", []))
        n_filtered = ReportsDataMixin._ultimo_valor(series.get("n_filtered", []))
        n_selected = ReportsDataMixin._ultimo_valor(series.get("n_selected", []))
        runtime = ReportsDataMixin._ultimo_valor(series.get("runtime_s", []))
        expensive = ReportsDataMixin._ultimo_valor(series.get("expensive_ran", []))
        best_cheap = ReportsDataMixin._ultimo_valor(series.get("best_cheap", []))
        best_expensive = ReportsDataMixin._ultimo_valor(series.get("best_expensive", []))

        if n_eval is None:
            n_eval = ReportsDataMixin._summary_value(summary, ["total_n_eval", "n_eval_total", "n_eval", "evals", "evaluations"])
        if n_selected is None:
            n_selected = ReportsDataMixin._summary_value(summary, ["n_pockets_used", "pockets_used", "n_selected"])

        if runtime is None:
            runtime = ReportsDataMixin._summary_value(
                summary,
                ["runtime_total_s", "total_runtime_sec", "runtime_sec", "total_s", "elapsed_s", "elapsed_seconds", "elapsed"],
            )
        if best_cheap is None:
            best_cheap = ReportsDataMixin._summary_value(
                summary,
                ["best_over_pockets_cheap", "triage_score", "best_score_cheap", "best_score", "best"],
            )
        if best_expensive is None:
            best_expensive = ReportsDataMixin._summary_value(
                summary,
                ["best_over_pockets_expensive", "final_rescore", "rescore_proxy", "best_score_expensive", "best_expensive"],
            )

        kept_ratio = None
        try:
            if n_eval is not None and n_selected is not None:
                kept_ratio = float(n_selected) / max(float(n_eval), 1.0)
        except (TypeError, ValueError):
            kept_ratio = None

        return {
            "n_eval_total": n_eval,
            "n_filtered": n_filtered,
            "kept_ratio_final": kept_ratio,
            "runtime_total_s": runtime,
            "expensive_ran": expensive,
            "best_cheap": best_cheap,
            "best_expensive": best_expensive,
        }

    @staticmethod
    def _render_png(path: Path, legenda: str) -> None:
        """Exibe um PNG no Streamlit."""

        if path.exists():
            st.image(str(path), caption=legenda, use_container_width=True)

    def _render_warning_missing(self, series: dict[str, Any]) -> None:
        """Mostra aviso resumido sobre chaves ausentes."""

        faltas = series.get("missing")
        if not isinstance(faltas, dict):
            return
        faltantes = {chave: count for chave, count in faltas.items() if count}
        if faltantes:
            st.caption(f"Campos ausentes em parte das métricas: {faltantes}")

    @staticmethod
    def _merge_summary_result(summary: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """Mescla dados de summary e result garantindo chaves úteis no topo."""

        merged: dict[str, Any] = {}
        merged.update(result or {})
        merged.update(summary or {})
        timing = summary.get("timing") if isinstance(summary.get("timing"), dict) else {}
        if not timing and isinstance(result.get("timing"), dict):
            timing = result.get("timing")
        if isinstance(timing, dict):
            merged.setdefault("total_s", timing.get("total_s"))
            merged.setdefault("elapsed_s", timing.get("elapsed_s"))
        return merged

    @staticmethod
    def _has_time_series(records: list[dict[str, Any]] | None) -> bool:
        """Indica se há série temporal real (mais de um ponto)."""

        if not records:
            return False
        if len(records) <= 1:
            return False
        series = legacy_reports_module().extract_series(records)
        return len(set(series.get("iter", []))) > 1

    @staticmethod
    def _select_series_records(
        metrics_records: list[dict[str, Any]] | None,
        metrics_timeseries: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """Seleciona o arquivo com série temporal preferencial."""

        return metrics_timeseries or metrics_records

    @staticmethod
    def _is_reduced_aggregate_fallback(summary: dict[str, Any]) -> bool:
        return (
            summary.get("mode") == "reduced_aggregate"
            and bool(summary.get("fallback_to_full"))
            and not bool(summary.get("per_pocket_results") or [])
        )

    @staticmethod
    def _resolve_compare_fallback_metrics(
        summary: dict[str, Any],
        compare_block: dict[str, Any] | None,
    ) -> dict[str, Any]:
        resolved: dict[str, Any] = dict(summary)
        if not compare_block:
            return resolved
        for dst, keys in {
            "n_eval_total": ["reduced_total_n_eval", "total_n_eval", "n_eval_total"],
            "runtime_total_s": ["reduced_total_runtime_sec", "total_runtime_sec", "runtime_total_s"],
            "best_cheap": ["reduced_best_over_pockets_cheap", "best_over_pockets_cheap", "triage_score", "best_score_cheap"],
        }.items():
            for key in keys:
                if compare_block.get(key) is not None:
                    resolved[dst] = compare_block.get(key)
                    break
        return resolved

"""Carregadores tolerantes para relatórios (PT-BR)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, Literal


_CANDIDATAS_SERIES: dict[str, list[str]] = {
    "iter": ["iter", "generation", "gen", "step"],
    "best_cheap": [
        "best_over_pockets_cheap",
        "triage_score",
        "best_score_cheap",
        "best_score",
        "best_cheap",
        "best",
        "pocket.best_score_cheap",
    ],
    "best_expensive": [
        "best_over_pockets_expensive",
        "final_rescore",
        "rescore_proxy",
        "best_score_expensive",
        "best_expensive",
        "best_energy",
        "vina",
        "pocket.best_score_expensive",
    ],
    "n_eval_total": [
        "total_n_eval",
        "n_eval_total",
        "n_eval_cumulative",
        "n_eval",
        "eval_total",
        "n_scored",
        "cheap_evals",
        "pocket.n_eval_total",
    ],
    "n_filtered": ["n_filtered", "filtered", "discarded", "expensive_skipped"],
    "n_selected": ["n_selected", "selected", "kept", "selected_pockets", "n_pockets_used"],
    "runtime_s": [
        "runtime.total_runtime_sec",
        "runtime.search_time_total_sec",
        "runtime.scan_time_sec",
        "runtime.pocketing_time_sec",
        "runtime_total_s",
        "total_runtime_sec",
        "runtime_sec",
        "runtime_s",
        "time_s",
        "elapsed_s",
        "total_s",
        "search_s",
        "scan.time_sec",
    ],
    "expensive_ran": ["expensive_ran", "expensive_ran_count", "n_expensive", "expensive_calls"],
    "pocket_id": ["pocket_id", "pocket", "pocket_index"],
    "pocket_rank": ["pocket_rank", "rank", "pocket_index"],
}
RunKind = Literal["full", "reduced", "unknown"]

SCIENTIFIC_ALIASES: dict[str, list[str]] = {
    "energy_evina": [
        "evina",
        "best_score_expensive",
        "energia_realista",
        "vina_score",
        "final_energy",
        "final_rescore",
        "best_over_pockets_expensive",
        "rescore_proxy",
    ],
    "runtime_sec": [
        "runtime_sec",
        "total_runtime_sec",
        "elapsed_sec",
        "tempo_execucao",
        "runtime_total_s",
        "runtime_s",
        "elapsed_s",
        "total_s",
    ],
    "n_eval": [
        "n_eval",
        "total_n_eval",
        "n_eval_total",
        "eval_total",
        "n_scored",
    ],
    "generation": ["generation", "iter", "step", "gen"],
    "algorithm": [
        "algorithm",
        "search_engine",
        "search_engine_name",
        "search_strategy",
        "search_algorithm_family",
        "metodo",
    ],
    "complex_id": [
        "complex_id",
        "pdb_id",
        "case_id",
        "input_id",
        "protein",
        "receptor_name",
    ],
    "round_index": ["experimental_round", "round_index", "rodada", "run_round"],
    "n_rounds": ["n_rounds", "total_rounds", "rodadas_total"],
}


def normalize_search_space_mode(mode: Any) -> str | None:
    """Normaliza search_space_mode para {full,reduced} no domínio interno."""

    if mode is None:
        return None
    lowered = str(mode).strip().lower()
    if lowered == "global":
        return "full"
    if lowered == "pockets":
        return "reduced"
    if lowered in {"full", "reduced"}:
        return lowered
    return None


@dataclass(frozen=True)
class ReportRun:
    """Describe os arquivos associados a uma execução."""

    run_dir: Path
    kind: RunKind
    summary_path: Path | None
    result_path: Path | None
    config_path: Path | None
    metrics_path: Path | None
    metrics_timeseries_path: Path | None
    run_id: str | None = None

    def label(self) -> str:
        """Rótulo humano para seleção na UI."""

        return self.run_id or self.run_dir.name


def _json_merge_priority(path: Path) -> tuple[int, str]:
    """Merge canonical final artifacts after auxiliary/audit snapshots."""

    name = path.name.lower()
    if ".pre_fallback." in name:
        priority = -1
    elif name == "config.json":
        priority = 1
    elif name == "result.json":
        priority = 2
    elif name == "summary.json":
        priority = 3
    else:
        priority = 0
    return priority, name


def _has_canonical_run_ancestor(run_dir: Path, root_dir: Path) -> bool:
    """Avoid counting pocket/fallback subdirectories as independent runs."""

    current = run_dir.parent
    while _is_relative_to(current, root_dir):
        if (current / "summary.json").is_file() or (current / "result.json").is_file():
            return True
        if current == root_dir:
            break
        current = current.parent
    return False


def find_report_runs(root_dir: Path) -> list[ReportRun]:
    """Varre uma pasta e encontra execuções single (Full/Reduced) com base no conteúdo."""

    if not root_dir.exists():
        return []

    json_paths = list(root_dir.rglob("*.json"))
    if not json_paths:
        return []

    dirs = {path.parent for path in json_paths}
    if any(path.parent == root_dir for path in json_paths):
        dirs.add(root_dir)

    runs: list[ReportRun] = []
    for run_dir in sorted(dirs):
        if _has_canonical_run_ancestor(run_dir, root_dir):
            continue
        run = _build_report_run(run_dir)
        if run is not None:
            runs.append(run)
    return runs


def pair_full_reduced(runs: Iterable[ReportRun]) -> tuple[ReportRun | None, ReportRun | None]:
    """Seleciona um par Full/Reduced entre as execuções encontradas."""

    full = None
    reduced = None
    for run in sorted(runs, key=lambda item: item.label()):
        if run.kind == "full" and full is None:
            full = run
        elif run.kind == "reduced" and reduced is None:
            reduced = run
    return full, reduced


def _build_report_run(run_dir: Path) -> ReportRun | None:
    json_paths = sorted(run_dir.glob("*.json"), key=_json_merge_priority)
    if not json_paths:
        return None

    summary_path = None
    result_path = None
    config_path = None
    payloads: list[dict[str, Any]] = []
    run_id = None

    for path in json_paths:
        try:
            payload = load_any_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        payloads.append(payload)
        role = _classify_json_payload(payload)
        if role == "summary":
            summary_path = path
        elif role == "result":
            result_path = path
            run_id = _safe_str(payload.get("run_id")) or run_id
        elif role == "config":
            config_path = path

        # PT-BR: compatibilidade com artefatos legados onde o result.json não
        # tem campos suficientes para classificação estrita.
        name = path.name.lower()
        if name == "summary.json" and summary_path is None:
            summary_path = path
        elif name == "result.json" and result_path is None:
            result_path = path
        elif name == "config.json" and config_path is None:
            config_path = path

    canonical_summary = run_dir / "summary.json"
    canonical_result = run_dir / "result.json"
    canonical_config = run_dir / "config.json"
    if canonical_summary.exists():
        summary_path = canonical_summary
    if canonical_result.exists():
        result_path = canonical_result
    if canonical_config.exists():
        config_path = canonical_config

    kind = _infer_run_kind(payloads)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_timeseries_path = run_dir / "metrics.timeseries.jsonl"
    return ReportRun(
        run_dir=run_dir,
        kind=kind,
        summary_path=summary_path,
        result_path=result_path,
        config_path=config_path,
        metrics_path=metrics_path if metrics_path.exists() else None,
        metrics_timeseries_path=metrics_timeseries_path if metrics_timeseries_path.exists() else None,
        run_id=run_id,
    )


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _classify_json_payload(payload: dict[str, Any]) -> str:
    if _is_summary_payload(payload):
        return "summary"
    if _is_result_payload(payload):
        return "result"
    if _is_config_payload(payload):
        return "config"
    return "unknown"


def _is_summary_payload(payload: dict[str, Any]) -> bool:
    if payload.get("mode") == "reduced_aggregate":
        return bool(payload.get("per_pocket_results") is not None and payload.get("total_n_eval") is not None)
    return bool(
        payload.get("run_id") is not None
        and payload.get("best_cheap_by_pocket") is not None
        and payload.get("n_eval_total") is not None
        and payload.get("mode") is not None
    )


def _is_result_payload(payload: dict[str, Any]) -> bool:
    has_primary_score = payload.get("best_score_cheap") is not None or payload.get("triage_score") is not None
    return bool(
        isinstance(payload.get("timing"), dict)
        and has_primary_score
        and payload.get("mode") is not None
    )

def _is_config_payload(payload: dict[str, Any]) -> bool:
    return bool(
        payload.get("seed") is not None
        and payload.get("generations") is not None
        and payload.get("pop_size") is not None
        and (_extract_flag(payload, "full_search") is not None or _extract_flag(payload, "search_space_mode") is not None)
    )


def _extract_flag(payload: dict[str, Any], key: str) -> Any:
    if key in payload:
        return payload.get(key)
    for container_key in ("config", "cfg", "settings", "options"):
        nested = payload.get(container_key)
        if isinstance(nested, dict) and key in nested:
            return nested.get(key)
    return None


def _infer_run_kind(payloads: Iterable[dict[str, Any]]) -> RunKind:
    full_signal = False
    reduced_signal = False
    reduced_fallback_signal = False
    for payload in payloads:
        full_search = _extract_flag(payload, "full_search")
        if full_search is True:
            full_signal = True
        elif full_search is False:
            reduced_signal = True
        search_space_mode = normalize_search_space_mode(_extract_flag(payload, "search_space_mode"))
        if search_space_mode == "full":
            full_signal = True
        elif search_space_mode == "reduced":
            reduced_signal = True
            if bool(payload.get("fallback_executed") or payload.get("fallback_to_full")):
                reduced_fallback_signal = True
    if reduced_fallback_signal:
        return "reduced"
    if full_signal:
        return "full"
    if reduced_signal:
        return "reduced"
    return "unknown"


def _to_str_list(value: Any) -> list[str]:
    """Normaliza valor para lista de strings preservando compatibilidade."""

    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _rejected_pocket_ids(value: Any) -> list[str]:
    """Extrai ids de bolsões rejeitados de formatos legados/canônicos."""

    if not isinstance(value, list):
        return []

    pocket_ids: list[str] = []
    for item in value:
        if isinstance(item, dict):
            pocket_id = item.get("pocket_id") or item.get("id")
            if pocket_id is not None:
                pocket_ids.append(str(pocket_id))
        elif item is not None:
            pocket_ids.append(str(item))
    return pocket_ids


def resolve_reduced_pocket_collections(payload: dict[str, Any]) -> dict[str, list[str]]:
    """Resolve coleções de bolsões com fallback compatível entre schemas.

    Ordem canônica para coleção final executada:
    `executed_pockets` -> `feasible_pockets` -> `accepted_pockets` -> `selected_pockets`.
    """

    candidate = _to_str_list(payload.get("candidate_pockets"))
    selected_after_scan = _to_str_list(payload.get("selected_after_scan_pockets"))
    if not selected_after_scan:
        selected_after_scan = _to_str_list(payload.get("selected_pre_filter_pockets"))

    accepted = _to_str_list(payload.get("accepted_pockets"))
    feasible = _to_str_list(payload.get("feasible_pockets"))
    if not feasible:
        feasible = list(accepted)

    selected_compat = _to_str_list(payload.get("selected_pockets"))
    executed = _to_str_list(payload.get("executed_pockets"))
    if not executed:
        executed = list(feasible) or list(accepted) or list(selected_compat)
    if not selected_compat:
        selected_compat = list(executed)

    return {
        "candidate_pockets": candidate,
        "selected_after_scan_pockets": selected_after_scan,
        "accepted_pockets": accepted,
        "feasible_pockets": feasible,
        "executed_pockets": executed,
        "selected_pockets_compat": selected_compat,
        "rejected_pockets": _rejected_pocket_ids(payload.get("rejected_pockets")),
    }


def load_any_json(path: str | Path) -> dict[str, Any]:
    """Carrega qualquer JSON e garante retorno em dicionário."""

    raw = Path(path).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if isinstance(payload, dict):
        return payload
    return {"data": payload}


def find_matching_jsonl(json_path: str | Path) -> Path | None:
    """Localiza um JSONL compatível com base em heurísticas simples."""

    path = Path(json_path)
    candidate = path.with_suffix(".jsonl")
    if candidate.exists():
        return candidate

    try:
        payload = load_any_json(path)
    except (OSError, json.JSONDecodeError):
        payload = {}

    metrics_path = payload.get("metrics_path")
    if metrics_path:
        metrics_candidate = Path(metrics_path).expanduser()
        if not metrics_candidate.is_absolute():
            metrics_candidate = path.parent / metrics_candidate
        if metrics_candidate.exists():
            return metrics_candidate

    out_dir = payload.get("outdir") or payload.get("out_dir") or payload.get("output_dir")
    if out_dir:
        out_path = Path(out_dir).expanduser()
        if not out_path.is_absolute():
            out_path = path.parent / out_path
        metrics_jsonl = out_path / "metrics.jsonl"
        if metrics_jsonl.exists():
            return metrics_jsonl
    return None


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Carrega JSONL tolerante, ignorando linhas inválidas."""

    records: list[dict[str, Any]] = []
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return records
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def load_scientific_runs(root_dir: str | Path) -> list[dict[str, Any]]:
    """Carrega execucoes cientificas de qualquer JSON/JSONL valido sob uma pasta.

    A funcao agrupa artefatos por diretorio, infere campos por aliases e evita
    depender de nomes fixos. Campos ausentes permanecem como ``None`` para a GUI
    exibir avisos amigaveis.
    """

    root = Path(root_dir).expanduser()
    if not root.exists():
        return []

    run_dirs: set[Path] = set()
    for path in root.rglob("*"):
        if path.suffix.lower() in {".json", ".jsonl"}:
            run_dirs.add(path.parent)
    if not run_dirs:
        return []

    runs: list[dict[str, Any]] = []
    for run_dir in sorted(run_dirs):
        if _has_canonical_run_ancestor(run_dir, root):
            continue
        run = _load_scientific_run_dir(run_dir, root)
        if run is not None:
            runs.append(run)
    return runs


def _load_scientific_run_dir(run_dir: Path, root_dir: Path) -> dict[str, Any] | None:
    if _is_scientific_auxiliary_dir(run_dir):
        return None

    json_payloads: list[dict[str, Any]] = []
    json_paths: list[Path] = []
    jsonl_payloads: dict[str, list[dict[str, Any]]] = {}

    for path in sorted(run_dir.glob("*.json"), key=_json_merge_priority):
        try:
            payload = load_any_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        json_payloads.append(payload)
        json_paths.append(path)

    for path in sorted(run_dir.glob("*.jsonl")):
        records = load_jsonl(path)
        if records:
            jsonl_payloads[path.name] = records

    if not json_payloads and not jsonl_payloads:
        return None
    if json_paths and not jsonl_payloads and all(path.name == "experiment_status.json" for path in json_paths):
        return None

    merged = _merge_payloads(json_payloads)
    timeseries_records = _select_timeseries_records(jsonl_payloads)
    metric_records = _select_metric_records(jsonl_payloads)
    series = extract_series(timeseries_records or metric_records) if (timeseries_records or metric_records) else {}

    kind = infer_scientific_run_kind(merged)
    fallback_executed = _safe_bool(_extract_alias(merged, ["fallback_executed", "fallback_to_full"]))
    if fallback_executed:
        kind = "fallback_full"

    algorithm = _safe_str(_extract_alias(merged, SCIENTIFIC_ALIASES["algorithm"])) or _infer_algorithm_from_path(run_dir)
    complex_id = (
        _safe_str(_extract_alias(merged, SCIENTIFIC_ALIASES["complex_id"]))
        or _infer_complex_from_context(run_dir, root_dir)
        or _infer_complex_from_path(run_dir)
    )
    energy = _extract_scientific_energy(merged)
    runtime = _safe_float(_extract_alias(merged, SCIENTIFIC_ALIASES["runtime_sec"]))
    n_eval = _safe_int(_extract_alias(merged, SCIENTIFIC_ALIASES["n_eval"]))
    round_index = _safe_int(_extract_alias(merged, SCIENTIFIC_ALIASES["round_index"])) or _infer_round_from_path(run_dir)
    n_rounds = _safe_int(_extract_alias(merged, SCIENTIFIC_ALIASES["n_rounds"]))

    return {
        "run_dir": str(run_dir),
        "run_dir_relative": str(run_dir.relative_to(root_dir)).replace("\\", "/")
        if _is_relative_to(run_dir, root_dir)
        else str(run_dir),
        "json_files": [str(path) for path in json_paths],
        "jsonl_files": [str(run_dir / name) for name in jsonl_payloads],
        "complex_id": complex_id,
        "algorithm": algorithm,
        "mode": kind,
        "energy_evina": energy,
        "final_rescore_score": _safe_float(_extract_alias(merged, ["final_rescore", "best_score_expensive"])),
        "runtime_sec": runtime,
        "n_eval": n_eval,
        "round_index": round_index,
        "n_rounds": n_rounds,
        "fallback_executed": fallback_executed,
        "fallback_reason": _safe_str(merged.get("fallback_reason")),
        "executed_mode": _safe_str(merged.get("executed_mode")),
        "search_space_mode": _safe_str(_extract_flag(merged, "search_space_mode")),
        "summary": merged,
        "metrics": metric_records,
        "timeseries": timeseries_records,
        "series": series,
        "warnings": _scientific_run_warnings(energy, runtime, n_eval, algorithm, complex_id),
    }


def _is_scientific_auxiliary_dir(run_dir: Path) -> bool:
    """Evita tratar artefatos/depuracao como execucoes independentes."""

    name = run_dir.name.lower()
    if name not in {"artifacts", "artifact", "debug", "logs"}:
        return False
    run_markers = {"summary.json", "result.json", "config.json", "metrics.jsonl", "metrics.timeseries.jsonl"}
    return not any((run_dir / marker).exists() for marker in run_markers)


def infer_scientific_run_kind(payload: dict[str, Any]) -> str:
    """Infere full/reduced/fallback pelo conteudo, com tolerancia a aliases."""

    fallback = payload.get("fallback_executed")
    if fallback is None:
        fallback = payload.get("fallback_to_full")
    if bool(fallback):
        return "fallback_full"

    executed_mode = normalize_search_space_mode(payload.get("executed_mode"))
    search_space_mode = normalize_search_space_mode(_extract_flag(payload, "search_space_mode"))
    mode = normalize_search_space_mode(payload.get("mode"))
    full_search = _extract_flag(payload, "full_search")
    if executed_mode in {"full", "reduced"}:
        return executed_mode
    if search_space_mode in {"full", "reduced"}:
        return search_space_mode
    if mode in {"full", "reduced"}:
        return mode
    if full_search is True:
        return "full"
    if full_search is False:
        return "reduced"
    return "unknown"


def _merge_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        merged.update(payload)
    return merged


def _select_timeseries_records(jsonl_payloads: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    for name, records in jsonl_payloads.items():
        if "timeseries" in name.lower():
            return records
    return []


def _select_metric_records(jsonl_payloads: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    for name, records in jsonl_payloads.items():
        if "timeseries" not in name.lower():
            return records
    return []


def _extract_alias(payload: dict[str, Any], aliases: list[str]) -> Any:
    for alias in aliases:
        value = _buscar_valor(payload, [alias])
        if value is not None:
            return value
    config_subset = payload.get("config_resolved_subset")
    if isinstance(config_subset, dict):
        for alias in aliases:
            value = _buscar_valor(config_subset, [alias])
            if value is not None:
                return value
    return None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _extract_scientific_energy(payload: dict[str, Any]) -> float | None:
    """Retorna energia Evina em kcal/mol, onde menor e melhor."""

    explicit_energy = _safe_float(
        _extract_alias(
            payload,
            [
                "final_rescore_energy",
                "evina_energy",
                "energy_evina",
                "evina",
                "energia_realista",
                "vina_score",
                "final_energy",
            ],
        )
    )
    if explicit_energy is not None:
        return explicit_energy

    for key in ("rescoring_selected_candidates", "selected_candidates", "rescoring_cheap_top5_candidates"):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        energies = []
        for row in rows:
            if isinstance(row, dict):
                value = _safe_float(row.get("final_rescore_energy"))
                if value is not None:
                    energies.append(value)
        if energies:
            return min(energies)

    final_rescore = _safe_float(_extract_alias(payload, ["final_rescore", "best_score_expensive", "best_over_pockets_expensive", "rescore_proxy"]))
    if final_rescore is None:
        return None
    return -float(final_rescore)


def _infer_algorithm_from_path(run_dir: Path) -> str | None:
    parts = [part.lower() for part in run_dir.parts]
    for candidate in ("hybrid", "abcgavgos", "abc_ga_vgos", "pso", "ag"):
        if candidate in parts:
            return "ABC+AG+VGOS" if candidate in {"hybrid", "abcgavgos", "abc_ga_vgos"} else candidate.upper()
    return None


def _infer_round_from_path(run_dir: Path) -> int | None:
    for part in reversed(run_dir.parts):
        lower = part.lower()
        if lower.startswith("rodada_"):
            value = lower.replace("rodada_", "", 1)
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _infer_complex_from_path(run_dir: Path) -> str | None:
    for part in reversed(run_dir.parts):
        lower = part.lower()
        if lower.startswith("case_") or lower.endswith(".pdb"):
            return part.replace("_receptor", "").replace("_peptide", "")
    return run_dir.name


def _infer_complex_from_context(run_dir: Path, root_dir: Path) -> str | None:
    candidates: list[str] = []
    for path in (root_dir, root_dir.parent):
        if path.name:
            candidates.append(path.name)
    if _is_relative_to(run_dir, root_dir):
        candidates.extend(run_dir.relative_to(root_dir).parts)
    candidates.extend(run_dir.parts)

    for candidate in candidates:
        if _looks_like_pdb_or_protein_id(candidate):
            return candidate
    return None


def _looks_like_pdb_or_protein_id(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return False
    if re.fullmatch(r"\d[A-Za-z0-9]{3}", text):
        return True
    if re.fullmatch(r"[A-Za-z]{1,4}\d{0,4}", text) and not text.lower().startswith(("run", "rodada")):
        return True
    return False


def _scientific_run_warnings(
    energy: float | None,
    runtime: float | None,
    n_eval: int | None,
    algorithm: str | None,
    complex_id: str | None,
) -> list[str]:
    warnings: list[str] = []
    if energy is None:
        warnings.append("Energia Evina/final ausente.")
    if runtime is None:
        warnings.append("Tempo de execucao ausente.")
    if n_eval is None:
        warnings.append("Numero de avaliacoes ausente.")
    if not algorithm:
        warnings.append("Algoritmo nao identificado.")
    if not complex_id:
        warnings.append("Complexo/proteina nao identificado.")
    return warnings


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def extract_series(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Extrai séries padronizadas de eventos JSONL."""

    series: dict[str, list[Any]] = {key: [] for key in _CANDIDATAS_SERIES}
    faltas: dict[str, int] = {key: 0 for key in _CANDIDATAS_SERIES if key != "iter"}

    for idx, evento in enumerate(events):
        iter_valor = _buscar_valor(evento, _CANDIDATAS_SERIES["iter"])
        if iter_valor is None:
            iter_valor = idx
        series["iter"].append(iter_valor)

        for nome_serie, chaves in _CANDIDATAS_SERIES.items():
            if nome_serie == "iter":
                continue
            valor = _buscar_valor(evento, chaves)
            if valor is None and _tem_nome_valor(evento, chaves):
                valor = evento.get("value")
            if valor is None:
                faltas[nome_serie] += 1
            series[nome_serie].append(valor)

    series["missing"] = faltas
    return series


def _buscar_valor(evento: dict[str, Any], chaves: list[str]) -> Any:
    for chave in chaves:
        if evento.get(chave) is not None:
            return evento.get(chave)
        if "." in chave:
            current: Any = evento
            found = True
            for part in chave.split("."):
                if not isinstance(current, dict) or part not in current:
                    found = False
                    break
                current = current.get(part)
            if found and current is not None:
                return current
    extras = evento.get("extras")
    if isinstance(extras, dict):
        for chave in chaves:
            if extras.get(chave) is not None:
                return extras.get(chave)
    return None


def _tem_nome_valor(evento: dict[str, Any], chaves: list[str]) -> bool:
    nome = evento.get("name")
    return nome is not None and nome in chaves

"""Visual reports for the AG vs PSO vs ABC+AG+VGOS protocol."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


ENGINE_LABELS = {
    "ag": "AG",
    "pso": "PSO",
    "hybrid": "ABC+AG+VGOS",
}


def build_algorithm_protocol_visual_report(run_dir: str | Path) -> dict[str, Any]:
    """Generate figures and an HTML dashboard from a protocol run directory."""

    root = Path(run_dir)
    rows = load_algorithm_run_table(root / "run_table.csv")
    if not rows:
        return {"generated": False, "reason": "empty_run_table", "figures": []}

    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize_algorithm_groups(rows)
    winners = compute_winner_counts(rows)

    figures: list[Path] = []
    plotters = [
        ("wins_by_engine.png", lambda path: plot_protocol_wins(winners, path)),
        ("final_rescore_boxplot.png", lambda path: plot_protocol_final_rescore(rows, path)),
        ("runtime_by_engine.png", lambda path: plot_protocol_runtime(rows, path)),
        ("fallback_by_arm.png", lambda path: plot_protocol_fallback(rows, path)),
        ("summary_table.png", lambda path: plot_protocol_summary_table(summary_rows, path)),
    ]
    for filename, plotter in plotters:
        target = figures_dir / filename
        if plotter(target):
            figures.append(target)

    write_algorithm_visual_summary(root / "visual_summary.json", summary_rows, winners, figures)
    write_algorithm_visual_report_md(root / "visual_report.md", rows, summary_rows, winners, figures)
    write_algorithm_dashboard_html(root / "dashboard.html", rows, summary_rows, winners, figures)
    _append_visual_section_to_report(root / "report.md", figures)
    return {
        "generated": True,
        "figures": [str(path) for path in figures],
        "dashboard_html": str(root / "dashboard.html"),
        "visual_report_md": str(root / "visual_report.md"),
    }


def load_algorithm_run_table(path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def summarize_algorithm_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        groups.setdefault((str(row.get("engine")), str(row.get("confidence_arm"))), []).append(row)

    out: list[dict[str, Any]] = []
    for (engine, arm), group_rows in sorted(groups.items()):
        out.append(
            {
                "engine": engine,
                "confidence_arm": arm,
                "n_runs": len(group_rows),
                "avg_final_rescore": _mean(_float(row.get("final_rescore")) for row in group_rows),
                "avg_best_score_cheap": _mean(_float(row.get("best_score_cheap")) for row in group_rows),
                "avg_runtime_sec": _mean(_float(row.get("runtime_sec")) for row in group_rows),
                "fallback_recommended_rate": _mean(
                    1.0 if _bool(row.get("fallback_recommended")) else 0.0 for row in group_rows
                ),
                "fallback_executed_rate": _mean(
                    1.0 if _bool(row.get("fallback_executed")) else 0.0 for row in group_rows
                ),
                "avg_uncertain_pockets_count": _mean(
                    _float(row.get("uncertain_pockets_count")) for row in group_rows
                ),
            }
        )
    return out


def compute_winner_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        seed = _int(row.get("seed"))
        if seed is None:
            continue
        grouped.setdefault(
            (str(row.get("case_id")), seed, str(row.get("confidence_arm"))),
            [],
        ).append(row)

    winners: dict[str, int] = {}
    for group_rows in grouped.values():
        present = {str(row.get("engine")) for row in group_rows}
        if not {"ag", "pso", "hybrid"}.issubset(present):
            continue
        ranked = sorted(
            group_rows,
            key=lambda row: (
                _float(row.get("final_rescore")) if _float(row.get("final_rescore")) is not None else float("-inf"),
                _float(row.get("best_score_cheap")) if _float(row.get("best_score_cheap")) is not None else float("-inf"),
                -(_float(row.get("runtime_sec")) if _float(row.get("runtime_sec")) is not None else float("inf")),
            ),
            reverse=True,
        )
        winner = str(ranked[0].get("engine"))
        winners[winner] = winners.get(winner, 0) + 1
    return winners


def plot_protocol_wins(winners: dict[str, int], out_png: str | Path) -> bool:
    if not winners:
        return False
    plt = _pyplot()
    engines = _ordered_engines(winners)
    values = [int(winners.get(engine, 0)) for engine in engines]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([_engine_label(engine) for engine in engines], values, color=_engine_colors(engines))
    _label_bars(ax, bars, values, "{:.0f}")
    ax.set_title("Vitorias por caso, seed e politica")
    ax.set_ylabel("Numero de vitorias")
    ax.set_xlabel("Algoritmo")
    _save(fig, out_png)
    return True


def plot_protocol_final_rescore(rows: list[dict[str, Any]], out_png: str | Path) -> bool:
    groups = _group_values(rows, "final_rescore")
    if not groups:
        return False
    plt = _pyplot()
    labels = [label for label, _values in groups]
    values = [vals for _label, vals in groups]
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.9), 5))
    ax.boxplot(values, labels=labels, showmeans=True)
    ax.set_title("Distribuicao do final_rescore")
    ax.set_ylabel("final_rescore")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_png)
    return True


def plot_protocol_runtime(rows: list[dict[str, Any]], out_png: str | Path) -> bool:
    summary = summarize_algorithm_groups(rows)
    if not summary:
        return False
    plt = _pyplot()
    labels = [_group_label(row["engine"], row["confidence_arm"]) for row in summary]
    values = [float(row["avg_runtime_sec"] or 0.0) for row in summary]
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.9), 5))
    bars = ax.bar(labels, values, color=[_engine_color(str(row["engine"])) for row in summary])
    _label_bars(ax, bars, values, "{:.3f}")
    ax.set_title("Tempo medio por algoritmo e braco")
    ax.set_ylabel("runtime medio (s)")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, out_png)
    return True


def plot_protocol_fallback(rows: list[dict[str, Any]], out_png: str | Path) -> bool:
    counts: dict[str, dict[str, int]] = {}
    effects = ["none", "high_uncertainty_flagged", "fallback_requested", "fallback_executed"]
    for row in rows:
        if row.get("status") != "ok":
            continue
        arm = str(row.get("confidence_arm"))
        effect = str(row.get("confidence_policy_effect") or "none")
        if effect not in effects:
            effect = "other"
        counts.setdefault(arm, {}).setdefault(effect, 0)
        counts[arm][effect] += 1
    if not counts:
        return False
    all_effects = [effect for effect in effects if any(counts[arm].get(effect, 0) for arm in counts)]
    if any(counts[arm].get("other", 0) for arm in counts):
        all_effects.append("other")

    plt = _pyplot()
    arms = sorted(counts)
    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = [0] * len(arms)
    colors = {
        "none": "#8a8f98",
        "high_uncertainty_flagged": "#d6a21f",
        "fallback_requested": "#d95f02",
        "fallback_executed": "#b2182b",
        "other": "#4c78a8",
    }
    for effect in all_effects:
        values = [counts[arm].get(effect, 0) for arm in arms]
        ax.bar(arms, values, bottom=bottoms, label=effect, color=colors.get(effect))
        bottoms = [base + value for base, value in zip(bottoms, values)]
    ax.set_title("Decisoes da heuristica por braco")
    ax.set_ylabel("Numero de execucoes")
    ax.set_xlabel("Braco de confianca")
    ax.legend(loc="upper right")
    _save(fig, out_png)
    return True


def plot_protocol_summary_table(summary_rows: list[dict[str, Any]], out_png: str | Path) -> bool:
    if not summary_rows:
        return False
    columns = [
        "engine",
        "arm",
        "n",
        "avg_final",
        "avg_runtime",
        "fb_req",
        "fb_exec",
    ]
    table_rows = []
    for row in summary_rows:
        table_rows.append(
            [
                _engine_label(str(row["engine"])),
                str(row["confidence_arm"]),
                str(row["n_runs"]),
                _fmt(row.get("avg_final_rescore"), 3),
                _fmt(row.get("avg_runtime_sec"), 3),
                _pct(row.get("fallback_recommended_rate")),
                _pct(row.get("fallback_executed_rate")),
            ]
        )
    plt = _pyplot()
    fig_height = max(3.2, 0.42 * len(table_rows) + 1.4)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    ax.set_title("Tabela consolidada do protocolo", pad=18)
    _save(fig, out_png)
    return True


def write_algorithm_visual_summary(
    path: str | Path,
    summary_rows: list[dict[str, Any]],
    winners: dict[str, int],
    figures: list[Path],
) -> None:
    payload = {
        "winner_counts": winners,
        "group_stats": summary_rows,
        "figures": [path.name for path in figures],
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_algorithm_visual_report_md(
    path: str | Path,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    winners: dict[str, int],
    figures: list[Path],
) -> None:
    lines = [
        "# Visualizacao Cientifica do Protocolo",
        "",
        f"- Execucoes validas: `{sum(1 for row in rows if row.get('status') == 'ok')}`",
        f"- Grupos consolidados: `{len(summary_rows)}`",
        "",
        "## Vitorias",
    ]
    for engine, count in sorted(winners.items()):
        lines.append(f"- {_engine_label(engine)}: `{count}`")
    lines.extend(["", "## Figuras", ""])
    for figure in figures:
        rel = f"figures/{figure.name}"
        lines.append(f"![{figure.stem}]({rel})")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_algorithm_dashboard_html(
    path: str | Path,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    winners: dict[str, int],
    figures: list[Path],
) -> None:
    winner_items = "".join(
        f"<li><strong>{html.escape(_engine_label(engine))}</strong>: {int(count)}</li>"
        for engine, count in sorted(winners.items())
    )
    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(_engine_label(str(row['engine'])))}</td>"
        f"<td>{html.escape(str(row['confidence_arm']))}</td>"
        f"<td>{int(row['n_runs'])}</td>"
        f"<td>{html.escape(_fmt(row.get('avg_final_rescore'), 4))}</td>"
        f"<td>{html.escape(_fmt(row.get('avg_runtime_sec'), 4))}</td>"
        f"<td>{html.escape(_pct(row.get('fallback_recommended_rate')))}</td>"
        f"<td>{html.escape(_pct(row.get('fallback_executed_rate')))}</td>"
        "</tr>"
        for row in summary_rows
    )
    figure_cards = "".join(
        "<section>"
        f"<h2>{html.escape(_figure_title(figure.name))}</h2>"
        f"<img src=\"figures/{html.escape(figure.name)}\" alt=\"{html.escape(figure.stem)}\">"
        "</section>"
        for figure in figures
    )
    body = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>DockingPP - Protocolo AG vs PSO vs ABC+AG+VGOS</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; background: #f7f8fa; }}
    header, section {{ background: white; border: 1px solid #d9dee7; border-radius: 8px; padding: 20px; margin-bottom: 18px; }}
    h1 {{ margin-top: 0; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #e3e7ef; border-radius: 6px; background: white; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border-bottom: 1px solid #e3e7ef; padding: 8px; text-align: left; }}
    th {{ background: #eef2f7; }}
    .links a {{ margin-right: 14px; }}
  </style>
</head>
<body>
  <header>
    <h1>DockingPP: visualizacao cientifica do protocolo</h1>
    <p>Comparacao de AG, PSO e ABC+AG+VGOS com bracos de confianca populacional.</p>
    <p><strong>Execucoes validas:</strong> {sum(1 for row in rows if row.get("status") == "ok")}</p>
    <div class="links">
      <a href="run_table.csv">run_table.csv</a>
      <a href="summary_stats.json">summary_stats.json</a>
      <a href="visual_summary.json">visual_summary.json</a>
    </div>
  </header>
  <section>
    <h2>Vitorias por algoritmo</h2>
    <ul>{winner_items}</ul>
  </section>
  <section>
    <h2>Tabela consolidada</h2>
    <table>
      <thead>
        <tr><th>Algoritmo</th><th>Braco</th><th>n</th><th>avg_final_rescore</th><th>avg_runtime_sec</th><th>fallback recomendado</th><th>fallback executado</th></tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </section>
  {figure_cards}
</body>
</html>
"""
    Path(path).write_text(body, encoding="utf-8")


def _append_visual_section_to_report(path: Path, figures: list[Path]) -> None:
    if not path.exists() or not figures:
        return
    current = path.read_text(encoding="utf-8")
    marker = "## Visualizacoes"
    if marker in current:
        current = current.split(marker, 1)[0].rstrip() + "\n\n"
    lines = [marker, ""]
    for figure in figures:
        lines.append(f"![{figure.stem}](figures/{figure.name})")
        lines.append("")
    lines.append("Dashboard navegavel: `dashboard.html`")
    path.write_text(current.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _group_values(rows: list[dict[str, Any]], metric: str) -> list[tuple[str, list[float]]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        value = _float(row.get(metric))
        if value is None:
            continue
        grouped.setdefault((str(row.get("engine")), str(row.get("confidence_arm"))), []).append(value)
    return [(_group_label(engine, arm), values) for (engine, arm), values in sorted(grouped.items()) if values]


def _ordered_engines(values: dict[str, Any]) -> list[str]:
    order = ["ag", "pso", "hybrid"]
    present = [engine for engine in order if engine in values]
    present.extend(sorted(engine for engine in values if engine not in present))
    return present


def _group_label(engine: str, arm: str) -> str:
    return f"{_engine_label(engine)}\n{arm}"


def _engine_label(engine: str) -> str:
    return ENGINE_LABELS.get(str(engine), str(engine))


def _engine_color(engine: str) -> str:
    return {
        "ag": "#4c78a8",
        "pso": "#59a14f",
        "hybrid": "#e15759",
    }.get(engine, "#8a8f98")


def _engine_colors(engines: list[str]) -> list[str]:
    return [_engine_color(engine) for engine in engines]


def _figure_title(filename: str) -> str:
    return {
        "wins_by_engine.png": "Vitorias: AG vs PSO vs ABC+AG+VGOS",
        "final_rescore_boxplot.png": "Distribuicao de final_rescore",
        "runtime_by_engine.png": "Tempo medio por algoritmo",
        "fallback_by_arm.png": "Decisoes de fallback por braco",
        "summary_table.png": "Tabela visual consolidada",
    }.get(filename, filename)


def _label_bars(ax: Any, bars: Any, values: list[float | int], fmt: str) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _mean(values: Any) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _fmt(value: Any, digits: int) -> str:
    number = _float(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def _pct(value: Any) -> str:
    number = _float(value)
    return "NA" if number is None else f"{number * 100.0:.1f}%"


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib nao disponivel para gerar graficos.") from exc
    return plt


def _save(fig: Any, out_png: str | Path) -> None:
    target = Path(out_png)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight", dpi=150)
    _pyplot().close(fig)

"""Módulo de relatórios para análise e visualização (PT-BR)."""

from dockingpp.reporting.loaders import extract_series, find_matching_jsonl, load_any_json, load_jsonl, load_scientific_runs
from dockingpp.reporting.algorithm_protocol import build_algorithm_protocol_visual_report
from dockingpp.reporting.plots import (
    plot_cost_quality,
    plot_filter_distribution,
    plot_omega_reduction,
    plot_pocket_rank_effect,
    plot_score_stability,
)

__all__ = [
    "extract_series",
    "build_algorithm_protocol_visual_report",
    "find_matching_jsonl",
    "load_any_json",
    "load_jsonl",
    "load_scientific_runs",
    "plot_cost_quality",
    "plot_filter_distribution",
    "plot_omega_reduction",
    "plot_pocket_rank_effect",
    "plot_score_stability",
]

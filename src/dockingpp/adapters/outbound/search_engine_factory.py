"""Adaptador que instancia a engine configurada.

A porta ``SearchEngine`` pertence a aplicacao; este modulo conhece as classes
concretas e funciona como parte da composicao externa.
"""

from __future__ import annotations

from typing import Any, Mapping

from dockingpp.application.ports.search_engine import SearchEngine


def resolve_search_engine(cfg: Any) -> tuple[SearchEngine, str]:
    """Resolve a engine configurada, incluindo aliases historicos."""

    engine_key = _resolve_engine_key(cfg)
    if engine_key == "hybrid":
        from dockingpp.search.algoritmos.hibrido import ABCGAVGOSSearch

        engine: SearchEngine = ABCGAVGOSSearch(cfg)
        return engine, type(engine).__name__
    if engine_key == "hybrid_memory":
        from dockingpp.search.algoritmos.hibrido_memoria import ABCGAVGOSMemorySearch

        engine = ABCGAVGOSMemorySearch(cfg)
        return engine, type(engine).__name__
    if engine_key == "ag":
        from dockingpp.search.algoritmos.ag import AGSearch

        engine = AGSearch(cfg)
        return engine, type(engine).__name__
    if engine_key == "pso":
        from dockingpp.search.algoritmos.pso import PSOSearch

        engine = PSOSearch(cfg)
        return engine, type(engine).__name__
    raise ValueError(f"Unsupported search engine: {engine_key}")


def _resolve_engine_key(cfg: Any) -> str:
    raw = _cfg_value(cfg, "search_engine", "hybrid")
    key = str(raw or "hybrid").strip().lower()
    aliases = {
        "hybrid": "hybrid",
        "abcgavgos": "hybrid",
        "abc_ga_vgos": "hybrid",
        "abc-ga-vgos": "hybrid",
        "hibrido": "hybrid",
        "hybrido": "hybrid",
        "hybrid_memory": "hybrid_memory",
        "hybrid-memory": "hybrid_memory",
        "abcgavgos_memory": "hybrid_memory",
        "abc_ga_vgos_memory": "hybrid_memory",
        "abc-ga-vgos-memory": "hybrid_memory",
        "hibrido_memoria": "hybrid_memory",
        "hibrido-com-memoria": "hybrid_memory",
        "hybrido_memoria": "hybrid_memory",
        "ag": "ag",
        "ga": "ag",
        "genetic": "ag",
        "genetic_algorithm": "ag",
        "pso": "pso",
        "particle_swarm": "pso",
    }
    return aliases.get(key, key)


def _cfg_value(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


__all__ = ["resolve_search_engine"]

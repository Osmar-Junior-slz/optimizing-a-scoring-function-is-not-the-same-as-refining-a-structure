"""Fachada legada do contrato e da fabrica de engines.

O contrato pertence a camada de aplicacao e a fabrica concreta pertence aos
adaptadores. Este modulo preserva o caminho publico historico.
"""

from dockingpp.adapters.outbound.search_engine_factory import resolve_search_engine
from dockingpp.application.ports.search_engine import SearchEngine

__all__ = ["SearchEngine", "resolve_search_engine"]

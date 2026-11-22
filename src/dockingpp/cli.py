"""Fachada legada do adaptador CLI."""

from dockingpp.adapters.inbound.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())

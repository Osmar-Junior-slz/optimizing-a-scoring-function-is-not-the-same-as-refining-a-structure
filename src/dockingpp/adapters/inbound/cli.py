"""Adaptador de entrada por linha de comando."""

from __future__ import annotations

import argparse
from typing import Sequence

from dockingpp.api import Config, run_pipeline
from dockingpp.data.io import load_config


def build_parser() -> argparse.ArgumentParser:
    """Monta o contrato externo da CLI."""

    parser = argparse.ArgumentParser(prog="dockingpp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the docking pipeline")
    run_parser.add_argument("--config", required=True, help="Path to YAML config")
    run_parser.add_argument("--receptor", required=True, help="Path to receptor file")
    run_parser.add_argument("--peptide", required=True, help="Path to peptide file")
    run_parser.add_argument("--out", required=True, help="Output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Traduz argumentos da CLI para o caso de uso publico."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        cfg = Config(**load_config(args.config))
        run_pipeline(cfg, args.receptor, args.peptide, args.out)
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]

"""Fachada de compatibilidade do antigo orquestrador de pipeline."""

from __future__ import annotations

from dockingpp.application.config import Config
from dockingpp.domain.models import RunResult


class OrquestradorPipelineDocking:
    """Preserva a API antiga e delega ao caso de uso hexagonal."""

    def executar(
        self,
        cfg: Config,
        receptor_path: str,
        peptide_path: str,
        out_dir: str,
    ) -> RunResult:
        """Executa o caso de uso montado pelo composition root legado."""

        from dockingpp.pipeline.run import _build_run_docking_use_case

        return _build_run_docking_use_case().execute(
            cfg=cfg,
            receptor_path=receptor_path,
            peptide_path=peptide_path,
            out_dir=out_dir,
        )


__all__ = ["OrquestradorPipelineDocking"]

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from docking_confidence.adapters.dockingpp_archive_engine import DockingPPArchiveIntegrityError
from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine


def test_child_imports_historical_or_instrumented_bytes_as_requested(tmp_path):
    historical = DockingPPLocalEngine()
    tracked = DockingPPLocalEngine(seed_lineage=True)
    assert historical.audit_descriptor()["source_manifest_sha256"] == (
        "8db51faec3fe0ac3cfbe7c3bae5fe596cf20196dd1f7483757aa644cc8191e0d"
    )
    for engine, expected in [(historical, "False"), (tracked, "True")]:
        environment = {**os.environ, "PYTHONPATH": str(engine._ensure_snapshot()),
                       "PYTHONDONTWRITEBYTECODE": "1"}
        observed = subprocess.check_output(
            [sys.executable, "-c", "from dockingpp.search.algoritmos import contexto; "
             "print(hasattr(contexto, 'SeedOrigin'))"],
            text=True, cwd=tmp_path, env=environment,
        )
        assert observed.strip() == expected


def test_changed_historical_source_is_rejected_before_execution():
    engine = DockingPPLocalEngine()
    snapshot = engine._ensure_snapshot()
    assert snapshot != Path(__file__).resolve().parents[1] / "src"
    operator = snapshot / "dockingpp/search/algoritmos/base.py"
    assert hashlib.sha256(operator.read_bytes()).hexdigest() == (
        "e6af6bf91162a81d3e506bb59c57bf03bfaf74e374bb3c766733be55ecdf040b"
    )
    operator.write_bytes(operator.read_bytes() + b"\n")
    with pytest.raises(DockingPPArchiveIntegrityError, match="hash failed"):
        engine._ensure_snapshot()
    with pytest.raises(DockingPPArchiveIntegrityError, match="hash failed"):
        engine.audit_descriptor()

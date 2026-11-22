"""Execute the DockingPP source shipped in this same Python distribution."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import yaml

from docking_confidence.adapters.dockingpp_archive_engine import (
    ADAPTER_SCHEMA_VERSION, CLI_MODULE, DockingPPArchiveEngine,
    DockingPPArchiveIntegrityError, _sha256_file,
)

# Exact manifest bytes pinned by the published factorial and the calibrated model.
_HISTORICAL_MANIFEST_SHA256 = "8db51faec3fe0ac3cfbe7c3bae5fe596cf20196dd1f7483757aa644cc8191e0d"


class DockingPPLocalEngine(DockingPPArchiveEngine):
    """Reuse the isolated execution contract with a bundled, verified motor."""

    name = "dockingpp_local"

    def __init__(self, *, python_executable=None, timeout_seconds=24 * 60 * 60, seed_lineage: bool = False):
        if not isinstance(seed_lineage, bool):
            raise TypeError("seed_lineage must be a boolean")
        self.module_root = Path(__file__).resolve().parents[2]
        self.package_root = self.module_root / "dockingpp"
        manifest_path = self.package_root / "_standalone_manifest.json"
        self._manifest_bytes = manifest_path.read_bytes()
        self._manifest = json.loads(self._manifest_bytes)
        if self._manifest.get("schema_version") != "bundled_dockingpp_v1":
            raise DockingPPArchiveIntegrityError("unsupported bundled engine manifest")
        self._validate_local_source()
        # Normal runs and resumptions execute the same bytes as the historical
        # study. Only an explicit replay uses the instrumented package in place.
        self._historical_snapshot = None
        if not seed_lineage:
            self._prepare_historical_snapshot()
        self.archive_sha256 = self._manifest["upstream_archive_sha256"]
        self.engine_patch = deepcopy(self._manifest["correction"])
        self.config_entry = "dockingpp/configs/default.yaml"
        config_path = self.package_root / "configs/default.yaml"
        self.config_sha256 = _sha256_file(config_path)
        self._base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.python_executable = str(python_executable or sys.executable)
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise TypeError("timeout_seconds must be numeric")
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self.scientific = True

    def _prepare_historical_snapshot(self):
        bundle = Path(__file__).with_name("factorial_v5.bundle")
        with ZipFile(bundle) as overrides:
            manifest_bytes = overrides.read("_standalone_manifest.json")
            if hashlib.sha256(manifest_bytes).hexdigest() != _HISTORICAL_MANIFEST_SHA256:
                raise DockingPPArchiveIntegrityError("historical engine manifest hash failed")
            historical = json.loads(manifest_bytes)
            available = set(overrides.namelist())
            expected = {row["path"] for row in historical["files"]} | {"_standalone_manifest.json"}
            if not available <= expected or len(available) != len(overrides.namelist()):
                raise DockingPPArchiveIntegrityError("unexpected historical engine archive member")
            # Only copy the pinned sources/configuration, never bytecode from
            # the installed Python package. Keep the directory for this engine's
            # lifetime so concurrent users cannot replace a snapshot in use.
            temporary = TemporaryDirectory(prefix="dockingpp-v5-")
            try:
                module_root = Path(temporary.name)
                package_root = module_root / "dockingpp"
                for record in historical["files"]:
                    relative = record["path"]
                    destination = package_root / relative
                    data = (overrides.read(relative) if relative in available
                            else (self.package_root / relative).read_bytes())
                    if hashlib.sha256(data).hexdigest() != record["sha256"]:
                        raise DockingPPArchiveIntegrityError(f"historical engine source hash failed: {relative}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                (package_root / "_standalone_manifest.json").write_bytes(manifest_bytes)
                self._historical_snapshot = temporary
                self.module_root, self.package_root = module_root, package_root
                self._manifest, self._manifest_bytes = historical, manifest_bytes
                self._validate_local_source()
            except BaseException:
                temporary.cleanup()
                raise

    def _validate_local_source(self):
        if (self.package_root / "_standalone_manifest.json").read_bytes() != self._manifest_bytes:
            raise DockingPPArchiveIntegrityError("bundled engine manifest changed")
        expected = set()
        for record in self._manifest["files"]:
            path = self.package_root / record["path"]
            if path.is_symlink() or not path.resolve().is_relative_to(self.package_root):
                raise DockingPPArchiveIntegrityError("bundled engine file escaped package")
            if not path.is_file() or _sha256_file(path) != record["sha256"]:
                raise DockingPPArchiveIntegrityError(f"bundled engine source hash failed: {record['path']}")
            if path.suffix == ".py":
                expected.add(path.relative_to(self.package_root).as_posix())
        observed = {path.relative_to(self.package_root).as_posix() for path in self.package_root.rglob("*.py")}
        if expected != observed:
            raise DockingPPArchiveIntegrityError("bundled engine contains unexpected Python source")

    def _ensure_snapshot(self):
        self._validate_local_source()
        return self.module_root

    def audit_descriptor(self):
        self._validate_local_source()
        return {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "engine": self.name, "scientific": self.scientific,
            "source": "bundled_in_this_distribution",
            "source_manifest_sha256": hashlib.sha256(self._manifest_bytes).hexdigest(),
            "source_file_count": len(self._manifest["files"]),
            "upstream_archive_sha256": self.archive_sha256,
            "engine_patch": self.engine_patch,
            "config_sha256": self.config_sha256,
            "cli_module": CLI_MODULE,
            "python_executable": self.python_executable,
            "python_runtime_version": sys.version,
            "supports_artifact_work_dir": True,
            "requires_reference_postprocess": True,
            "mobile_partner_domain": self.mobile_partner_domain.as_dict(),
        }

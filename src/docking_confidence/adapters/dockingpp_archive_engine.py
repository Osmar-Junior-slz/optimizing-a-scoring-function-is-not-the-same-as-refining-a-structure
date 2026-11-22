from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from time import perf_counter
from typing import Any, Mapping
import uuid
import zipfile

import yaml

from docking_confidence.adapters import dockingpp_v5_historical_patch, dockingpp_v5_patch
from docking_confidence.domain.execution import (
    Algorithm,
    DockingRunResult,
    ExecutionJob,
    FIXED_EVALUATION_BUDGET,
    ProtocolMode,
)
from docking_confidence.domain.mobile_partner import (
    DEFAULT_MOBILE_PARTNER_DOMAIN,
)


ADAPTER_SCHEMA_VERSION = "dockingpp_archive_adapter_v3"
SNAPSHOT_SCHEMA_VERSION = "dockingpp_archive_snapshot_v2"
CLI_MODULE = "dockingpp.adapters.inbound.cli"
_SNAPSHOT_MANIFEST = ".dockingpp_snapshot.json"
_HEX_SHA256_LENGTH = 64
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FALLBACK_FLAGS = {
    "fallback_to_full",
    "fallback_executed",
    "full_fallback_executed",
    "fallback_used",
}


class DockingPPArchiveError(RuntimeError):
    """Base failure raised by the archived DockingPP adapter."""


class DockingPPArchiveIntegrityError(DockingPPArchiveError):
    """The ZIP, pinned config, or extracted snapshot failed validation."""


class DockingPPExecutionError(DockingPPArchiveError):
    """The isolated CLI process failed or timed out."""


class DockingPPOutputError(DockingPPArchiveError):
    """The CLI returned an incomplete or inconsistent output bundle."""


class DockingPPReducedError(DockingPPOutputError):
    """An explicit, unsuccessful REDUCED arm, never a FULL result."""

    def __init__(self, status: str, reason: str) -> None:
        self.status = status
        self.reason = reason
        super().__init__(f"{status}: {reason}; RUN_FULL_REQUIRED (separate FULL job)")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_expected_sha256(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a SHA-256 string or None")
    normalized = value.strip().lower()
    if len(normalized) != _HEX_SHA256_LENGTH:
        raise ValueError(f"{field} must contain 64 hexadecimal characters")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be hexadecimal") from exc
    return normalized


def _safe_zip_member(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise DockingPPArchiveIntegrityError(f"unsafe ZIP member name: {name!r}")
    path = PurePosixPath(name.rstrip("/"))
    if not path.parts or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DockingPPArchiveIntegrityError(f"unsafe ZIP member path: {name!r}")
    if ":" in path.parts[0]:
        raise DockingPPArchiveIntegrityError(f"drive-qualified ZIP member: {name!r}")
    return path.as_posix()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DockingPPOutputError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise DockingPPOutputError(f"{label} must be a JSON object: {path}")
    return payload


def _finite_number(value: Any, *, field: str, allow_none: bool = True) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DockingPPOutputError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise DockingPPOutputError(f"{field} must be finite")
    return number


class DockingPPArchiveEngine:
    """Run a hash-pinned DockingPP ZIP through its supported inbound CLI.

    The archive is inspected at construction but extracted only on the first run.
    Extraction is traversal-safe and published atomically under ``snapshot_dir``.
    An unpinned archive remains usable for engineering tests, but is deliberately
    marked ``scientific=False``.
    """

    name = "dockingpp_archive"
    supports_artifact_work_dir = True
    requires_reference_postprocess = True
    mobile_partner_domain = DEFAULT_MOBILE_PARTNER_DOMAIN

    def __init__(
        self,
        archive_path: str | Path,
        snapshot_dir: str | Path,
        *,
        config_entry: str | None = None,
        expected_archive_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        python_executable: str | Path | None = None,
        timeout_seconds: float = 24 * 60 * 60,
        max_members: int = 50_000,
        max_uncompressed_bytes: int = 8 * 1024**3,
        seed_lineage: bool = False,
    ) -> None:
        original_archive = Path(archive_path).expanduser()
        if original_archive.is_symlink():
            raise DockingPPArchiveIntegrityError("archive_path must not be a symbolic link")
        self.archive_path = original_archive.resolve(strict=True)
        if not self.archive_path.is_file():
            raise DockingPPArchiveIntegrityError("archive_path must be a regular file")
        if self.archive_path.suffix.lower() != ".zip":
            raise DockingPPArchiveIntegrityError("archive_path must be a ZIP file")

        self.snapshot_dir = Path(snapshot_dir).expanduser().resolve(strict=False)
        if self.snapshot_dir.exists() and self.snapshot_dir.is_symlink():
            raise DockingPPArchiveIntegrityError("snapshot_dir must not be a symbolic link")
        self.expected_archive_sha256 = _strict_expected_sha256(
            expected_archive_sha256, field="expected_archive_sha256"
        )
        self.expected_config_sha256 = _strict_expected_sha256(
            expected_config_sha256, field="expected_config_sha256"
        )
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise TypeError("timeout_seconds must be numeric")
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        for field, value in (
            ("max_members", max_members),
            ("max_uncompressed_bytes", max_uncompressed_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        self.max_members = max_members
        self.max_uncompressed_bytes = max_uncompressed_bytes
        executable = str(python_executable or sys.executable).strip()
        if not executable:
            raise ValueError("python_executable must not be empty")
        self.python_executable = executable

        self.archive_sha256 = _sha256_file(self.archive_path)
        if not isinstance(seed_lineage, bool):
            raise TypeError("seed_lineage must be a boolean")
        self._archive_patch = dockingpp_v5_patch if seed_lineage else dockingpp_v5_historical_patch
        self.engine_patch = self._archive_patch.descriptor(self.archive_sha256)
        if seed_lineage and self.engine_patch is None:
            raise DockingPPArchiveIntegrityError("seed lineage requires the recognized DockingPP V5 archive")
        if (
            self.expected_archive_sha256 is not None
            and self.archive_sha256 != self.expected_archive_sha256
        ):
            raise DockingPPArchiveIntegrityError("DockingPP ZIP SHA-256 does not match its pin")

        (
            self._members,
            self.config_entry,
            self._base_config_bytes,
            self.config_sha256,
            self._module_root_relative,
            self.dockingpp_version,
            self._archive_uncompressed_bytes,
        ) = self._inspect_archive(config_entry)
        if (
            self.expected_config_sha256 is not None
            and self.config_sha256 != self.expected_config_sha256
        ):
            raise DockingPPArchiveIntegrityError("DockingPP config entry SHA-256 does not match its pin")
        try:
            loaded_config = yaml.safe_load(self._base_config_bytes.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as exc:
            raise DockingPPArchiveIntegrityError("config entry is not valid UTF-8 YAML") from exc
        if loaded_config is None:
            loaded_config = {}
        if not isinstance(loaded_config, dict) or not all(
            isinstance(key, str) for key in loaded_config
        ):
            raise DockingPPArchiveIntegrityError("config entry must contain a string-keyed mapping")
        self._base_config: dict[str, Any] = loaded_config
        self.scientific = bool(
            self.expected_archive_sha256 is not None
            and self.expected_config_sha256 is not None
        )
        self._snapshot_path: Path | None = None
        self._snapshot_lock = threading.Lock()

    def _inspect_archive(
        self, requested_config_entry: str | None
    ) -> tuple[
        tuple[tuple[str, str, int], ...],
        str,
        bytes,
        str,
        str,
        str | None,
        int,
    ]:
        try:
            archive = zipfile.ZipFile(self.archive_path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise DockingPPArchiveIntegrityError("archive_path is not a valid ZIP") from exc
        with archive:
            infos = archive.infolist()
            if not infos or len(infos) > self.max_members:
                raise DockingPPArchiveIntegrityError("ZIP member count is empty or exceeds the limit")
            total_size = sum(info.file_size for info in infos if not info.is_dir())
            if total_size > self.max_uncompressed_bytes:
                raise DockingPPArchiveIntegrityError("ZIP uncompressed size exceeds the limit")

            members: list[tuple[str, str, int]] = []
            by_portable_name: dict[str, str] = {}
            file_names: list[str] = []
            info_by_name: dict[str, zipfile.ZipInfo] = {}
            for info in infos:
                normalized = _safe_zip_member(info.filename)
                portable = normalized.casefold()
                if portable in by_portable_name:
                    raise DockingPPArchiveIntegrityError(
                        f"duplicate/case-colliding ZIP member: {normalized}"
                    )
                by_portable_name[portable] = normalized
                if info.flag_bits & 0x1:
                    raise DockingPPArchiveIntegrityError("encrypted ZIP members are not supported")
                mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if stat.S_ISLNK(mode) or file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise DockingPPArchiveIntegrityError(
                        f"special or symbolic ZIP member is forbidden: {normalized}"
                    )
                if not info.is_dir():
                    members.append((normalized, info.filename, info.file_size))
                    file_names.append(normalized)
                    info_by_name[normalized] = info

            module_candidates = sorted(
                name[: -len("dockingpp/__init__.py")]
                for name in file_names
                if name.endswith("dockingpp/__init__.py")
            )
            if len(set(module_candidates)) != 1:
                raise DockingPPArchiveIntegrityError(
                    "ZIP must contain exactly one importable dockingpp package root"
                )
            module_root = module_candidates[0].rstrip("/")

            if requested_config_entry is None:
                yaml_entries = sorted(
                    name for name in file_names if name.lower().endswith((".yaml", ".yml"))
                )
                if len(yaml_entries) != 1:
                    raise DockingPPArchiveIntegrityError(
                        "config_entry is required when the ZIP has zero or multiple YAML files"
                    )
                selected_config = yaml_entries[0]
            else:
                selected_config = _safe_zip_member(requested_config_entry)
                canonical = by_portable_name.get(selected_config.casefold())
                if canonical is None or canonical not in info_by_name:
                    raise DockingPPArchiveIntegrityError(
                        f"config_entry is absent or not a regular file: {selected_config}"
                    )
                selected_config = canonical
            try:
                config_bytes = archive.read(info_by_name[selected_config])
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise DockingPPArchiveIntegrityError("cannot read the pinned config entry") from exc

            version = self._read_archive_version(archive, file_names, module_root)
            return (
                tuple(sorted(members)),
                selected_config,
                config_bytes,
                _sha256_bytes(config_bytes),
                module_root,
                version,
                total_size,
            )

    @staticmethod
    def _read_archive_version(
        archive: zipfile.ZipFile, file_names: list[str], module_root: str
    ) -> str | None:
        pyproject = f"{module_root + '/' if module_root else ''}pyproject.toml"
        if pyproject in file_names:
            try:
                import tomllib

                payload = tomllib.loads(archive.read(pyproject).decode("utf-8"))
                version = payload.get("project", {}).get("version")
                if isinstance(version, str) and version.strip():
                    return version.strip()
            except (KeyError, OSError, UnicodeDecodeError, ValueError):
                pass
        return None

    def _assert_archive_unchanged(self) -> None:
        if _sha256_file(self.archive_path) != self.archive_sha256:
            raise DockingPPArchiveIntegrityError("DockingPP ZIP changed after adapter construction")

    def audit_descriptor(self) -> dict[str, Any]:
        self._assert_archive_unchanged()
        return {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "engine": self.name,
            "scientific": self.scientific,
            "supports_artifact_work_dir": self.supports_artifact_work_dir,
            "requires_reference_postprocess": self.requires_reference_postprocess,
            "cli_module": CLI_MODULE,
            "archive_path": str(self.archive_path),
            "archive_filename": self.archive_path.name,
            "archive_sha256": self.archive_sha256,
            "engine_patch": self.engine_patch,
            "archive_sha256_pinned": self.expected_archive_sha256 is not None,
            "archive_size_bytes": self.archive_path.stat().st_size,
            "archive_uncompressed_bytes": self._archive_uncompressed_bytes,
            "archive_member_count": len(self._members),
            "snapshot_dir": str(self.snapshot_dir),
            "config_entry": self.config_entry,
            "config_sha256": self.config_sha256,
            "config_sha256_pinned": self.expected_config_sha256 is not None,
            "dockingpp_version": self.dockingpp_version,
            "python_runtime_version": sys.version,
            "python_executable": self.python_executable,
            "mobile_partner_domain": self.mobile_partner_domain.as_dict(),
        }

    def _snapshot_manifest_core(self, files: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "archive_sha256": self.archive_sha256,
            "engine_patch": self.engine_patch,
            "config_entry": self.config_entry,
            "config_sha256": self.config_sha256,
            "module_root": self._module_root_relative,
            "files": files,
        }

    def _validate_snapshot(self, snapshot: Path) -> None:
        manifest_path = snapshot / _SNAPSHOT_MANIFEST
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DockingPPArchiveIntegrityError("snapshot manifest is missing or invalid") from exc
        if not isinstance(manifest, dict):
            raise DockingPPArchiveIntegrityError("snapshot manifest must be an object")
        recorded_manifest_sha = manifest.pop("manifest_sha256", None)
        if recorded_manifest_sha != _sha256_bytes(_canonical_json_bytes(manifest)):
            raise DockingPPArchiveIntegrityError("snapshot manifest integrity hash failed")
        if (
            manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
            or manifest.get("archive_sha256") != self.archive_sha256
            or manifest.get("engine_patch") != self.engine_patch
            or manifest.get("config_entry") != self.config_entry
            or manifest.get("config_sha256") != self.config_sha256
            or manifest.get("module_root") != self._module_root_relative
        ):
            raise DockingPPArchiveIntegrityError("snapshot does not match this adapter")
        records = manifest.get("files")
        if not isinstance(records, list):
            raise DockingPPArchiveIntegrityError("snapshot file inventory is invalid")
        expected_paths: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise DockingPPArchiveIntegrityError("snapshot file record is invalid")
            relative = _safe_zip_member(record.get("path"))
            path = snapshot.joinpath(*PurePosixPath(relative).parts)
            if path.is_symlink() or not path.is_file():
                raise DockingPPArchiveIntegrityError(f"snapshot file is missing: {relative}")
            if path.stat().st_size != record.get("size") or _sha256_file(path) != record.get("sha256"):
                raise DockingPPArchiveIntegrityError(f"snapshot file hash failed: {relative}")
            expected_paths.add(relative)
        actual_paths = {
            path.relative_to(snapshot).as_posix()
            for path in snapshot.rglob("*")
            if path.is_file() and path.name != _SNAPSHOT_MANIFEST
        }
        if actual_paths != expected_paths:
            raise DockingPPArchiveIntegrityError("snapshot contains missing or unexpected files")

    def _remove_private_staging(self, staging: Path) -> None:
        if not staging.exists():
            return
        resolved_parent = staging.resolve(strict=False).parent
        if resolved_parent != self.snapshot_dir.resolve(strict=False):
            raise DockingPPArchiveIntegrityError("refusing to remove staging outside snapshot_dir")
        shutil.rmtree(staging)

    def _extract_snapshot(self, destination: Path) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        staging = self.snapshot_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        staging.mkdir()
        records: list[dict[str, Any]] = []
        try:
            with zipfile.ZipFile(self.archive_path) as archive:
                by_original = {info.filename: info for info in archive.infolist()}
                for normalized, original, expected_size in self._members:
                    output = staging.joinpath(*PurePosixPath(normalized).parts)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    observed_size = 0
                    with archive.open(by_original[original], "r") as source, output.open("xb") as target:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            observed_size += len(chunk)
                            digest.update(chunk)
                            target.write(chunk)
                        target.flush()
                        os.fsync(target.fileno())
                    if observed_size != expected_size:
                        raise DockingPPArchiveIntegrityError(
                            f"ZIP member size changed during extraction: {normalized}"
                        )
                    if self.engine_patch and normalized in self.engine_patch["sources"]:
                        try:
                            patched = self._archive_patch.patch_member(normalized, output.read_bytes())
                        except ValueError as exc:
                            raise DockingPPArchiveIntegrityError(str(exc)) from exc
                        _atomic_write(output, patched)
                        observed_size = len(patched)
                        digest = hashlib.sha256(patched)
                    records.append(
                        {"path": normalized, "size": observed_size, "sha256": digest.hexdigest()}
                    )
            self._assert_archive_unchanged()
            core = self._snapshot_manifest_core(records)
            manifest = {**core, "manifest_sha256": _sha256_bytes(_canonical_json_bytes(core))}
            _atomic_write(staging / _SNAPSHOT_MANIFEST, _canonical_json_bytes(manifest))
            try:
                os.replace(staging, destination)
            except OSError:
                if not destination.is_dir():
                    raise
                self._remove_private_staging(staging)
            self._validate_snapshot(destination)
        except Exception:
            self._remove_private_staging(staging)
            raise

    def _ensure_snapshot(self) -> Path:
        with self._snapshot_lock:
            self._assert_archive_unchanged()
            if self._snapshot_path is not None:
                return self._snapshot_path
            patch_suffix = f"-{self.engine_patch['sha256'][:16]}" if self.engine_patch else ""
            destination = self.snapshot_dir / f"snapshot-{self.archive_sha256}{patch_suffix}"
            if destination.exists():
                if destination.is_symlink() or not destination.is_dir():
                    raise DockingPPArchiveIntegrityError("snapshot destination is not a safe directory")
                self._validate_snapshot(destination)
            else:
                self._extract_snapshot(destination)
            module_root = (
                destination.joinpath(*PurePosixPath(self._module_root_relative).parts)
                if self._module_root_relative
                else destination
            )
            if not (module_root / "dockingpp" / "__init__.py").is_file():
                raise DockingPPArchiveIntegrityError("extracted dockingpp package is missing")
            self._snapshot_path = module_root
            return module_root

    @staticmethod
    def _resolved_job_config(job: ExecutionJob, base: Mapping[str, Any]) -> dict[str, Any]:
        if job.protocol not in (ProtocolMode.FULL, ProtocolMode.REDUCED):
            raise ValueError("ExecutionJob must be a concrete FULL or REDUCED arm")
        if job.profile.mode != job.protocol:
            raise ValueError("ExecutionJob profile mode does not match its protocol")
        if job.algorithm not in (Algorithm.AG, Algorithm.PSO, Algorithm.HYBRID):
            raise ValueError("unsupported search algorithm")
        reduced = job.protocol == ProtocolMode.REDUCED
        config = deepcopy(dict(base))
        docking_task = str(getattr(job.profile.docking_task, "value", job.profile.docking_task))
        config.update(
            {
                "search_engine": job.algorithm.value,
                "docking_task": docking_task,
                "search_space_mode": "reduced" if reduced else "full",
                "full_search": not reduced,
                "seed": job.seed,
                "generations": job.profile.generations,
                "pop_size": job.profile.population,
                "top_pockets": job.profile.top_pockets,
                "max_pockets_used": job.profile.top_pockets,
                # DockingPP retains these legacy key names internally.  Their
                # values come from the one application-wide mobile-partner
                # contract rather than from the archived peptide default.
                "peptide_input_validation_enabled": True,
                "receptor_peptide_pair_validation_enabled": True,
                "peptide_min_residues": (
                    DEFAULT_MOBILE_PARTNER_DOMAIN.minimum_residues
                ),
                "peptide_max_residues": (
                    DEFAULT_MOBILE_PARTNER_DOMAIN.maximum_residues
                ),
                "reduced_geometric_fallback_enabled": False,
                "confidence_calibrator_path": None,
                "confidence_calibrator_required": False,
                "confidence_policy": {
                    "enabled": False,
                    "mode": "disabled",
                    "apply_fallback": False,
                    "search_control": {"enabled": False},
                },
                "docking_confidence_audit": {
                    "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
                    "job_id": job.job_id,
                    "protocol": job.protocol.value,
                    "algorithm": job.algorithm.value,
                    "seed": job.seed,
                    "evaluation_budget": job.profile.evaluation_budget,
                    "docking_task": docking_task,
                    "runtime_basis": "engine_subprocess_wall_seconds",
                    "mobile_partner_domain": (
                        DEFAULT_MOBILE_PARTNER_DOMAIN.as_dict()
                    ),
                },
            }
        )
        if docking_task == "local_refinement":
            config.update(
                search_anchor_mode="input_centroid",
                local_refinement_reduced_scale=0.5,
            )
            if reduced:
                config["objective_evaluation_budget_per_pocket"] = job.profile.evaluation_budget
            config["docking_confidence_audit"].update(
                input_relative_placement_used=True,
                initialization="supplied_pose_plus_uniform_population" if reduced else "original_uniform_population",
                effective_max_trans=float(config.get("max_trans", 5.0)) * (0.5 if reduced else 1.0),
                effective_max_rot_deg=float(config.get("max_rot_deg", 25.0)) * (0.5 if reduced else 1.0),
                reduction_basis="nested_rigid_search_domain_and_objective_budget",
                evaluation_budget_is_hard_cap=bool(reduced),
            )
        initialization = getattr(job.profile, "initialization", None)
        if initialization is not None:
            extent = getattr(job.profile, "search_extent", None)
            policy = getattr(job.profile, "budget_policy", None)
            if docking_task != "local_refinement":
                raise ValueError("factorial controls require local_refinement")
            if initialization not in {"uniform", "seeded"}:
                raise ValueError("initialization must be uniform or seeded")
            if extent != ("reduced" if reduced else "expanded"):
                raise ValueError("factorial search_extent does not match executable protocol")
            if policy not in {"common_ceiling", FIXED_EVALUATION_BUDGET}:
                raise ValueError("unsupported factorial budget_policy")
            fixed_evaluations = policy == FIXED_EVALUATION_BUDGET
            scale = float(job.profile.search_scale)
            translation = float(job.profile.max_translation)
            rotation = float(job.profile.max_rotation_degrees)
            if not all(math.isfinite(value) for value in (scale, translation, rotation)):
                raise ValueError("factorial search bounds must be finite")
            if (reduced and not 0.0 < scale < 1.0) or (not reduced and scale != 1.0):
                raise ValueError("factorial expanded scale must be 1 and reduced scale in (0, 1)")
            if translation <= 0.0 or rotation <= 0.0:
                raise ValueError("factorial search bounds must be positive")
            if job.profile.evaluation_budget < job.profile.population:
                raise ValueError("common evaluation ceiling must cover the initial population")
            pose_source = getattr(job.profile, "pose_source", None)
            if not isinstance(pose_source, str) or not pose_source.strip():
                raise ValueError("factorial experiment requires a documented pose_source")
            config.update(
                initialization=initialization,
                search_extent=extent,
                experiment_budget_policy=policy,
                pose_source=pose_source.strip(),
                max_trans=translation,
                max_rot_deg=rotation,
                local_refinement_reduced_scale=scale,
                objective_evaluation_budget_per_pocket=job.profile.evaluation_budget,
                # Refinement uses one global pocket. Avoid deriving generations
                # from the ceiling through the historical split allocator.
                budget_policy="replicated",
            )
            if fixed_evaluations:
                config["hybrid_plateau_patience"] = 0
            config["docking_confidence_audit"].update(
                experiment_design=getattr(job.profile, "execution_strategy", None) or "factorial_2x2",
                initialization=initialization,
                search_extent=extent,
                search_scale=scale,
                effective_max_trans=translation * scale,
                effective_max_rot_deg=rotation * scale,
                pose_source=pose_source.strip(),
                budget_policy=policy,
                evaluation_budget_is_hard_cap=True,
                evaluation_budget_is_fixed_consumption=fixed_evaluations,
                reduction_basis="independent_rigid_domain_and_initialization_controls",
            )
        return config

    @staticmethod
    def _prepare_work_dir(work_dir: Path | None, job: ExecutionJob) -> tuple[Path, Path]:
        if work_dir is None:
            raise ValueError("DockingPPArchiveEngine requires an explicit work_dir")
        original = Path(work_dir).expanduser()
        if original.exists() and original.is_symlink():
            raise DockingPPArchiveIntegrityError("work_dir must not be a symbolic link")
        root = original.resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        if not _SAFE_JOB_ID.fullmatch(job.job_id):
            raise ValueError("ExecutionJob.job_id is not a safe path component")
        run_dir = root / "dockingpp_runs" / f"{job.job_id}-{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return root, run_dir

    @staticmethod
    def _relative_artifact(path: Path, work_dir: Path) -> str:
        try:
            return path.resolve(strict=True).relative_to(work_dir).as_posix()
        except ValueError as exc:
            raise DockingPPOutputError(f"artifact escaped work_dir: {path}") from exc

    @staticmethod
    def _resolve_output_file(output_dir: Path, value: Any, *, field: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise DockingPPOutputError(f"{field} must be a non-empty path string")
        candidate = Path(value)
        unresolved = candidate if candidate.is_absolute() else output_dir / candidate
        if unresolved.is_symlink():
            raise DockingPPOutputError(f"{field} must not be a symbolic link")
        try:
            path = unresolved.resolve(strict=True)
        except OSError as exc:
            raise DockingPPOutputError(f"{field} is missing or unreadable") from exc
        try:
            path.relative_to(output_dir)
        except ValueError as exc:
            raise DockingPPOutputError(f"{field} escaped the engine output directory") from exc
        if path.is_symlink() or not path.is_file():
            raise DockingPPOutputError(f"{field} is not a regular file")
        return path

    @staticmethod
    def _validate_mode(payload: Mapping[str, Any], expected: str, *, label: str) -> None:
        values = []
        for field in ("search_space_mode", "mode"):
            if field in payload:
                value = payload[field]
                if not isinstance(value, str) or not value.strip():
                    raise DockingPPOutputError(f"{label}.{field} must be a non-empty string")
                values.append(value.strip().lower())
        if not values:
            raise DockingPPOutputError(f"{label} does not declare mode/search_space_mode")
        allowed = (
            {"reduced", "reduced_aggregate"}
            if expected == "reduced"
            else {"full", "full_search", "single"}
        )
        if any(value not in allowed for value in values):
            raise DockingPPOutputError(f"{label} mode does not match the requested {expected} arm")
        if payload.get("executed_mode", expected) not in allowed:
            raise DockingPPOutputError(f"{label}.executed_mode does not match {expected}")

    @staticmethod
    def _validate_algorithm(payload: Mapping[str, Any], job: ExecutionJob, *, label: str) -> None:
        expected_names = {
            "ag": {"ag", "AGSearch"},
            "pso": {"pso", "PSOSearch"},
            "hybrid": {"hybrid", "ABCGAVGOSSearch"},
        }[job.algorithm.value]
        for key in ("engine_name", "search_engine_name", "search_engine", "search_strategy"):
            if key in payload and payload[key] not in expected_names:
                raise DockingPPOutputError(
                    f"{label}.{key}={payload[key]!r} does not match requested algorithm {job.algorithm.value}"
                )
        nested = payload.get("config_resolved_subset")
        if isinstance(nested, Mapping):
            DockingPPArchiveEngine._validate_algorithm(nested, job, label=f"{label}.config_resolved_subset")

    @classmethod
    def _raise_reported_reduced_failure(cls, output_dir: Path, job: ExecutionJob) -> None:
        if job.protocol != ProtocolMode.REDUCED:
            return
        for filename in ("summary.json", "result.json"):
            if not (output_dir / filename).exists():
                continue
            path = cls._resolve_output_file(output_dir, filename, field=filename)
            payload = _read_json_object(path, label=filename)
            status = payload.get("status")
            if status in {"NO_FEASIBLE_POCKET", "REDUCED_FAILED", "RUN_FULL_REQUIRED"}:
                raise DockingPPReducedError(status, str(payload.get("failure_reason", status)))
            # Inspect this before metrics/pose resolution: legacy fallback logs
            # live under fallback_full, and are not REDUCED metrics.
            cls._reject_reduced_fallback(payload, location=filename)
            if payload.get("fallback_reason") == "no_feasible_pocket":
                raise DockingPPReducedError("NO_FEASIBLE_POCKET", "no_feasible_pocket")
        if any(path.name.casefold() == "fallback_full" for path in output_dir.rglob("*")):
            raise DockingPPOutputError("REDUCED output contains a FULL fallback artifact")

    @classmethod
    def _reject_reduced_fallback(cls, payload: Any, *, location: str) -> None:
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                normalized = str(key).strip().lower()
                if normalized in _FALLBACK_FLAGS:
                    if not isinstance(value, bool):
                        raise DockingPPOutputError(f"{location}.{key} must be boolean")
                    if value:
                        raise DockingPPOutputError(
                            f"REDUCED output executed or requested FULL fallback at {location}.{key}"
                        )
                cls._reject_reduced_fallback(value, location=f"{location}.{key}")
        elif isinstance(payload, list):
            for index, value in enumerate(payload):
                cls._reject_reduced_fallback(value, location=f"{location}[{index}]")

    @staticmethod
    def _validate_metrics_jsonl(path: Path) -> None:
        seen = 0
        line_number = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("record is not an object")
                    seen += 1
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise DockingPPOutputError(f"metrics.jsonl is invalid at/near line {line_number}") from exc
        if seen == 0:
            raise DockingPPOutputError("metrics.jsonl contains no JSON records")

    @staticmethod
    def _validate_pose(
        payload: Mapping[str, Any],
        *,
        expected_mode: str,
        expected_peptide_atom_count: int,
    ) -> None:
        if payload.get("schema_version") != "1.0":
            raise DockingPPOutputError("best_pose.json has an unsupported schema_version")
        if payload.get("pose_role") != "best_computed_pose":
            raise DockingPPOutputError("best_pose.json has an invalid pose_role")
        mode = payload.get("search_space_mode")
        if not isinstance(mode, str) or mode.strip().lower() != expected_mode:
            raise DockingPPOutputError("best_pose.json search_space_mode does not match the job")
        if not isinstance(payload.get("engine_name"), str) or not payload["engine_name"].strip():
            raise DockingPPOutputError("best_pose.json engine_name is missing")
        coordinates = payload.get("coords")
        if not isinstance(coordinates, list) or not coordinates:
            raise DockingPPOutputError("best_pose.json coords must be a non-empty list")
        if len(coordinates) != expected_peptide_atom_count:
            raise DockingPPOutputError(
                "best_pose.json coordinate count does not match the peptide input"
            )
        for index, coordinate in enumerate(coordinates):
            if not isinstance(coordinate, list) or len(coordinate) != 3:
                raise DockingPPOutputError(f"best_pose.json coords[{index}] must have three values")
            for axis, value in enumerate(coordinate):
                _finite_number(value, field=f"best_pose.coords[{index}][{axis}]", allow_none=False)
        for field in ("score_cheap", "score_expensive", "final_rescore"):
            _finite_number(payload.get(field), field=f"best_pose.{field}", allow_none=False)
        if payload.get("pocket_id") is None:
            raise DockingPPOutputError("best_pose.json pocket_id is missing")
        generation = payload.get("generation")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise DockingPPOutputError("best_pose.json generation must be a non-negative integer")
        if not isinstance(payload.get("meta"), dict):
            raise DockingPPOutputError("best_pose.json meta must be an object")

    @staticmethod
    def _find_number(payloads: list[Mapping[str, Any]], field: str) -> float | None:
        containers: list[Mapping[str, Any]] = []
        for payload in payloads:
            containers.append(payload)
            for nested_name in ("features", "metrics", "confidence_features", "result"):
                nested = payload.get(nested_name)
                if isinstance(nested, Mapping):
                    containers.append(nested)
        for container in containers:
            if field in container and container[field] is not None:
                return _finite_number(container[field], field=field, allow_none=False)
        return None

    @staticmethod
    def _nested_number(
        payloads: list[Mapping[str, Any]], path: tuple[str, ...]
    ) -> float | None:
        for payload in payloads:
            current: Any = payload
            for component in path:
                if not isinstance(current, Mapping) or component not in current:
                    break
                current = current[component]
            else:
                if current is not None:
                    return _finite_number(
                        current, field=".".join(path), allow_none=False
                    )
        return None

    @staticmethod
    def _factorial_audit_fields(
        job: ExecutionJob, summary: Mapping[str, Any], result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Verify factor attestations and actual consumption before a receipt."""

        if getattr(job.profile, "initialization", None) is None:
            return {}
        records = summary.get("population_confidence_by_pocket")
        if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], Mapping):
            raise DockingPPOutputError("factorial refinement requires one audited pocket")
        pocket = records[0]
        initial = pocket.get("initial_population_audit")
        execution = pocket.get("execution_audit")
        for name, record in (("initial_population_audit", initial), ("execution_audit", execution)):
            if not isinstance(record, Mapping):
                raise DockingPPOutputError(f"factorial {name} is missing")
            try:
                digest = _sha256_bytes(_canonical_json_bytes({key: value for key, value in record.items() if key != "sha256"}))
            except (TypeError, ValueError) as exc:
                raise DockingPPOutputError(f"factorial {name} is not finite JSON") from exc
            if record.get("sha256") != digest:
                raise DockingPPOutputError(f"factorial {name} hash failed")
        other = result.get("population_confidence_by_pocket")
        if not isinstance(other, list) or len(other) != 1 or not isinstance(other[0], Mapping):
            raise DockingPPOutputError("factorial result requires one audited pocket")
        if other[0].get("initial_population_audit") != initial or other[0].get("execution_audit") != execution:
            raise DockingPPOutputError("factorial summary and result audit records differ")
        if initial.get("schema") != "factorial_initial_population_v1" or execution.get("schema") != "factorial_execution_audit_v1":
            raise DockingPPOutputError("factorial audit schema is unsupported")
        for key, expected in (("seed", job.seed), ("initialization", job.profile.initialization),
                              ("search_extent", job.profile.search_extent), ("pose_source", job.profile.pose_source.strip()),
                              ("population_size", job.profile.population), ("recorded_before_objective_evaluations", 0)):
            if initial.get(key) != expected:
                raise DockingPPOutputError(f"factorial initial population {key} does not match plan")
        expected_high = ([float(job.profile.max_translation) * float(job.profile.search_scale)] * 3
                         + [float(job.profile.max_rotation_degrees) * float(job.profile.search_scale)] * 3)
        if initial.get("domain_high") != expected_high or initial.get("domain_low") != [-value for value in expected_high]:
            raise DockingPPOutputError("factorial initial population domain does not match plan")
        vectors = initial.get("solution_vectors")
        if not isinstance(vectors, list) or len(vectors) != job.profile.population:
            raise DockingPPOutputError("factorial initial population has an invalid size")
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != 6:
                raise DockingPPOutputError("factorial initial vectors must have six coordinates")
            for axis, value in enumerate(vector):
                numeric = _finite_number(value, field="initial solution vector", allow_none=False)
                if abs(numeric) > expected_high[axis] + 1e-10:
                    raise DockingPPOutputError("factorial initial vector lies outside requested domain")
        seeded = job.profile.initialization == "seeded"
        if initial.get("seeded_candidate_count") != (1 if seeded else 0) or (seeded and vectors[0] != [0.0] * 6):
            raise DockingPPOutputError("factorial seeded candidate does not match initialization")
        policy = job.profile.budget_policy
        fixed_evaluations = policy == FIXED_EVALUATION_BUDGET
        rules = initial.get("stop_rules")
        if not isinstance(rules, Mapping) or rules.get("generations") != job.profile.generations or rules.get("objective_evaluation_ceiling") != job.profile.evaluation_budget or rules.get("budget_policy") != policy:
            raise DockingPPOutputError("factorial stopping rules do not match plan")
        if fixed_evaluations and (
            rules.get("generation_limit_enabled") is not False
            or rules.get("hybrid_plateau_patience") != 0
            or rules.get("confidence_search_control_enabled") is not False
        ):
            raise DockingPPOutputError("factorial fixed evaluations require disabling early stopping")
        if execution.get("stop_rules") != rules or execution.get("initial_population_sha256") != initial["sha256"]:
            raise DockingPPOutputError("factorial execution does not bind initial state and rules")
        if execution.get("seed") != job.seed or execution.get("algorithm") != job.algorithm.value or execution.get("budget_policy") != policy or execution.get("objective_evaluation_ceiling") != job.profile.evaluation_budget:
            raise DockingPPOutputError("factorial execution factors do not match plan")
        count = execution.get("objective_evaluations_actual")
        series = execution.get("objective_evaluations_by_generation")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= job.profile.evaluation_budget:
            raise DockingPPOutputError("factorial actual objective count violates common ceiling")
        if not isinstance(series, list) or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in series) or sum(series) != count or pocket.get("n_eval_actual") != count:
            raise DockingPPOutputError("factorial objective count and generation records differ")
        completed = execution.get("completed_generations")
        if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0 or (not fixed_evaluations and completed > job.profile.generations) or completed != len(series) - 1 or completed != pocket.get("completed_generations"):
            raise DockingPPOutputError("factorial generation count does not match execution")
        if execution.get("evaluation_count_basis") != "new_cheap_objective_calls_including_internal_operators_excluding_cache_hits_and_final_rescoring":
            raise DockingPPOutputError("factorial objective counter basis is missing or invalid")
        stop_reason = execution.get("stop_reason")
        if stop_reason not in {"generation_limit", "objective_evaluation_ceiling", "hybrid_plateau", "confidence_early_stop"}:
            raise DockingPPOutputError("factorial stop reason is missing or invalid")
        if fixed_evaluations and (count != job.profile.evaluation_budget or stop_reason != "objective_evaluation_ceiling"):
            raise DockingPPOutputError("factorial execution did not complete the fixed evaluation budget")
        return {
            "actual_eval_count": count,
            "stop_reason": stop_reason,
            "initial_population_sha256": initial["sha256"],
            "execution_audit_sha256": execution["sha256"],
        }

    def _validate_outputs(
        self,
        *,
        job: ExecutionJob,
        work_dir: Path,
        run_dir: Path,
        output_dir: Path,
        resolved_config: Path,
        stdout_log: Path,
        stderr_log: Path,
        elapsed_seconds: float,
        peptide_atom_count: int,
    ) -> DockingRunResult:
        self._raise_reported_reduced_failure(output_dir, job)
        summary_path = self._resolve_output_file(
            output_dir, "summary.json", field="summary.json"
        )
        result_path = self._resolve_output_file(output_dir, "result.json", field="result.json")
        metrics_path = self._resolve_output_file(
            output_dir, "metrics.jsonl", field="metrics.jsonl"
        )
        summary = _read_json_object(summary_path, label="summary.json")
        result = _read_json_object(result_path, label="result.json")
        factorial_audit = self._factorial_audit_fields(job, summary, result)
        for label, payload in (("summary", summary), ("result", result)):
            self._validate_algorithm(payload, job, label=label)
            if payload.get("schema_version") != "2.0":
                raise DockingPPOutputError(
                    f"{label}.schema_version must be the DockingPP 2.0 contract"
                )
        expected_mode = job.protocol.value
        self._validate_mode(summary, expected_mode, label="summary")
        self._validate_mode(result, expected_mode, label="result")
        if summary.get("final_pose_available") is not True:
            raise DockingPPOutputError("summary.final_pose_available must be true")

        # FULL writes ``artifacts/best_pose.json`` at the output root, whereas
        # REDUCED writes ``<pocket_id>/artifacts/best_pose.json``.  Trust the
        # declared path only after checking the complete output tree.
        for artifact in output_dir.rglob("*"):
            if artifact.is_symlink():
                raise DockingPPOutputError("symbolic links are forbidden in engine artifacts")
        pose_path = self._resolve_output_file(
            output_dir, summary.get("pose_output_path"), field="summary.pose_output_path"
        )
        if pose_path.name != "best_pose.json" or pose_path.parent.name.casefold() != "artifacts":
            raise DockingPPOutputError(
                "summary.pose_output_path must identify an artifacts/best_pose.json file"
            )
        pose = _read_json_object(pose_path, label="best_pose.json")
        self._validate_algorithm(pose, job, label="best_pose")
        self._validate_pose(
            pose,
            expected_mode=expected_mode,
            expected_peptide_atom_count=peptide_atom_count,
        )
        self._validate_metrics_jsonl(metrics_path)

        expected_task = str(getattr(job.profile.docking_task, "value", job.profile.docking_task))
        if expected_task == "local_refinement":
            meta = pose.get("meta", {})
            if not isinstance(meta, Mapping) or meta.get("docking_task") != expected_task:
                raise DockingPPOutputError("best_pose does not attest requested local_refinement task")
            if meta.get("search_anchor_mode") != "input_centroid" or meta.get("input_relative_placement_used") is not True:
                raise DockingPPOutputError("local_refinement pose does not attest the supplied input frame")

        reduced = job.protocol == ProtocolMode.REDUCED
        if reduced:
            self._reject_reduced_fallback(summary, location="summary")
            self._reject_reduced_fallback(result, location="result")
            self._reject_reduced_fallback(pose, location="best_pose")
            gate = pose.get("meta", {}).get("final_hard_clash_gate", {})
            if isinstance(gate, Mapping) and gate.get("triggered") is True:
                raise DockingPPReducedError("REDUCED_FAILED", "final_pose_hard_clash")
            if any(path.name.casefold() == "fallback_full" for path in output_dir.rglob("*")):
                raise DockingPPOutputError("REDUCED output contains a fallback_full artifact")

        payloads = [result, summary]
        # Summaries historically exclude final rescoring. The application must
        # compare elapsed engine time, including startup, search and rescoring.
        runtime = elapsed_seconds
        if runtime < 0:
            raise DockingPPOutputError("runtime must be non-negative")

        reduced_features: dict[str, float | None] = {
            "scan_gap_norm": None,
            "viable_fraction_mean": None,
            "elite_dispersion": None,
            "uncertainty_count": None,
            "energy_q20_norm": None,
        }
        if reduced:
            feature_version = summary.get("confidence_features_version")
            if feature_version is None:
                feature_version = result.get("confidence_features_version")
            if feature_version != "1.3":
                raise DockingPPOutputError(
                    "REDUCED confidence_features_version must be '1.3'"
                )
            reduced_features = {
                "scan_gap_norm": self._nested_number(
                    payloads,
                    (
                        "confidence_features",
                        "scan_overview",
                        "scan_score_gap_top1_top2_per_peptide_atom",
                    ),
                ),
                "viable_fraction_mean": self._nested_number(
                    payloads,
                    (
                        "confidence_features",
                        "run",
                        "popconf_best_pocket_viable_fraction_mean",
                    ),
                ),
                "elite_dispersion": self._nested_number(
                    payloads,
                    (
                        "confidence_features",
                        "run",
                        "popconf_best_pocket_elite_coords_dispersion_last",
                    ),
                ),
                "uncertainty_count": self._nested_number(
                    payloads,
                    (
                        "confidence_features",
                        "run",
                        "popconf_best_pocket_uncertainty_count",
                    ),
                ),
                "energy_q20_norm": self._nested_number(
                    payloads,
                    (
                        "confidence_features",
                        "best_pocket",
                        "features",
                        "energy_q20_per_peptide_atom",
                    ),
                ),
            }
            # A metric can be mathematically undefined (for example, the
            # top-1/top-2 gap when only one pocket survives).  Preserve that as
            # ``None`` instead of inventing a value.  The later confidence gate
            # must fail closed on missing evidence, while the docking pose and
            # reference metrics remain valid, auditable observations.

        complex_pose_path: Path | None = None
        complex_pose_value = summary.get("complex_pose_pdb_path") or result.get(
            "complex_pose_pdb_path"
        )
        if complex_pose_value is not None:
            complex_pose_path = self._resolve_output_file(
                output_dir, complex_pose_value, field="complex_pose_pdb_path"
            )
        raw_structural_metrics = pose_path.parent / "structural_metrics.json"
        if raw_structural_metrics.exists():
            raw_metrics = _read_json_object(
                self._resolve_output_file(
                    output_dir,
                    raw_structural_metrics.relative_to(output_dir).as_posix(),
                    field="pose structural_metrics.json",
                ),
                label="structural_metrics.json",
            )
            # Native metrics compare to the unbound input, not the experimental
            # reference. The engine explicitly records unavailable metrics when
            # that input has no contacts. Keep the pose for the mandatory external
            # reference evaluator; these optional native metrics never supply DockQ.
            native_metrics_unavailable = (
                raw_metrics.get("status") in ("unavailable", "not_applicable")
                and raw_metrics.get("available") is False
                and "metrics" in raw_metrics and raw_metrics["metrics"] is None
                and isinstance(raw_metrics.get("error"), dict)
                and isinstance(raw_metrics["error"].get("message"), str)
                and bool(raw_metrics["error"]["message"].strip())
            )
            if raw_metrics.get("schema_version") != "1.0" or not (
                isinstance(raw_metrics.get("metrics"), dict) or native_metrics_unavailable
            ):
                raise DockingPPOutputError(
                    "raw structural_metrics.json has an invalid contract"
                )

        return DockingRunResult(
            job_id=job.job_id,
            algorithm=job.algorithm.value,
            protocol=job.protocol.value,
            seed=job.seed,
            generations=job.profile.generations,
            population=job.profile.population,
            top_pockets=job.profile.top_pockets,
            evaluation_budget=job.profile.evaluation_budget,
            runtime_sec=float(runtime),
            **factorial_audit,
            # DockingPP's native structural metrics compare against the unbound
            # --peptide input. They are not valid PPI4DOCK reference metrics and
            # are intentionally left unset for the external reference evaluator.
            dockq=None,
            scan_gap_norm=reduced_features["scan_gap_norm"],
            viable_fraction_mean=reduced_features["viable_fraction_mean"],
            elite_dispersion=reduced_features["elite_dispersion"],
            uncertainty_count=reduced_features["uncertainty_count"],
            energy_q20_norm=reduced_features["energy_q20_norm"],
            fnat=None,
            irmsd=None,
            lrmsd=None,
            pose_artifact_path=self._relative_artifact(pose_path, work_dir),
            complex_pose_pdb_path=(
                self._relative_artifact(complex_pose_path, work_dir)
                if complex_pose_path is not None
                else None
            ),
            reference_metrics_path=None,
            engine_summary_path=self._relative_artifact(summary_path, work_dir),
            engine_result_path=self._relative_artifact(result_path, work_dir),
            resolved_config_path=self._relative_artifact(resolved_config, work_dir),
            stdout_log_path=self._relative_artifact(stdout_log, work_dir),
            stderr_log_path=self._relative_artifact(stderr_log, work_dir),
            exit_code=0,
            engine=self.name,
            scientific=self.scientific,
        )

    @staticmethod
    def _dockingpp_structure_counts(
        path: Path,
        *,
        label: str,
    ) -> tuple[int, int]:
        atom_count = 0
        model_count = 0
        residue_sites: set[tuple[str, str, str]] = set()
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = line[:6].strip().upper()
                    if record == "MODEL":
                        model_count += 1
                        continue
                    if record not in {"ATOM", "HETATM"}:
                        continue
                    try:
                        coordinates = (
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        )
                    except (ValueError, IndexError):
                        continue
                    if all(math.isfinite(value) for value in coordinates):
                        atom_count += 1
                        if record == "ATOM":
                            residue_number = line[22:26].strip()
                            if residue_number:
                                residue_sites.add(
                                    (
                                        line[21:22].strip(),
                                        residue_number,
                                        line[26:27].strip(),
                                    )
                                )
        except (OSError, UnicodeDecodeError) as exc:
            raise DockingPPArchiveIntegrityError(f"cannot read {label} as a PDB") from exc
        if model_count > 1:
            raise DockingPPArchiveIntegrityError(
                f"{label} must be single-model because DockingPP loaders disagree on MODEL handling"
            )
        if atom_count == 0:
            raise DockingPPArchiveIntegrityError(
                f"{label} contains no finite, parseable ATOM/HETATM coordinates"
            )
        if not residue_sites:
            raise DockingPPArchiveIntegrityError(
                f"{label} contains no finite, parseable ATOM residues"
            )
        return atom_count, len(residue_sites)

    def run(
        self,
        *,
        job: ExecutionJob,
        protein_path: Path,
        peptide_path: Path,
        work_dir: Path | None = None,
    ) -> DockingRunResult:
        module_root = self._ensure_snapshot()
        receptor = Path(protein_path).expanduser()
        peptide = Path(peptide_path).expanduser()
        for label, original in (("protein_path", receptor), ("peptide_path", peptide)):
            if original.is_symlink():
                raise DockingPPArchiveIntegrityError(f"{label} must not be a symbolic link")
            resolved = original.resolve(strict=True)
            if not resolved.is_file():
                raise DockingPPArchiveIntegrityError(f"{label} must be a regular file")
            if label == "protein_path":
                receptor = resolved
            else:
                peptide = resolved
        receptor_atom_count, receptor_residue_count = (
            self._dockingpp_structure_counts(receptor, label="protein_path")
        )
        peptide_atom_count, mobile_partner_residue_count = (
            self._dockingpp_structure_counts(
                peptide, label="peptide_path"
            )
        )
        if not self.mobile_partner_domain.contains(mobile_partner_residue_count):
            raise DockingPPArchiveIntegrityError(
                "mobile partner residue count is outside the configured "
                f"operational range {self.mobile_partner_domain.label}: "
                f"observed={mobile_partner_residue_count}"
            )

        work_root, run_dir = self._prepare_work_dir(work_dir, job)
        output_dir = run_dir / "output"
        output_dir.mkdir()
        resolved_config = run_dir / "resolved_config.yaml"
        stdout_log = run_dir / "stdout.log"
        stderr_log = run_dir / "stderr.log"
        config = self._resolved_job_config(job, self._base_config)
        config["docking_confidence_audit"].update(
            {
                "archive_sha256": self.archive_sha256,
                "engine_patch": self.engine_patch,
                "config_entry": self.config_entry,
                "config_sha256": self.config_sha256,
                "receptor_sha256": _sha256_file(receptor),
                "peptide_sha256": _sha256_file(peptide),
                "receptor_atom_count": receptor_atom_count,
                "receptor_residue_count": receptor_residue_count,
                "mobile_partner_atom_count": peptide_atom_count,
                "mobile_partner_residue_count": mobile_partner_residue_count,
            }
        )
        config_bytes = yaml.safe_dump(
            config,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=True,
        ).encode("utf-8")
        _atomic_write(resolved_config, config_bytes)

        command = [
            self.python_executable,
            "-m",
            CLI_MODULE,
            "run",
            "--config",
            str(resolved_config),
            "--receptor",
            str(receptor),
            "--peptide",
            str(peptide),
            "--out",
            str(output_dir),
        ]
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(module_root) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        started = perf_counter()
        try:
            # CreateProcess rejects long working directories even with the
            # extended prefix. Use a short, private ephemeral cwd; all retained
            # artifacts still use absolute paths inside the unique run_dir.
            with tempfile.TemporaryDirectory(prefix="dockingpp-") as process_cwd, stdout_log.open("w", encoding="utf-8", newline="") as stdout_handle, stderr_log.open(
                "w", encoding="utf-8", newline=""
            ) as stderr_handle:
                for key in ("TMP", "TEMP", "TMPDIR"):
                    environment[key] = process_cwd
                completed = subprocess.run(
                    command,
                    cwd=process_cwd,
                    env=environment,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    timeout=self.timeout_seconds,
                    check=False,
                    shell=False,
                )
        except subprocess.TimeoutExpired as exc:
            with stderr_log.open("a", encoding="utf-8") as handle:
                handle.write(f"\nDockingPP timeout after {self.timeout_seconds} seconds.\n")
            raise DockingPPExecutionError(
                f"DockingPP exceeded timeout_seconds={self.timeout_seconds}"
            ) from exc
        except OSError as exc:
            raise DockingPPExecutionError("failed to start the DockingPP Python process") from exc
        elapsed = perf_counter() - started
        if isinstance(completed.returncode, bool) or not isinstance(completed.returncode, int):
            raise DockingPPExecutionError("DockingPP subprocess returned an invalid exit code")
        if completed.returncode != 0:
            self._raise_reported_reduced_failure(output_dir, job)
            raise DockingPPExecutionError(
                f"DockingPP failed with exit code {completed.returncode}; see {stderr_log}"
            )
        return self._validate_outputs(
            job=job,
            work_dir=work_root,
            run_dir=run_dir,
            output_dir=output_dir,
            resolved_config=resolved_config,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            elapsed_seconds=elapsed,
            peptide_atom_count=peptide_atom_count,
        )

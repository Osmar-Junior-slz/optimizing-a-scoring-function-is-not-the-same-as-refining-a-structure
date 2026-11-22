from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re


MANIFEST_SCHEMA_VERSION = "2.0"
MANIFEST_SCHEMA = "integrity_manifest_not_authenticity"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_BUNDLE_FILENAMES = {
    "artifact.json",
    "validation_report.json",
    "probability_calibration_predictions.csv",
    "risk_calibration_predictions.csv",
    "locked_test_predictions.csv",
    "policy_candidates.csv",
    "resolved_config.json",
    "calibration_examples.csv",
    "training_assignments.csv",
    "training_cross_validation_predictions.csv",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_bytes_once(path: str | Path) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(read_bytes_once(path))


@dataclass(frozen=True)
class VerifiedBundle:
    """Bytes verified against an integrity-only, unsigned manifest."""

    manifest_path: Path
    manifest_bytes: bytes
    manifest: dict
    files: dict[Path, bytes]
    source_inputs: dict[Path, dict]

    def bytes_for(self, path: str | Path) -> bytes:
        target = Path(path).resolve()
        try:
            return self.files[target]
        except KeyError as exc:
            raise ValueError(f"audit manifest has no entry for {target}") from exc


def _manifest_relative_path(path: Path, bundle_root: Path) -> str:
    return Path(os.path.relpath(path, bundle_root)).as_posix()


def build_manifest(
    paths: list[str | Path],
    *,
    snapshot_bytes: dict[str | Path, bytes] | None = None,
) -> dict:
    if not paths:
        raise ValueError("audit manifest requires at least one file")
    resolved = [Path(path).resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("audit manifest input paths must be unique")
    bundle_root = resolved[0].parent
    supplied_snapshots = {
        Path(path).resolve(): payload
        for path, payload in (snapshot_bytes or {}).items()
    }
    unknown_snapshots = set(supplied_snapshots) - set(resolved)
    if unknown_snapshots:
        raise ValueError("audit manifest snapshots must correspond to manifest paths")
    bundle_files = []
    source_inputs = []
    for path in resolved:
        payload = supplied_snapshots.get(path)
        if payload is None:
            payload = read_bytes_once(path)
        if not isinstance(payload, bytes):
            raise ValueError("audit manifest snapshot payloads must be bytes")
        entry = {
            "path": _manifest_relative_path(path, bundle_root),
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
        if path.parent == bundle_root and path.name in _BUNDLE_FILENAMES:
            bundle_files.append(entry)
        else:
            source_inputs.append(entry)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "schema": MANIFEST_SCHEMA,
        "algorithm": "sha256",
        "bundle_files": bundle_files,
        "source_inputs": source_inputs,
    }


def _parse_manifest(manifest_bytes: bytes) -> dict:
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("audit manifest must be valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("audit manifest must be a JSON object")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("algorithm") != "sha256"
    ):
        raise ValueError("unsupported audit manifest")
    entries = manifest.get("bundle_files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("audit manifest bundle_files must be a non-empty list")
    if not isinstance(manifest.get("source_inputs"), list):
        raise ValueError("audit manifest source_inputs must be a list")
    return manifest


def read_verified_bundle(manifest_path: str | Path) -> VerifiedBundle:
    manifest_file = Path(manifest_path).resolve()
    manifest_bytes = read_bytes_once(manifest_file)
    manifest = _parse_manifest(manifest_bytes)
    files: dict[Path, bytes] = {}
    entry_paths: set[str] = set()
    all_entry_paths: set[str] = set()
    for entry in manifest["bundle_files"]:
        if not isinstance(entry, dict):
            raise ValueError("audit manifest file entries must be JSON objects")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError("audit manifest contains an invalid file path")
        if relative in entry_paths:
            raise ValueError(f"audit manifest contains duplicate path {relative!r}")
        entry_paths.add(relative)
        all_entry_paths.add(relative)
        target = (manifest_file.parent / Path(relative)).resolve()
        if target == manifest_file:
            raise ValueError("audit manifest cannot include itself as a bundle input")
        if target in files:
            raise ValueError("audit manifest paths resolve to the same file")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if type(size) is not int or size < 0:
            raise ValueError(f"audit manifest size is invalid for {relative}")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"audit manifest SHA-256 is invalid for {relative}")
        try:
            payload = read_bytes_once(target)
        except OSError as exc:
            raise ValueError(f"audit manifest bundle input is unavailable: {relative}") from exc
        if len(payload) != size:
            raise ValueError(f"audit manifest size mismatch for {relative}")
        if sha256_bytes(payload) != digest:
            raise ValueError(f"audit manifest SHA-256 mismatch for {relative}")
        files[target] = payload
    source_inputs: dict[Path, dict] = {}
    for entry in manifest["source_inputs"]:
        if not isinstance(entry, dict):
            raise ValueError("audit manifest source entries must be JSON objects")
        relative = entry.get("path")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError("audit manifest contains an invalid source path")
        if relative in all_entry_paths:
            raise ValueError(f"audit manifest contains duplicate path {relative!r}")
        all_entry_paths.add(relative)
        if type(size) is not int or size < 0:
            raise ValueError(f"audit manifest source size is invalid for {relative}")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"audit manifest source SHA-256 is invalid for {relative}")
        target = (manifest_file.parent / Path(relative)).resolve()
        if target == manifest_file or target in files or target in source_inputs:
            raise ValueError("audit manifest source paths are not unique")
        source_inputs[target] = dict(entry)
    return VerifiedBundle(
        manifest_path=manifest_file,
        manifest_bytes=manifest_bytes,
        manifest=manifest,
        files=files,
        source_inputs=source_inputs,
    )


def verify_manifest_entry(path: str | Path, manifest_path: str | Path) -> None:
    read_verified_bundle(manifest_path).bytes_for(path)

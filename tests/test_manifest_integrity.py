import json
from pathlib import Path

import pytest

from docking_confidence.audit.manifest import (
    MANIFEST_SCHEMA,
    build_manifest,
    read_verified_bundle,
)


def test_manifest_declares_integrity_without_authenticity_and_verifies_every_file(
    tmp_path: Path,
):
    run = tmp_path / "run"
    run.mkdir()
    artifact = run / "artifact.json"
    report = run / "validation_report.json"
    dataset = tmp_path / "source.csv"
    artifact.write_bytes(b'{"artifact": true}\n')
    report.write_bytes(b'{"report": true}\n')
    dataset.write_bytes(b"value\n1\n")

    manifest_payload = build_manifest([artifact, report, dataset])
    assert manifest_payload["schema"] == MANIFEST_SCHEMA
    assert [entry["path"] for entry in manifest_payload["bundle_files"]] == [
        "artifact.json",
        "validation_report.json",
    ]
    assert [entry["path"] for entry in manifest_payload["source_inputs"]] == [
        "../source.csv"
    ]
    manifest = run / "audit_manifest.json"
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    bundle = read_verified_bundle(manifest)
    assert bundle.bytes_for(artifact) == artifact.read_bytes()
    assert bundle.bytes_for(report) == report.read_bytes()
    assert dataset.resolve() in bundle.source_inputs

    dataset.unlink()
    read_verified_bundle(manifest)  # Source snapshots are audit records, not live dependencies.

    report.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest.*(size|SHA-256)"):
        read_verified_bundle(manifest)


def test_manifest_rejects_duplicate_resolved_inputs(tmp_path: Path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        build_manifest([artifact, artifact])

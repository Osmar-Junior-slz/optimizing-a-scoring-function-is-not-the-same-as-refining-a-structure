"""Self-contained ZIP copies without mutating frozen model artifacts."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

from docking_confidence.application.generalized_calibrator import load_calibrator
from docking_confidence.audit.manifest import read_bytes_once, read_verified_bundle, sha256_bytes


def export_calibrator_zip(artifact_path):
    """Verify source bytes and relocate manifest paths only inside the ZIP.

    Model bytes stay identical. The exact original manifest is included as a
    verified provenance member; its disk copy and the frozen protocol stay intact.
    """
    artifact_path = Path(artifact_path).resolve(strict=True)
    loaded = load_calibrator(artifact_path)
    bundle = read_verified_bundle(artifact_path.parent / "audit_manifest.json")
    manifest = deepcopy(bundle.manifest)
    members = {name: payload for name, payload in loaded["downloads"].items()
               if name != "audit_manifest.json"}
    for entry in manifest["bundle_files"]:
        name = PurePosixPath(entry["path"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Portable export requires model-relative bundle files")
        members[name.as_posix()] = bundle.bytes_for(artifact_path.parent / Path(entry["path"]))
    relocated = []
    dataset_digest = loaded["artifact"].get("training_dataset_snapshot", {}).get("sha256")
    for index, (source, entry) in enumerate(bundle.source_inputs.items()):
        payload = read_bytes_once(source)
        if len(payload) != entry["size_bytes"] or sha256_bytes(payload) != entry["sha256"]:
            raise ValueError("Training source changed; portable export was refused")
        name = "inputs/dataset.csv" if entry["sha256"] == dataset_digest else f"inputs/source_{index:03d}_{source.name}"
        if name in members:
            raise ValueError("Portable source paths collide")
        members[name] = payload
        relocated.append({**entry, "path": name})
    if not relocated:
        raise ValueError("Portable export requires verified training source bytes")
    manifest["source_inputs"] = relocated
    provenance = "provenance/audit_manifest_original.json"
    members[provenance] = bundle.manifest_bytes
    manifest["bundle_files"].append({"path": provenance, "size_bytes": len(bundle.manifest_bytes),
        "sha256": sha256_bytes(bundle.manifest_bytes)})
    manifest["export_provenance"] = {"artifact_bytes_unchanged": True,
        "original_manifest_sha256": sha256_bytes(bundle.manifest_bytes),
        "original_manifest_archive_path": provenance, "source_paths_relocated_in_export_only": True}
    members["audit_manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(name, payload)
    return buffer.getvalue()

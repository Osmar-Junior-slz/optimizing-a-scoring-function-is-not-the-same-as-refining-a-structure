"""Rebuild the bundled engine from the recorded upstream ZIP and corrections.

Development utility only: normal execution never needs the upstream archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docking_confidence.adapters import dockingpp_v5_patch as patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    assert hashlib.sha256(args.archive.read_bytes()).hexdigest() == patch.ARCHIVE_SHA256
    destination = Path(__file__).resolve().parents[1] / "src/dockingpp"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    patch_info = patch.descriptor(patch.ARCHIVE_SHA256)
    with zipfile.ZipFile(args.archive) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            if name.startswith("dockingpp/dockingpp/"):
                relative = name.removeprefix("dockingpp/dockingpp/")
            elif name == "dockingpp/configs/default.yaml":
                relative = "configs/default.yaml"
            else:
                continue
            assert ".." not in PurePosixPath(relative).parts
            data = archive.read(name)
            if name in patch_info["sources"]:
                data = patch.patch_member(name, data)
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            records.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {
        "schema_version": "bundled_dockingpp_v1",
        "upstream_archive_sha256": patch.ARCHIVE_SHA256,
        "correction": patch_info,
        "files": sorted(records, key=lambda record: record["path"]),
    }
    (destination / "_standalone_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Integrated {len(records)} files into {destination}")


if __name__ == "__main__":
    main()

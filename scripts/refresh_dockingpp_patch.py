"""Rebuild the audited source corrections after an intentional motor edit."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import pprint
import zipfile


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    package = root / "src/dockingpp"
    patch_path = root / "src/docking_confidence/adapters/dockingpp_v5_patch.py"
    namespace = {}
    exec(compile(patch_path.read_bytes(), str(patch_path), "exec"), {"__file__": str(patch_path)}, namespace)
    if sha(args.archive.read_bytes()) != namespace["ARCHIVE_SHA256"]:
        raise ValueError("upstream archive hash mismatch")
    manifest_path = package / "_standalone_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corrections = {}
    sources = {}
    with zipfile.ZipFile(args.archive) as archive:
        for record in manifest["files"]:
            relative = record["path"]
            if not relative.endswith(".py"):
                continue
            member = "dockingpp/dockingpp/" + relative
            original = archive.read(member)
            updated = (package / relative).read_bytes()
            if member == namespace["MEMBER"] or original == updated:
                continue
            compile(updated, relative, "exec")
            before = original.decode("utf-8").splitlines(keepends=True)
            after = updated.decode("utf-8").splitlines(keepends=True)
            spans = [(i, j, "".join(after[k:l])) for tag, i, j, k, l in
                     difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes()
                     if tag != "equal"]
            corrections[member] = {"before_sha256": sha(original), "after_sha256": sha(updated), "spans": spans}
            sources[member] = (original, updated)
    content = patch_path.read_text(encoding="utf-8")
    start = content.index("_SOURCE_CORRECTIONS = ")
    end = content.index("\ndef descriptor(", start)
    content = content[:start] + "_SOURCE_CORRECTIONS = " + pprint.pformat(corrections, width=100, sort_dicts=True) + "\n\n" + content[end:]
    patch_path.write_text(content, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location("rebuilt_patch", patch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for member, (before, after) in sources.items():
        assert module.patch_member(member, before) == after, member
    with zipfile.ZipFile(args.archive) as archive:
        assert module.patch_member(module.MEMBER, archive.read(module.MEMBER)) == (package / module.MEMBER.removeprefix("dockingpp/dockingpp/")).read_bytes()
    manifest["correction"] = module.descriptor(module.ARCHIVE_SHA256)
    for record in manifest["files"]:
        record["sha256"] = sha((package / record["path"]).read_bytes())
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"patched_sources": len(corrections) + 1, "manifest_sha256": sha(manifest_path.read_bytes())}))


if __name__ == "__main__":
    main()

"""Atomic helpers for scientific run artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: str | os.PathLike[str], content: str) -> None:
    """Write text through a sibling temporary file and atomically replace it."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> None:
    """Serialize one JSON object through :func:`atomic_write_text`."""

    content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    atomic_write_text(path, content)


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of an artifact without changing it."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

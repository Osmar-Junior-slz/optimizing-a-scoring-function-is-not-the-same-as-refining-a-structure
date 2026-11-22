"""Isolated official DockQ worker; no project imports or output-file writes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import sys


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _backend():
    import DockQ.DockQ as backend
    from DockQ import constants
    version = importlib.metadata.version("DockQ")
    if not version.startswith("2."):
        raise RuntimeError(f"Official DockQ v2 required; installed version is {version}")
    if constants.FNAT_THRESHOLD != 5.0 or constants.INTERFACE_THRESHOLD != 10.0:
        raise RuntimeError("Official protein-protein constants are not 5/10 angstroms")
    for name in ("load_PDB", "run_on_all_native_interfaces"):
        if not callable(getattr(backend, name, None)):
            raise RuntimeError(f"Official DockQ API is missing {name}")
    provenance = {
        "available": True, "implementation": "DockQ", "implementation_version": version,
        "implementation_sha256": _sha(inspect.getfile(backend)),
        "constants_sha256": _sha(inspect.getfile(constants)),
        "python_executable": sys.executable,
    }
    return backend, provenance


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported DockQ result type {type(value).__name__}")


def run(request):
    backend, provenance = _backend()
    if request.get("preflight"):
        return provenance
    model = Path(request["model_pdb"]).resolve(strict=True)
    native = Path(request["reference_pdb"]).resolve(strict=True)
    if request.get("expected_version") not in (None, provenance["implementation_version"]):
        raise ValueError("Installed DockQ differs from the frozen metric version")
    for path, field in ((model, "model_sha256"), (native, "reference_sha256")):
        if _sha(path) != request[field]:
            raise ValueError(f"Input hash changed before evaluation: {field}")
    chain_map = request["native_to_model_chain_map"]
    model_structure = backend.load_PDB(str(model), chains=list(chain_map.values()), small_molecule=False)
    native_structure = backend.load_PDB(str(native), chains=list(chain_map), small_molecule=False)
    mapping, total = backend.run_on_all_native_interfaces(
        model_structure, native_structure, chain_map=chain_map,
        no_align=False, capri_peptide=False, low_memory=False,
    )
    if len(mapping) != 1:
        raise ValueError("Exactly one non-null protein-protein interface is required")
    interface, result = next(iter(mapping.items()))
    values = {name: float(result[name]) for name in ("DockQ", "fnat", "iRMSD", "LRMSD")}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Official DockQ returned non-finite metrics")
    if not (0 <= values["DockQ"] <= 1 and 0 <= values["fnat"] <= 1):
        raise ValueError("DockQ and fnat must be in [0,1]")
    if values["iRMSD"] < 0 or values["LRMSD"] < 0:
        raise ValueError("RMSDs must be nonnegative")
    for path, field in ((model, "model_sha256"), (native, "reference_sha256")):
        if _sha(path) != request[field]:
            raise ValueError(f"Input hash changed during evaluation: {field}")
    return {**provenance, "interface": interface, "metrics": values,
            "official_interface_result": result, "total_dockq": float(total)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        result = run({"preflight": True} if args.preflight else json.load(sys.stdin))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, default=_json_default))
    except Exception as exc:
        print(json.dumps({"available": False, "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

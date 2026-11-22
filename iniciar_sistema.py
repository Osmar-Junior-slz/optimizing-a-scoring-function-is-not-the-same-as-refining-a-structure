"""Open this checkout with its own virtual environment, even from global Python."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
APP = SOURCE / "docking_confidence" / "ui" / "app.py"


def project_python() -> Path:
    return ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main(arguments: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    python = project_python()
    if not python.is_file():
        print(f"Ambiente do sistema não encontrado: {python}\n"
              "Consulte a instalação da .venv no README.md desta pasta.", file=sys.stderr)
        return 1
    # Isolation ignores PYTHONHOME, user site-packages and an old PYTHONPATH.
    # Waiting for the child keeps the server alive until the user stops it.
    if Path(sys.executable).resolve() != python.resolve() or not sys.flags.isolated:
        try:
            return subprocess.call([str(python), "-I", str(Path(__file__).resolve()), *arguments], cwd=ROOT)
        except KeyboardInterrupt:
            return 130

    sys.path.insert(0, str(SOURCE))
    # Batch workers use `python -m docking_confidence` and inherit this path.
    os.environ["PYTHONPATH"] = str(SOURCE)
    os.environ.pop("PYTHONHOME", None)
    os.chdir(ROOT)
    if arguments[:1] == ["revisao"]:
        actions = {"curvas": "revision_convergence.py", "controle": "revision_random_control.py",
                   "linhagem": "revision_seed_replay.py", "taxa-poses": "revision_random_pose_rate.py"}
        if len(arguments) < 2 or arguments[1] not in actions:
            print("Uso: iniciar_sistema.cmd revisao {curvas|controle|taxa-poses|linhagem} [opções]\n"
                  "Use --help após a etapa para ver as opções.")
            return 0 if arguments[1:] in ([], ["--help"], ["-h"]) else 2
        action, options = arguments[1], arguments[2:]
        if action == "controle" and not any(s == "--output-dir" or s.startswith("--output-dir=") for s in options):
            folder = "controle_piloto" if any(s.split("=")[0] == "--limit-complexes" for s in options) else "controle"
            options = ["--output-dir", str(ROOT / "output/revisao_professor" / folder), *options]
        if action == "linhagem" and not any(s.split("=")[0] == "--output-dir" for s in options):
            if any(s.split("=")[0] in {"--complex", "--method", "--region"} for s in options):
                options = ["--output-dir", str(ROOT / "output/revisao_professor/linhagem_piloto"), *options]
        if action == "taxa-poses" and not any(s.split("=")[0] == "--output-dir" for s in options):
            if any(s.split("=")[0] in {"--draw-limit", "--limit-complexes", "--control-dir"} for s in options):
                options = ["--output-dir", str(ROOT / "output/revisao_professor/taxa_poses_piloto"), *options]
        try:
            return subprocess.call([str(python), "-B", "-X", "utf8", str(ROOT / "scripts" / actions[action]), *options], cwd=ROOT)
        except KeyboardInterrupt:
            return 130

    packages = {}
    for name in ("docking_confidence", "dockingpp"):
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if origin.parent != SOURCE / name:
            print(f"O pacote {name} pertence a outra instalação: {origin}", file=sys.stderr)
            return 1
        packages[name] = str(origin)

    # Initialize Python's process-exit hook in the main thread, before the
    # Streamlit script thread can import scikit-learn/joblib.
    import concurrent.futures.process  # noqa: F401

    if arguments == ["--check"]:
        for name in ("streamlit", "sklearn"):
            module = importlib.import_module(name)
            packages[name] = {"version": module.__version__, "path": module.__file__}
        print(json.dumps({"python": sys.executable, "isolated": bool(sys.flags.isolated),
                          "app": str(APP), "packages": packages}, ensure_ascii=False, indent=2))
        return 0

    from streamlit.web.cli import main as streamlit_main

    sys.argv = ["streamlit", "run", str(APP), *arguments]
    try:
        return streamlit_main() or 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

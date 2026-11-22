"""Startup must use this checkout even with an unrelated editable installation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "iniciar_sistema.py"
APP = ROOT / "src/docking_confidence/ui/app.py"


def _launcher():
    spec = importlib.util.spec_from_file_location("local_ui_launcher", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_reports_current_packages_despite_foreign_pythonpath(tmp_path):
    foreign = tmp_path / "another_project"
    for name in ("docking_confidence", "dockingpp"):
        package = foreign / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("raise AssertionError('wrong project imported')\n")
    env = {**os.environ, "PYTHONPATH": str(foreign)}
    completed = subprocess.run([sys.executable, str(LAUNCHER), "--check"],
                               cwd=tmp_path, env=env, text=True, capture_output=True, timeout=90)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert Path(report["python"]) == _launcher().project_python()
    assert report["isolated"] is True
    assert Path(report["app"]) == APP
    for name in ("docking_confidence", "dockingpp"):
        assert Path(report["packages"][name]).parent == ROOT / "src" / name
    assert Path(report["packages"]["streamlit"]["path"]).is_relative_to(ROOT / ".venv")


def test_launcher_delegates_to_streamlit_cli_without_starting_a_server(tmp_path):
    completed = subprocess.run([sys.executable, str(LAUNCHER), "--help"],
                               cwd=tmp_path, encoding="utf-8", capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert "Usage:" in completed.stdout and "streamlit run" in completed.stdout


def test_launcher_waits_for_isolated_local_interpreter_and_forwards_exit_status(monkeypatch):
    launcher = _launcher()
    calls = []
    monkeypatch.setattr(launcher.sys, "flags", SimpleNamespace(isolated=0))
    monkeypatch.setattr(launcher.subprocess, "call", lambda command, **kwargs: calls.append((command, kwargs)) or 7)
    assert launcher.main(["--server.port", "8502"]) == 7
    command, options = calls[0]
    assert command == [str(launcher.project_python()), "-I", str(LAUNCHER), "--server.port", "8502"]
    assert options == {"cwd": ROOT}


def test_launcher_missing_venv_does_not_fall_back_to_global_python(tmp_path, monkeypatch, capsys):
    launcher = _launcher()
    monkeypatch.setattr(launcher, "project_python", lambda: tmp_path / "missing-python.exe")
    assert launcher.main([]) == 1
    assert "Ambiente do sistema" in capsys.readouterr().err


def test_app_bootstrap_prefers_local_source_before_importing_pages(tmp_path):
    foreign = tmp_path / "another_project"
    package = foreign / "docking_confidence"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise AssertionError('wrong project imported')\n")
    bootstrap = APP.read_text(encoding="utf-8").split("\nimport streamlit as st", 1)[0]
    code = ("import sys; sys.path.insert(0, sys.argv[2]); "
            "namespace={'__file__':sys.argv[1]}; exec(sys.argv[3], namespace); "
            "import docking_confidence; print(docking_confidence.__file__)")
    completed = subprocess.run([sys.executable, "-I", "-c", code, str(APP), str(foreign), bootstrap],
                               cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()).parent == ROOT / "src" / "docking_confidence"


def test_app_bootstrap_rejects_already_loaded_foreign_package(tmp_path, monkeypatch):
    foreign = SimpleNamespace(__file__=str(tmp_path / "old_project" / "__init__.py"))
    monkeypatch.setitem(sys.modules, "docking_confidence", foreign)
    monkeypatch.setattr(sys, "path", list(sys.path))
    bootstrap = APP.read_text(encoding="utf-8").split("\nimport streamlit as st", 1)[0]
    with pytest.raises(RuntimeError, match="iniciar_sistema.cmd"):
        exec(bootstrap, {"__file__": str(APP)})

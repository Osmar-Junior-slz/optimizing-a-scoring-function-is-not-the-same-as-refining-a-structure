from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from docking_confidence.application import article_study as study
from docking_confidence.ui import article_page
from test_article_study import article_model, _input
from test_training_dataset import _FixturePoseEngine


APP = Path(__file__).parents[1] / "src/docking_confidence/ui/app.py"


def _legacy_article_app():
    # Historical study UI stays testable independently of the calibrator entrypoint.
    return AppTest.from_string(
        "from docking_confidence.ui.article_page import render_article_page\nrender_article_page()"
    )


def test_main_app_has_one_article_workflow_and_never_starts_automatically(article_model, tmp_path, monkeypatch):
    source = _input(tmp_path / "inputs")
    output = tmp_path / "article"
    study.prepare_article_study(source, output, pose_source="Synthetic UI verification", artifact_path=article_model, engine=_FixturePoseEngine(set()))
    calls = []
    monkeypatch.setattr(article_page, "configured_calibrator_path", lambda: article_model)
    monkeypatch.setattr(article_page, "launch_article_study", lambda *args, **kwargs: calls.append((args, kwargs)) or SimpleNamespace(pid=123, log_path=output / "worker.log"))
    app = _legacy_article_app()
    app.session_state["ppi4dock_input_dir"] = str(source)
    app.session_state["ppi4dock_output_dir"] = str(output)
    app.session_state["article_pose_source"] = "Synthetic UI verification"
    app.run(timeout=30)
    assert not app.exception
    assert not app.selectbox and not app.radio
    assert len(app.dataframe[0].value) == 6
    assert {int(value) for value in app.dataframe[0].value.Avaliações} == {5700, 7600}
    assert any("18 execuções" in item.value for item in app.markdown)
    assert not calls
    app.button(key="article_start").click().run(timeout=30)
    assert not app.exception and len(calls) == 1
    assert calls[0] == ((output.resolve(),), {"resume": True})
    app.run(timeout=30)
    assert len(calls) == 1


def test_main_app_prepares_but_does_not_launch_on_prepare(article_model, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(article_page, "configured_calibrator_path", lambda: article_model)
    monkeypatch.setattr(article_page, "prepare_article_study", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(article_page, "launch_article_study", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not launch")))
    app = _legacy_article_app()
    app.session_state["ppi4dock_input_dir"] = str(tmp_path / "inputs")
    app.session_state["ppi4dock_output_dir"] = str(tmp_path / "new_output")
    app.session_state["article_pose_source"] = "Synthetic test"
    app.run(timeout=30)
    app.button(key="article_prepare").click().run(timeout=30)
    assert not app.exception and len(calls) == 1


def test_wrong_python_shows_recovery_command_without_starting_study(tmp_path, monkeypatch):
    model = tmp_path / "synthetic-model.json"
    model.write_text('{"decision_capable": false}', encoding="utf-8")
    descriptor = {"python_executable": "project/.venv/python.exe", "python_runtime_version": "3.12 fixture"}
    gate = SimpleNamespace(protocol_issues=["execution_engine_differs_from_training"],
        contract={"engine": {"name": "dockingpp_local", "scientific": True, "descriptor": descriptor}})
    engine = SimpleNamespace(name="dockingpp_local", scientific=True,
        audit_descriptor=lambda: {**descriptor, "python_executable": "global/python.exe"})

    def reject(*args, **kwargs):
        raise ValueError(study._calibrator_incompatibility_message(gate, engine))

    monkeypatch.setattr(article_page, "configured_calibrator_path", lambda: model)
    monkeypatch.setattr(article_page, "prepare_article_study", reject)
    monkeypatch.setattr(article_page, "launch_article_study", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not launch")))
    app = _legacy_article_app()
    app.session_state["ppi4dock_input_dir"] = str(tmp_path / "inputs")
    app.session_state["ppi4dock_output_dir"] = str(tmp_path / "output")
    app.session_state["article_pose_source"] = "Synthetic environment-mismatch fixture"
    app.run(timeout=30)
    app.button(key="article_prepare").click().run(timeout=30)
    assert not app.exception and not app.success
    text = "\n".join(item.value for item in app.error)
    assert "iniciar_sistema.cmd" in text and "Ctrl+C" in text
    assert "global/python.exe" in text and "project/.venv/python.exe" in text
    assert not (tmp_path / "output" / study.PROTOCOL).exists()

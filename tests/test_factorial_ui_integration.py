from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from docking_confidence.io.batch_discovery import discover_ppi4dock
from docking_confidence.ui import batch_page
from test_batch_discovery import _write_target


def test_factorial_prepares_without_running_and_only_explicit_launch_starts_worker(tmp_path, monkeypatch):
    source = tmp_path / "input"
    _write_target(source)
    report = discover_ppi4dock(source)
    calls = []
    monkeypatch.setattr(batch_page, "launch_real_batch", lambda **kwargs: calls.append(kwargs) or SimpleNamespace(pid=123, log_path=tmp_path / "worker.log"))
    app = AppTest.from_string("from docking_confidence.ui.batch_page import render_batch_page\nrender_batch_page()")
    app.session_state["ppi4dock_input_dir"] = str(source)
    app.session_state["ppi4dock_output_dir"] = str(tmp_path / "output")
    app.session_state["ppi4dock_validated_report"] = report
    app.run(timeout=20)
    app.selectbox(key="ppi4dock_real_protocol").set_value("factorial_2x2").run(timeout=20)
    assert not app.exception
    app.text_area(key="factorial_pose_source").input("Synthetic input for UI test; reference not used to create placement")
    next(button for button in app.button if button.label == "Preparar protocolo fatorial").click().run(timeout=20)
    assert not app.exception
    assert calls == []
    protocol_path = tmp_path / "output" / "experimental_protocol.json"
    assert protocol_path.exists()
    assert len(app.dataframe[-1].value) == 12
    assert app.button(key="factorial_launch").label == "Iniciar/retomar as quatro condições"
    app.button(key="factorial_launch").click().run(timeout=20)
    assert not app.exception
    assert len(calls) == 1
    assert calls[0]["protocol"] == "factorial_2x2"
    assert calls[0]["evaluation_budget"] == 7600
    assert calls[0]["top_pockets"] == 1
    assert Path(calls[0]["experimental_protocol"]) == protocol_path
    app.run(timeout=20)
    assert len(calls) == 1


def test_missing_pose_origin_cannot_prepare_or_launch(tmp_path, monkeypatch):
    source = tmp_path / "input"
    _write_target(source)
    report = discover_ppi4dock(source)
    calls = []
    monkeypatch.setattr(batch_page, "launch_real_batch", lambda **kwargs: calls.append(kwargs))
    app = AppTest.from_string("from docking_confidence.ui.batch_page import render_batch_page\nrender_batch_page()")
    app.session_state["ppi4dock_input_dir"] = str(source)
    app.session_state["ppi4dock_output_dir"] = str(tmp_path / "output")
    app.session_state["ppi4dock_validated_report"] = report
    app.run(timeout=20)
    app.selectbox(key="ppi4dock_real_protocol").set_value("factorial_2x2").run(timeout=20)
    next(button for button in app.button if button.label == "Preparar protocolo fatorial").click().run(timeout=20)
    assert not app.exception
    assert any("Origem da pose fornecida" in error.value for error in app.error)
    assert not (tmp_path / "output" / "experimental_protocol.json").exists()
    assert calls == []


def test_saved_protocol_restores_all_settings_and_reports_runtime_conflict(tmp_path, monkeypatch):
    from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
    from docking_confidence.application.experimental_protocol import prepare_experimental_protocol
    from docking_confidence.application.planning import build_execution_plan
    from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, FactorialExperimentConfig, ProtocolMode

    source = tmp_path / "input"
    _write_target(source)
    report = discover_ppi4dock(source)
    jobs = build_execution_plan(
        algorithms=[Algorithm.PSO, Algorithm.AG, Algorithm.HYBRID], protocol_mode=ProtocolMode.FACTORIAL,
        base_config=BaseSearchConfig(population=40, top_pockets=1, experiment=FactorialExperimentConfig(
            reduced_search_scale=.25, max_translation=4., max_rotation_degrees=20.,
            pose_source="Saved provenance for UI recovery", execution_order_seed=1234,
        )), seeds=[4, 5],
    )
    output = tmp_path / "output"
    original_descriptor = DockingPPLocalEngine.audit_descriptor
    with monkeypatch.context() as old_runtime:
        old_runtime.setattr(DockingPPLocalEngine, "audit_descriptor", lambda self: {
            **original_descriptor(self), "python_executable": "old/python.exe", "python_runtime_version": "3.11.9",
        })
        saved = prepare_experimental_protocol(source, output, jobs=jobs, engine=DockingPPLocalEngine(timeout_seconds=120), report=report)
    original = Path(saved["path"]).read_bytes()
    calls = []
    monkeypatch.setattr(batch_page, "launch_real_batch", lambda **kwargs: calls.append(kwargs))
    app = AppTest.from_string("from docking_confidence.ui.batch_page import render_batch_page\nrender_batch_page()")
    app.session_state["ppi4dock_output_dir"] = str(output)
    app.run(timeout=30)
    assert not app.exception
    assert any("Python 3.11.9" in item.value for item in app.warning)
    app.button(key="factorial_load_saved").click().run(timeout=20)
    assert not app.exception
    assert app.text_input(key="ppi4dock_input_dir").value == str(source.resolve())
    assert app.selectbox(key="ppi4dock_real_protocol").value == "factorial_2x2"
    assert app.multiselect(key="ppi4dock_real_algorithms").value == ["pso", "ag", "hybrid"]
    assert app.number_input(key="ppi4dock_real_seed_start").value == 4
    assert app.number_input(key="ppi4dock_real_seed_count").value == 2
    assert app.number_input(key="factorial_population").value == 40
    assert app.number_input(key="factorial_scale").value == .25
    assert app.number_input(key="factorial_translation").value == 4.
    assert app.number_input(key="factorial_rotation").value == 20.
    assert app.number_input(key="factorial_order_seed").value == 1234
    assert app.number_input(key="ppi4dock_real_timeout").value == 120.
    assert app.text_area(key="factorial_pose_source").value == "Saved provenance for UI recovery"
    assert not calls and Path(saved["path"]).read_bytes() == original
    app.session_state["ppi4dock_validated_report"] = report
    app.run(timeout=20)
    next(button for button in app.button if button.label == "Preparar protocolo fatorial").click().run(timeout=20)
    assert not app.exception
    assert any("difere do protocolo já salvo" in item.value for item in app.error)
    frame = next(item.value for item in app.dataframe if "Campo alterado" in item.value.columns)
    assert set(frame["Campo alterado"]) == {"Executável do Python", "Versão do Python"}
    assert not calls and Path(saved["path"]).read_bytes() == original
    # A fresh destination can be prepared with these restored settings in the current runtime.
    fresh_output = tmp_path / "new_output"
    app.text_input(key="ppi4dock_output_dir").input(str(fresh_output)).run(timeout=20)
    next(button for button in app.button if button.label == "Preparar protocolo fatorial").click().run(timeout=20)
    assert not app.exception
    assert not any("Não foi possível preparar" in item.value for item in app.error)
    assert (fresh_output / "experimental_protocol.json").is_file()
    assert app.button(key="factorial_launch")
    assert not calls


def test_invalid_saved_protocol_is_reported_without_crashing_page(tmp_path):
    (tmp_path / "experimental_protocol.json").write_text('{"schema_version": "tampered"}')
    app = AppTest.from_string("from docking_confidence.ui.batch_page import render_batch_page\nrender_batch_page()")
    app.session_state["ppi4dock_output_dir"] = str(tmp_path)
    app.run(timeout=20)
    assert not app.exception
    assert any("protocolo salvo não pôde ser carregado" in item.value for item in app.warning)
    assert not any(button.key == "factorial_load_saved" for button in app.button)

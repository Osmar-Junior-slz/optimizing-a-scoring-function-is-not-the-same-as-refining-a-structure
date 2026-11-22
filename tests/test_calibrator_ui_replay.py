"""Displaying a saved predictor must not imply fitting it on new input fields."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from docking_confidence.application.generalized_calibrator import create_calibrator
from docking_confidence.ui import calibrator_page
from test_generalized_creation import _dataset


APP = Path(__file__).parents[1] / "src/docking_confidence/ui/app.py"


def _registered_source(app):
    return [caption.value for caption in app.caption
            if caption.value.startswith("Origem do ajuste registrada no artefato:")]


def test_loading_existing_model_uses_registered_source_not_current_input_fields(tmp_path, monkeypatch):
    dataset, config, _, _ = _dataset(tmp_path)
    output = tmp_path / "saved_gva"
    result = create_calibrator(dataset, config, output)
    artifact = Path(result["paths"]["artifact.json"])
    original_bytes = artifact.read_bytes()
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Loading saved output must never start training")
    monkeypatch.setattr(calibrator_page, "create_from_source", forbidden)
    app = AppTest.from_file(str(APP))
    app.session_state["gva_source"] = str(tmp_path / "unrelated_input.csv")
    app.session_state["gva_output"] = str(output)
    app.run(timeout=30)
    assert not app.exception and not app.error and not calls
    assert not app.success
    assert any("carregado e verificado" in info.value for info in app.info)
    assert _registered_source(app) == ["Origem do ajuste registrada no artefato: " + str(dataset.resolve())]

    app.text_input(key="gva_source").set_value(str(tmp_path / "another_input.csv")).run(timeout=30)
    assert not app.exception and not app.error and not calls and not app.success
    assert any("carregado e verificado" in info.value for info in app.info)
    assert _registered_source(app) == ["Origem do ajuste registrada no artefato: " + str(dataset.resolve())]
    assert artifact.read_bytes() == original_bytes


def test_created_message_is_only_for_explicit_creation_action(tmp_path, monkeypatch):
    dataset, _, _, config_file = _dataset(tmp_path)
    output = tmp_path / "new_gva"
    original = calibrator_page.create_from_source
    calls = []
    def observed(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(calibrator_page, "create_from_source", observed)
    app = AppTest.from_file(str(APP))
    app.session_state["gva_source"] = str(dataset)
    app.session_state["gva_output"] = str(output)
    app.session_state["gva_config"] = str(config_file)
    app.run(timeout=30)
    assert not app.exception and not calls and not app.success

    app.button(key="gva_create").click().run(timeout=30)
    assert not app.exception and not app.error and len(calls) == 1
    assert any("criado e salvo nesta ação" in success.value for success in app.success)
    assert _registered_source(app) == ["Origem do ajuste registrada no artefato: " + str(dataset.resolve())]

    app.run(timeout=30)
    assert not app.exception and not app.error and len(calls) == 1
    assert not app.success
    assert any("carregado e verificado" in info.value for info in app.info)
    assert _registered_source(app) == ["Origem do ajuste registrada no artefato: " + str(dataset.resolve())]

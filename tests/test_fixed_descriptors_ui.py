"""One-calibrator hybrid UI behavior without fitting models inside interface tests."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from docking_confidence.ui import calibrator_page


APP = Path(__file__).parents[1] / "src/docking_confidence/ui/app.py"
DESCRIPTORS = [
    "viable_fraction_mean", "elite_dispersion", "uncertainty_count",
    "graph_min_cross_distance_angstrom", "graph_interface_edge_count",
    "graph_interface_density", "graph_receptor_contact_fraction",
    "graph_ligand_contact_fraction", "graph_largest_component_fraction",
    "graph_clash_atom_pair_count", "graph_receptor_residue_count",
    "graph_ligand_residue_count", "graph_degree_std", "graph_max_degree",
    "graph_connected_component_count",
]


def _result(output, *, schema=None):
    return {
        "artifact": {
            "schema_version": schema or calibrator_page.SCHEMA,
            "feature_columns": DESCRIPTORS,
            "source_feature_columns": DESCRIPTORS,
            "training_dataset_snapshot": {"path": "registered_training.csv"},
            "partition": {
                "proper_training": {"complex_count": 34, "row_count": 100},
                "calibration": {"complex_count": 15, "row_count": 43},
            },
        },
        "report": {
            "development_validation": {
                "available": True, "group_mae": .123,
                "skill_against_fold_training_mean": -.1,
            },
            "calibration_diagnostics": {
                "negative_row_count": 2, "negative_complex_count": 2,
                "point_can_be_negative": False,
            },
        },
        "downloads": {"artifact.json": b"{}"},
        "paths": {"artifact.json": str(output / "artifact.json")},
    }


def _texts(app):
    return "\n".join(element.value for kind in (app.markdown, app.caption) for element in kind)


def test_create_shows_fifteen_fixed_descriptors_and_reserved_calibration(tmp_path, monkeypatch):
    output = tmp_path / "new_model"
    calls = []

    def create(source, target, **kwargs):
        calls.append((source, target))
        return _result(output)

    monkeypatch.setattr(calibrator_page, "create_from_source", create)
    app = AppTest.from_file(str(APP))
    app.session_state["gva_source"] = "training.csv"
    app.session_state["gva_output"] = str(output)
    app.run(timeout=30)
    assert not app.exception and not calls
    app.button(key="gva_create").click().run(timeout=30)
    assert not app.exception and not app.error
    assert calls == [("training.csv", str(output))]
    text = _texts(app)
    assert "Descritores utilizados: 15" in text
    assert "Preditor híbrido: 50% linear + 50% árvores" in text
    assert "uma única calibração Venn–Abers ao final" in text
    assert all(name in text for name in DESCRIPTORS)
    assert "0.123 de margem DockQ" in text
    assert "A validação separa complexos somente dentro do treino" in text
    assert "a padronização e os dois componentes são ajustados em cada divisão" in text
    assert "dados de calibração ficam reservados" in text
    assert "escolha do descritor" not in text
    assert "2 exemplos de perda, de 2 complexos" in text
    assert any("não contém margens negativas" in warning.value for warning in app.warning)
    assert any("ainda não mostrou ganho" in warning.value for warning in app.warning)
    assert "Não representam porcentagem de confiança" in text
    assert not app.selectbox and not app.multiselect
    assert len(app.get("download_button")) == 1
    assert all(name not in text for name in ["algorithm_ag", "algorithm_pso", "algorithm_hybrid"])


def test_reloading_current_model_keeps_fixed_descriptors_without_refitting(tmp_path, monkeypatch):
    output = tmp_path / "saved_model"
    output.mkdir()
    (output / "artifact.json").write_text("{}", encoding="utf-8")
    loads = []

    def load(path):
        loads.append(path)
        return _result(output)

    def forbidden(*args, **kwargs):
        raise AssertionError("Reading a saved model cannot retrain it")

    monkeypatch.setattr(calibrator_page, "load_calibrator", load)
    monkeypatch.setattr(calibrator_page, "create_from_source", forbidden)
    app = AppTest.from_file(str(APP))
    app.session_state["gva_source"] = "unrelated.csv"
    app.session_state["gva_output"] = str(output)
    app.run(timeout=30)
    assert not app.exception and not app.error and len(loads) == 1
    assert all(name in _texts(app) for name in DESCRIPTORS)
    assert "Preditor híbrido: 50% linear + 50% árvores" in _texts(app)
    assert any("carregado e verificado" in info.value for info in app.info)
    assert not app.success
    app.run(timeout=30)
    assert not app.exception and len(loads) == 1


def test_historical_model_is_identified_without_current_descriptor_claim(tmp_path, monkeypatch):
    output = tmp_path / "historical"
    output.mkdir()
    (output / "artifact.json").write_text("{}", encoding="utf-8")
    model = _result(output, schema="historical_schema")
    model["artifact"]["feature_columns"] = ["old_first", "old_second"]
    monkeypatch.setattr(calibrator_page, "load_calibrator", lambda path: model)
    app = AppTest.from_file(str(APP))
    app.session_state["gva_output"] = str(output)
    app.run(timeout=30)
    assert not app.exception and not app.error
    assert any("Modelo histórico carregado" in info.value for info in app.info)
    assert "Descritores utilizados:" not in _texts(app)
    assert "Preditor híbrido:" not in _texts(app)
    assert "A validação separa complexos somente dentro do treino" not in _texts(app)

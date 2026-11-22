from copy import deepcopy
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from docking_confidence.ui import training_page


APP = Path(__file__).parents[1] / "src/docking_confidence/ui/app.py"


def _app():
    return AppTest.from_string(
        "from docking_confidence.ui.training_page import render_training_page\n"
        "render_training_page()"
    )


def _preparation(tmp_path, valid=True):
    return {
        "valid": valid, "dataset_path": str(tmp_path / "dataset.csv") if valid else None,
        "candidate_dataset_path": str(tmp_path / "candidate.csv"),
        "report_path": str(tmp_path / "dataset_build_report.json"),
        "config_path": "configs/local_refinement_pilot.yaml",
        "docking_task": "local_refinement",
        "feature_schema": "local_refinement_decision_features_v1",
        "feature_columns": ["viable_fraction_mean", "elite_dispersion"],
        "dataset_sha256": "a" * 64,
        "counts": {"complexes_found": 50, "valid_pairs": 135, "reduced_failed": 9},
        "preview": {"row_count": 135, "independent_complex_count": 48,
                    "algorithm_count": 3, "seeds": [1], "preserved_count": 117,
                    "non_preserved_count": 18, "reduced_failed_count": 9},
        "issues": [] if valid else [{"code": "duplicate_fingerprint", "message": "Aliases ambíguos"}],
    }


def test_article_component_keeps_one_workflow_without_retraining_test_inputs(tmp_path, monkeypatch):
    from docking_confidence.ui import article_page
    model = tmp_path / "artifact.json"
    model.write_text('{"decision_capable": false}')
    monkeypatch.setattr(article_page, "configured_calibrator_path", lambda: model)
    def forbidden(*args, **kwargs):
        raise AssertionError("An article render/rerun must not prepare, launch or train")
    monkeypatch.setattr(article_page, "prepare_article_study", forbidden)
    monkeypatch.setattr(article_page, "launch_article_study", forbidden)
    monkeypatch.setattr(training_page, "train_prepared_dataset", forbidden)
    app = AppTest.from_string(
        "from docking_confidence.ui.article_page import render_article_page\n"
        "render_article_page()"
    )
    app.session_state["ppi4dock_output_dir"] = "output_from_docking"
    app.run(timeout=20)
    assert not app.exception
    assert not app.tabs
    assert app.text_input(key="ppi4dock_output_dir").value == "output_from_docking"
    assert not any(button.key == "training_train" for button in app.button)
    assert app.button(key="article_prepare").disabled
    app.run(timeout=20)
    assert not app.exception
    assert app.text_input(key="ppi4dock_output_dir").value == "output_from_docking"


@pytest.mark.parametrize("valid", [True, False])
def test_prepare_and_reruns_never_train_implicitly(tmp_path, monkeypatch, valid):
    build_calls = []
    training_calls = []
    def build(batch_dir, output_dir, *, profile, fingerprint_alias_policy, progress_callback, factorial_initialization):
        assert factorial_initialization == "uniform"
        build_calls.append((batch_dir, Path(output_dir), profile, fingerprint_alias_policy))
        progress_callback(0.3, "Validando pares")
        progress_callback(1.0, "Preparação concluída" if valid else "Análise concluída: dataset bloqueado")
        return _preparation(tmp_path, valid)
    def train(prepared, output, *, progress_callback):
        training_calls.append((prepared, output))
        progress_callback(0.45, "Calibrando Venn–Abers")
        raise ValueError("probability-calibration evidence gate: total=8/10")
    monkeypatch.setattr(training_page, "build_training_dataset", build)
    monkeypatch.setattr(training_page, "train_prepared_dataset", train)
    app = _app()
    app.session_state["training_batch_dir"] = "batch_results"
    app.session_state["training_output_dir"] = str(tmp_path / "output")
    app.run(timeout=20)
    app.button(key="training_prepare").click().run()
    assert not app.exception
    assert len(build_calls) == 1
    assert build_calls[0][1].parent.name == "dataset_preparation"
    assert build_calls[0][2] == "operational"
    assert build_calls[0][3] == "first_complex_id"
    assert app.get("progress")[0].proto.value == 100
    assert "decorridos" in app.get("progress")[0].proto.text
    assert app.button(key="training_train").disabled is not valid
    assert not training_calls
    app.run()
    assert len(build_calls) == 1 and not training_calls
    if valid:
        app.button(key="training_train").click().run()
        assert len(training_calls) == 1
        assert any("total=8/10" in error.value for error in app.error)
        assert app.get("progress")[0].proto.value == 45
        assert app.status[0].label == "Treinamento interrompido; confira o diagnóstico"
        assert app.status[0].state == "error"
        app.run()
        assert len(training_calls) == 1
    else:
        assert any("duplicate_fingerprint" in value.value for value in app.warning)


def test_changed_inputs_invalidate_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(training_page, "build_training_dataset", lambda *a, **k: _preparation(tmp_path))
    app = _app()
    app.session_state["training_batch_dir"] = "batch_results"
    app.session_state["training_output_dir"] = str(tmp_path / "output")
    app.run(timeout=20)
    app.button(key="training_prepare").click().run()
    assert not app.button(key="training_train").disabled
    app.text_input(key="training_batch_dir").set_value("another_batch").run()
    assert app.button(key="training_train").disabled


@pytest.mark.parametrize("study,expected", [("operational_cascade","operational"),("factorial_2x2","default")])
def test_article_training_recipe_is_bound_to_saved_experiment(tmp_path,study,expected):
    from docking_confidence.application.batch import _build_plan,_manifest_for_plan,BatchPlanConflictError
    from docking_confidence.application.planning import build_execution_plan
    from docking_confidence.application.operational import load_operational_config
    from docking_confidence.domain.execution import Algorithm,BaseSearchConfig,FactorialExperimentConfig,ProtocolMode
    from test_batch_runner import _make_report,_RecordingEngine
    import json
    source=tmp_path/"input";report=_make_report(source,count=1)
    base=(load_operational_config() if study=="operational_cascade" else BaseSearchConfig(
        experiment=FactorialExperimentConfig(pose_source="Synthetic placement for protocol test")))
    jobs=build_execution_plan(algorithms=[Algorithm.AG],protocol_mode=ProtocolMode(study),base_config=base,seeds=[1])
    plan,_=_build_plan(report=report,input_root=source,jobs=jobs,engine=_RecordingEngine(),eligible_only=True,skip_invalid=False)
    output=tmp_path/"batch";output.mkdir();manifest=_manifest_for_plan(plan)
    (output/"batch_manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    profile,digest,factorial=training_page._article_dataset_settings(str(output))
    assert profile==expected and digest==manifest["manifest_sha256"]
    assert factorial==(study=="factorial_2x2")
    manifest["plan"]["jobs"][0]["profile"]["evaluation_budget"]+=1
    (output/"batch_manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    with pytest.raises(BatchPlanConflictError):
        training_page._article_dataset_settings(str(output))
    app=_app()
    app.session_state["training_batch_dir"]=str(output)
    app.session_state["training_output_dir"]=str(tmp_path/"training")
    app.run(timeout=30)
    assert not app.exception
    assert app.button(key="training_prepare").disabled
    assert any("integrity" in item.value for item in app.error)


def test_existing_unverified_training_is_not_overwritten(tmp_path, monkeypatch):
    (tmp_path / "artifact.json").write_text("{}")
    called = []
    monkeypatch.setattr(training_page, "train_prepared_dataset", lambda *a: called.append(a))
    app = _app()
    app.session_state["training_output_dir"] = str(tmp_path)
    app.run(timeout=20)
    assert not app.exception
    assert any(x.value == "Treinamento já existe nesta pasta." for x in app.info)
    assert app.button(key="training_train").disabled
    assert any("não pôde ser verificado" in x.value for x in app.error)
    assert not called


def test_article_resolves_aliases_deterministically_and_reports_exclusions(tmp_path, monkeypatch):
    policies = []
    def build(*args, **kwargs):
        policy = kwargs["fingerprint_alias_policy"]
        policies.append(policy)
        prepared = _preparation(tmp_path, valid=policy == "first_complex_id")
        if prepared["valid"]:
            prepared["alias_resolutions"] = [{"selected_complex_id": "1i1r_BC", "excluded_complex_ids": ["1i1r_BD"]}]
        return prepared
    monkeypatch.setattr(training_page, "build_training_dataset", build)
    app = _app()
    app.session_state["training_batch_dir"] = "batch_results"
    app.session_state["training_output_dir"] = str(tmp_path / "output")
    app.run(timeout=20)
    assert not any(item.key == "training_resolve_aliases" for item in app.checkbox)
    app.button(key="training_prepare").click().run()
    assert not app.exception
    assert policies == ["first_complex_id"]
    assert not app.button(key="training_train").disabled
    assert any("primeiro nome em ordem alfabética" in x.value for x in app.warning)
    resolution_table = next(x.value for x in app.table if "Nome mantido" in x.value.columns)
    assert resolution_table.iloc[0].to_dict() == {"Nome mantido": "1i1r_BC", "Nomes excluídos": "1i1r_BD"}
    app.run()
    assert len(policies) == 1  # No implicit rebuilding or training on rerun.


@pytest.mark.parametrize("method,risk_label,risk_value", [
    ("cluster_bootstrap_run_risk", "Erro observado por execução", 0.075),
    ("learn_then_test_bonferroni_binomial_iid", "Erro observado por complexo", 6 / 35),
])
def test_result_ui_distinguishes_training_from_authorization_and_shows_all_gates(
    method, risk_label, risk_value,
):
    artifact = {
        "decision_capable": False,
        "calibration": {"independent_complex_count": 10, "positive_class_complex_count": 8, "negative_class_complex_count": 2},
        "policy": {
            "supported": False, "threshold": None, "reason": "risk gate failed",
            "best_observed_candidate": {
                "threshold": 0.9, "empirical_run_risk": 0.075,
                "empirical_conditional_complex_risk": 6 / 35,
                "accepted_row_count": 80, "false_acceptance_row_count": 6,
                "simultaneous_risk_upper_bound": 0.4,
            },
        },
        "locked_test_audit": {"supported": False, "reason": "no selected threshold"},
        "runtime_gate": {"passed": False, "speedup_full_over_reduced": 0},
        "resolved_config": {"authorization": {"allow_automatic_decision": False}, "policy": {"method": method, "risk_target": 0.35}, "features": {"schema_version": "local_refinement_decision_features_v1"}, "docking_task": "local_refinement"},
        "development_summary": {"locked_test_probability_metrics": {"brier_score": 0.2}},
        "authorization": {"gates": {"profile_allows_automatic_decision": False, "scientific_provenance_passed": False, "risk_calibration_passed": False, "locked_test_passed": False, "runtime_gate_passed": False}},
    }
    result = {"artifact": artifact, "paths": {}, "downloads": {name: b'{}' for name in ["artifact.json", "validation_report.json", "audit_manifest.json"]}}
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from docking_confidence.ui.training_page import _render_training_result\n"
        "_render_training_result(st.session_state['result'])"
    )
    app.session_state["result"] = result
    app.run(timeout=20)
    assert not app.exception
    text = "\n".join(x.value for x in app.markdown)
    for gate in artifact["authorization"]["gates"]:
        assert f"{gate}: NÃO" in text
    assert any("modelo foi treinado" in item.value for item in app.warning)
    table = app.table[0].value
    assert table.loc[table["Indicador"] == "Modelo treinado", "Resultado"].iloc[0] == "SIM"
    assert table.loc[table["Indicador"] == "Autorização automática / decision_capable", "Resultado"].iloc[0] == "NÃO"
    risk_table = next(item.value for item in app.table if risk_label in item.value["Indicador"].values)
    assert float(risk_table.loc[risk_table["Indicador"] == risk_label, "Resultado"].iloc[0]) == pytest.approx(risk_value)
    if method == "cluster_bootstrap_run_risk":
        assert "Limite superior aproximado" in risk_table["Indicador"].values
        assert any("intervalo é aproximado" in item.value for item in app.caption)
        assert any("não constitui uma nova confirmação independente" in item.value for item in app.caption)
    assert len(app.get("download_button")) == 3

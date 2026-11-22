"""UI transitions preserve the validation/test boundary and honest status."""
from pathlib import Path
import hashlib
import json

from streamlit.testing.v1 import AppTest

from docking_confidence.ui import decision_page


def _app():
    return AppTest.from_string(
        "from docking_confidence.ui.decision_page import render_decision_page\n"
        "render_decision_page()"
    )


def _text(app):
    return "\n".join(element.value for kind in (app.markdown, app.caption) for element in kind)


def _report(*, capable=False, status="failed", baseline_equal=True):
    metrics = {"planned_runs":153,"valid_runs":142,"accepted_runs":142,"accepted_complexes":50,
               "preservation":130/142,"adverse_accepted_runs":12,"coverage":142/153,
               "failed_or_missing_runs":11,"runtime_complete":False}
    return {"status":status,"decision_capable":capable,"selected":dict(metrics),"baseline":dict(metrics),
            "difference":{"baseline_equal":baseline_equal},
            "criteria":{"preservation_met":True,"coverage_met":True,
                        "minimum_complexes_met":True,"selective_rule":False,
                        "empirical_baseline_improvement":False}}


def test_initial_page_never_prepares_selects_or_evaluates_automatically(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Workflow actions require a form submission")
    monkeypatch.setattr(decision_page,"prepare_decision_protocol",forbidden)
    monkeypatch.setattr(decision_page,"select_decision_policy",forbidden)
    monkeypatch.setattr(decision_page,"evaluate_frozen_policy",forbidden)
    app=_app().run(timeout=30)
    assert not app.exception and not app.success
    assert "Teste final pendente" in _text(app)
    assert "não é uma porcentagem de confiança" in _text(app)
    assert app.number_input(key="decision_min_preservation").value==90.0
    assert not app.checkbox(key="decision_prospective").value
    assert len(app.button)==3
    app.run(timeout=30)
    assert not app.exception and not app.success


def test_preparation_uses_only_declared_group_split_and_advances_to_validation(tmp_path,monkeypatch):
    split_path=tmp_path/"split.json"
    groups={"schema_version":"decision_group_split_v1","validation_groups":["a"*64],
            "test_groups":["b"*64],"prior_exposure_groups":["c"*64],
            "algorithms":["ag","pso"],"seeds":[1,2]}
    split_path.write_text(json.dumps(groups),encoding="utf-8")
    output=tmp_path/"protocol"
    calls=[]
    def prepare(*args,**kwargs):
        calls.append((args,kwargs))
        return {"protocol":{"analysis_role":"prospective"},"protocol_path":str(output/"protocol.json")}
    monkeypatch.setattr(decision_page,"prepare_decision_protocol",prepare)
    app=_app()
    app.session_state["decision_artifact"]="model/artifact.json"
    app.session_state["decision_config"]="model_source.yaml"
    app.session_state["decision_split"]=str(split_path)
    app.session_state["decision_prepare_output"]=str(output)
    app.session_state["decision_prospective"]=True
    app.session_state["decision_min_coverage"]=50.0
    app.session_state["decision_min_complexes"]=8
    app.run(timeout=30)
    assert not app.exception and not calls
    app.button(key="decision_prepare").click().run(timeout=30)
    assert not app.exception and not app.error and len(calls)==1
    args,kwargs=calls[0]
    assert args==("model/artifact.json",["a"*64],["b"*64],str(output))
    assert kwargs["config_path"]=="model_source.yaml"
    assert kwargs["minimum_preservation"]==.90 and kwargs["minimum_coverage"]==.50
    assert kwargs["minimum_accepted_complexes"]==8
    assert kwargs["analysis_role"]=="prospective" and kwargs["prior_exposure_groups"]==["c"*64]
    assert kwargs["algorithms"]==("ag","pso") and kwargs["seeds"]==(1,2)
    assert app.text_input(key="decision_protocol_path").value==str(output/"protocol.json")
    assert any("teste final ainda estão pendentes" in x.value for x in app.success)
    assert "Teste final pendente" in _text(app)
    app.run(timeout=30)
    assert not app.exception and len(calls)==1


def test_invalid_group_plan_does_not_create_or_advance_protocol(tmp_path,monkeypatch):
    path=tmp_path/"labels.json"
    path.write_text('{"rows":[{"label":1}]}',encoding="utf-8")
    calls=[]
    monkeypatch.setattr(decision_page,"prepare_decision_protocol",lambda *a,**k:calls.append(a))
    app=_app()
    for key,value in {"decision_artifact":"model.json","decision_config":"source.yaml",
                      "decision_split":str(path),"decision_prepare_output":str(tmp_path/"out")}.items():
        app.session_state[key]=value
    app.run(timeout=30).button(key="decision_prepare").click().run(timeout=30)
    assert not app.exception and not calls and not app.success
    assert any("decision_group_split_v1" in x.value for x in app.error)
    assert app.text_input(key="decision_protocol_path").value==""


def test_nonselective_rule_is_not_presented_as_ready_for_use(tmp_path,monkeypatch):
    calls=[]
    result={"policy":{"status":"no_useful_rule","validation_report":_report()},
            "policy_path":str(tmp_path/"policy.json"),"report":_report()}
    def select(*args,**kwargs):
        calls.append((args,kwargs));return result
    monkeypatch.setattr(decision_page,"select_decision_policy",select)
    app=_app()
    for key,value in {"decision_protocol_path":"protocol.json","decision_validation_dataset":"validation.csv",
                      "decision_select_output":str(tmp_path/"selection")}.items():app.session_state[key]=value
    app.run(timeout=30).button(key="decision_select").click().run(timeout=30)
    assert not app.exception and not app.success and len(calls)==1
    assert any("Nenhuma regra útil" in x.value for x in app.warning)
    assert any("mesmas aceitações" in x.value for x in app.warning)
    table=app.dataframe[0].value
    assert table["Execuções planejadas"].tolist()==["153","153"]
    assert table["Preservação entre aceitos"].tolist()==["91,55%","91,55%"]
    assert table["Falhas ou ausências"].tolist()==["11","11"]
    assert "Teste final pendente" in _text(app)


def test_exploratory_final_report_never_enables_automatic_decisions(tmp_path,monkeypatch):
    calls=[]
    result={"report":_report(status="exploratory"),"report_path":str(tmp_path/"final.json")}
    def evaluate(*args,**kwargs):
        calls.append((args,kwargs));return result
    monkeypatch.setattr(decision_page,"evaluate_frozen_policy",evaluate)
    app=_app()
    for key,value in {"decision_policy_path":"policy.json","decision_test_dataset":"test.csv",
                      "decision_evaluate_output":str(tmp_path/"final")}.items():app.session_state[key]=value
    app.run(timeout=30).button(key="decision_evaluate").click().run(timeout=30)
    assert not app.exception and not app.success and len(calls)==1
    assert any("não habilita decisões automáticas" in x.value for x in app.warning)
    app.run(timeout=30)
    assert not app.exception and len(calls)==1
    app.text_input(key="decision_test_dataset").set_value("another_test.csv").run(timeout=30)
    assert "Teste final pendente" in _text(app)
    assert not app.get("download_button")


def test_unavailable_preservation_is_not_displayed_as_perfect_or_zero_risk(tmp_path,monkeypatch):
    report=_report()
    report["selected"].update(accepted_runs=0,preservation=None,risk=None,adverse_accepted_runs=0,coverage=0)
    monkeypatch.setattr(decision_page,"evaluate_frozen_policy",lambda *a,**k:{"report":report,"report_path":str(tmp_path/"final.json")})
    app=_app()
    for key,value in {"decision_policy_path":"policy.json","decision_test_dataset":"test.csv",
                      "decision_evaluate_output":str(tmp_path/"final")}.items():app.session_state[key]=value
    app.run(timeout=30).button(key="decision_evaluate").click().run(timeout=30)
    assert not app.exception and not app.success
    table=app.dataframe[0].value
    assert table.iloc[0]["Preservação entre aceitos"]=="Indisponível"
    assert table.iloc[0]["Cobertura das execuções"]=="0,00%"


def test_failed_new_selection_clears_previous_final_authorization(tmp_path,monkeypatch):
    report=_report(capable=True,status="passed",baseline_equal=False)
    monkeypatch.setattr(decision_page,"evaluate_frozen_policy",lambda *a,**k:{
        "report":report,"report_path":str(tmp_path/"old_final.json")})
    def reject(*args,**kwargs):
        raise ValueError("configuration mismatch")
    monkeypatch.setattr(decision_page,"select_decision_policy",reject)
    app=_app()
    for key,value in {"decision_protocol_path":"new_protocol.json",
                      "decision_validation_dataset":"new_validation.csv",
                      "decision_select_output":str(tmp_path/"selection"),
                      "decision_policy_path":"old_policy.json","decision_test_dataset":"old_test.csv",
                      "decision_evaluate_output":str(tmp_path/"old_final")}.items():app.session_state[key]=value
    app.run(timeout=30).button(key="decision_evaluate").click().run(timeout=30)
    assert not app.exception
    assert any("atendeu aos critérios" in x.value for x in app.success)
    app.button(key="decision_select").click().run(timeout=30)
    assert not app.exception and not app.success
    assert any("configuration mismatch" in x.value for x in app.error)
    assert app.text_input(key="decision_policy_path").value==""
    assert "Teste final pendente" in _text(app)


def test_real_service_prepares_diagnostic_but_refuses_to_promote_legacy_model(tmp_path):
    from docking_confidence.application.generalized_calibrator import create_calibrator
    from test_generalized_creation import _dataset

    dataset,config,_,config_path=_dataset(tmp_path)
    model=create_calibrator(dataset,config,tmp_path/"model")
    artifact_path=Path(model["paths"]["artifact.json"])
    before=artifact_path.read_bytes()
    split=tmp_path/"split.json"
    split.write_text(json.dumps({"schema_version":"decision_group_split_v1",
        "validation_groups":[hashlib.sha256(b'ui validation group').hexdigest()],
        "test_groups":[hashlib.sha256(b'ui final group').hexdigest()],
        "prior_exposure_groups":[]}),encoding="utf-8")
    app=_app()
    for key,value in {"decision_artifact":str(artifact_path),"decision_config":str(config_path),
                      "decision_split":str(split),"decision_prepare_output":str(tmp_path/"diagnostic")}.items():
        app.session_state[key]=value
    app.run(timeout=30).button(key="decision_prepare").click().run(timeout=30)
    assert not app.exception and not app.error
    path=Path(app.text_input(key="decision_protocol_path").value)
    saved=json.loads(path.read_bytes())
    assert saved["analysis_role"]=="exploratory" and saved["decision_capable"] is False
    assert saved["prospective_context_issues"]
    assert len(saved["planned_runs"]["validation"])==3 and len(saved["planned_runs"]["test"])==3
    assert any("exploratório" in x.value for x in app.warning)
    assert artifact_path.read_bytes()==before

    app.checkbox(key="decision_prospective").check()
    app.text_input(key="decision_prepare_output").set_value(str(tmp_path/"prospective"))
    app.button(key="decision_prepare").click().run(timeout=30)
    assert not app.exception and not app.success
    assert any("medida estrutural padronizada" in x.value for x in app.error)
    assert not (tmp_path/"prospective").exists()
    assert app.text_input(key="decision_protocol_path").value==""
    assert artifact_path.read_bytes()==before

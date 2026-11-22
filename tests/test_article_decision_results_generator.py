"""Synthetic end-to-end transcription checks; no biological evidence is read."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest
import pandas as pd

from docking_confidence.application import decision_validation as service
from test_decision_validation import frozen_model, _prepare
from test_decision_validation import QUALITY
from standard_quality_fixtures import bind_standard_quality_records


@pytest.fixture
def generator(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    candidates = [project / "output/artigo_revisado_decisao_20261119/gerar_resultados_decisao.py",
                  project.parent / "artigo/gerar_resultados_decisao.py"]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        pytest.skip("Article generator is not present in this source-only checkout")
    spec = spec_from_file_location("article_decision_generator_smoke", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    article = tmp_path / "article"
    (article / "dados").mkdir(parents=True)
    (article / "figuras").mkdir()
    (article / "dados/resultados_nova_avaliacao.tex").write_text("UNCHANGED FIT SECTION", encoding="utf-8")
    monkeypatch.setattr(module, "ARTICLE", article)
    monkeypatch.setattr(module, "REPO", tmp_path / "isolated_project")
    monkeypatch.setattr(service, "_EXPOSURE_ROOT", tmp_path / "isolated_ledger")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib_cache"))
    return module


def completed_synthetic_stage(model, root, monkeypatch, **criteria):
    prepared = _prepare(model, root, monkeypatch, **criteria)
    stage = root / "stage"
    result = service.select_decision_policy(prepared["protocol_path"], root / "validation.csv", stage / "decision")
    report = result["report"]
    summary = {key: report[key] for key in ("status", "selected", "baseline", "difference", "criteria", "decision_capable")}
    summary.update(role="validation", report_path=result["report_path"])
    (stage / "stage_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return prepared, stage, result


def test_generator_selected_rule_keeps_test_pending_without_reading_test_outcomes(frozen_model, tmp_path, monkeypatch, generator):
    prepared, stage, result = completed_synthetic_stage(frozen_model, tmp_path, monkeypatch)
    assert result["policy"]["status"] == "selected"
    # An absent final dataset is compatible with the validation-only update.
    (tmp_path / "test.csv").unlink()
    generated = generator.generate(stage)
    text = generated.read_text(encoding="utf-8")
    assert "independente permanece pendente" in text
    assert "EstadoDecisaoResumoEN" in text and "decision_capable" not in text
    assert "Diagnósticos da margem" in text and "IC descritivo" in text
    assert "\\%" in text
    assert (generator.ARTICLE / "figuras/cobertura_preservacao_validacao.pdf").is_file()
    assert (generator.ARTICLE / "dados/resultados_nova_avaliacao.tex").read_text(encoding="utf-8") == "UNCHANGED FIT SECTION"


def test_generator_no_useful_rule_is_unavailable_policy_not_approved_reject_all(frozen_model, tmp_path, monkeypatch, generator):
    _, stage, result = completed_synthetic_stage(frozen_model, tmp_path, monkeypatch, minimum_coverage=1.)
    assert result["policy"]["status"] == "no_useful_rule"
    generated = generator.generate(stage)
    text = generated.read_text(encoding="utf-8")
    assert "Nenhum limiar foi selecionado" in text and "teste final reservado" in text
    assert "Sem regra/FULL" in text and "indisponível" in text
    assert "não representa uma política de rejeição total aprovada" in text
    assert "risco zero" not in text
    assert text.count("\\begin{table}") == text.count("\\end{table}")


@pytest.mark.parametrize("outcome", ["passed", "failed"])
def test_generator_final_reports_concrete_test_results_and_correct_operational_status(
        frozen_model, tmp_path, monkeypatch, generator, outcome):
    _, stage, selection = completed_synthetic_stage(frozen_model, tmp_path, monkeypatch)
    frame = pd.read_csv(tmp_path / "test.csv", float_precision="round_trip")
    if outcome == "passed":
        # Preserve all selected cases and all but one unselected case. The
        # empirical rule passes while clustered gain uncertainty includes zero.
        frame.loc[frame.index != 1, ["dockq_reduced", "dockq_full"]] = [.8, .8]
    else:
        frame[["dockq_reduced", "dockq_full"]] = [.1, .9]
    quality_root = tmp_path / "final_quality"
    frame = bind_standard_quality_records(frame, quality_root, QUALITY)
    for arm in ("reduced", "full"):
        field = f"quality_{arm}_record_path"
        frame[field] = frame[field].map(lambda p: str(Path(quality_root.name) / p))
    final_input = tmp_path / "final_input.csv"
    frame.to_csv(final_input, index=False)
    final_stage = tmp_path / "final_stage"
    evaluation = service.evaluate_frozen_policy(selection["policy_path"], final_input, final_stage / "decision")
    report = evaluation["report"]
    assert report["status"] == outcome
    assert report["decision_capable"] is (outcome == "passed")
    summary = {key: report[key] for key in ("status", "selected", "baseline", "difference", "criteria", "decision_capable")}
    summary.update(role="test", report_path=evaluation["report_path"])
    (final_stage / "stage_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    target = generator.generate(stage, test_stage=final_stage)
    text = target.read_text(encoding="utf-8")
    a = report["selected"]
    assert f"No teste final, {a['preserved_runs']}/{a['accepted_runs']}" in text
    assert f"cobertura {a['accepted_runs']}/{a['planned_runs']}" in text
    assert "On the final test" in text and "final_report_sha256=" in text
    assert "independente permanece pendente" not in text
    assert "independent final evaluation remains pending" not in text
    if outcome == "passed":
        assert report["uncertainty"]["preservation_gain_ci95"][0] <= 0
        assert "superioridade permanece incerta" in text
        assert "atendeu aos critérios empíricos" in text
    else:
        assert "não habilita decisão operacional" in text
        assert "FULL fallback remains" in text
    # A wrong final claim must fail before rewriting either the section or plot.
    previous = target.read_bytes()
    figure = generator.ARTICLE / "figuras/cobertura_preservacao_validacao.pdf"
    figure_before = figure.read_bytes()
    claim_path = Path(selection["policy_path"]).parent / "final_evaluation_claim.json"
    claim = json.loads(claim_path.read_bytes())
    claim["report_sha256"] = "0" * 64
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    with pytest.raises(ValueError, match="linked single final evaluation"):
        generator.generate(stage, test_stage=final_stage)
    assert target.read_bytes() == previous and figure.read_bytes() == figure_before

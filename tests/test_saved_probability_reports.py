import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from docking_confidence.application.demo import make_demo
from docking_confidence.application.probability_reporting import DiagnosticOptions, saved_test_diagnostics
from docking_confidence.application.train import train
from docking_confidence.application.training_results import load_training_result
from docking_confidence.audit.manifest import sha256_file
from docking_confidence.io.config import load_config


@pytest.fixture(scope="module")
def new_training_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("diagnostic_bundle")
    dataset = root / "synthetic.csv"
    make_demo(str(dataset), seed=99, n_complexes=25, runs_per_complex=8)
    config = load_config(Path(__file__).parents[1] / "configs/demo.yaml")
    config["diagnostics"] = {"bootstrap_repetitions": 100, "bootstrap_seed": 8, "confidence_level": 0.9}
    out = root / "model"
    train(str(dataset), config, str(out))
    return out


def test_diagnostics_are_persisted_inside_existing_verified_bundle(new_training_bundle):
    result = load_training_result(new_training_bundle)
    diagnostics = result["diagnostics"]
    assert diagnostics == result["report"]["diagnostics"]
    assert len(result["paths"]) == 8
    assert set(diagnostics["descriptor_comparison"]["variants"]) == {"population", "interface", "combined"}
    assert diagnostics["probability"]["bootstrap"]["bootstrap_repetitions"] == 100
    assert result["artifact"]["decision_capable"] is False
    assert diagnostics["operational_cost_authorization_applied"]["accepted_row_count"] == 0
    assert diagnostics["operational_cost_authorization_applied"]["saved_sec"] < 0
    json.dumps(result["report"], allow_nan=False)


def test_rehashed_probability_report_still_must_reproduce_frozen_predictions(tmp_path, new_training_bundle):
    out = tmp_path / "copy"
    shutil.copytree(new_training_bundle, out)
    report_path = out / "validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["diagnostics"]["probability"]["evaluations"]["after_venn_abers"]["metrics"]["roc_auc"] = -1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest_path = out / "audit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["bundle_files"]:
        if entry["path"] == "validation_report.json":
            entry.update(size_bytes=report_path.stat().st_size, sha256=sha256_file(report_path))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostics disagree"):
        load_training_result(out)


def _frame():
    return pd.DataFrame({"complex_fingerprint_sha256": np.repeat(["g1", "g2", "g3", "g4"], 2),
        "preserved": np.tile([0, 1], 4), "base_raw_score": np.tile([-1, 1], 4),
        "probability": np.tile([0.2, 0.8], 4), "probability_lower": np.tile([0.1, 0.7], 4),
        "in_applicability_domain": True, "accepted_by_locked_policy": np.tile([False, True], 4),
        "runtime_reduced_sec": 1, "runtime_full_sec": 5, "partition": "locked_test"})


def test_probability_ui_renders_curves_and_fallback_cost_without_training():
    diagnostics = saved_test_diagnostics(_frame(), options=DiagnosticOptions(100))
    app = AppTest.from_string(
        "import streamlit as st\nfrom docking_confidence.ui.training_page import _render_probability_diagnostics\n"
        "_render_probability_diagnostics(st.session_state['diagnostics'])")
    app.session_state["diagnostics"] = diagnostics
    app.run(timeout=20)
    assert not app.exception
    assert len(app.get("vega_lite_chart")) == 4
    assert "PR-AUC (trapézios)" in app.dataframe[0].value["Métrica"].values
    assert any("Overhead não medido" in caption.value for caption in app.caption)


def test_saved_csv_script_creates_new_retrospective_report_without_training(tmp_path):
    source = tmp_path / "predictions.csv"
    destination = tmp_path / "diagnostics.json"
    _frame().to_csv(source, index=False)
    command = [sys.executable, str(Path(__file__).parents[1] / "scripts/report_saved_predictions.py"),
        "--predictions", str(source), "--output", str(destination), "--bootstrap-repetitions", "100"]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["source"]["verification"] == "standalone_csv_provenance_not_verified"
    assert report["operational_cost"]["paired_strategy_sec"] == 28
    assert source.read_bytes()
    original = destination.read_bytes()
    repeat = subprocess.run(command, text=True, capture_output=True)
    assert repeat.returncode != 0
    assert destination.read_bytes() == original

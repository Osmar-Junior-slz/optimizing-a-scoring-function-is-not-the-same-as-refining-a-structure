"""One IVAP, two disjoint data roles, and probabilities without decision gates."""
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from docking_confidence.application.legacy_ivap import (
    SCHEMA, create_calibrator, create_from_source, load_calibrator, proper_calibration_partition,
)
from docking_confidence.application.predict import predict
from docking_confidence.calibration.venn_abers import VennAbersCalibrator
from test_training_context import _context_frame, _local_config




def _dataset(tmp_path):
    config = _local_config()
    # The old policy/evidence thresholds have no role in fitting this predictor.
    config["calibration"]["minimum_independent_complexes"] = 10000
    dataset = tmp_path / "dataset.csv"
    frame = _context_frame(dataset, config, fifty=True)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    frame["config"] = str(config_file)
    frame.to_csv(dataset, index=False)
    return dataset, config, frame, config_file


def test_ivap_only_fits_one_calibrator_with_two_disjoint_roles(tmp_path, monkeypatch):
    from docking_confidence.risk import policy
    monkeypatch.setattr(policy, "select_policy", lambda *a, **k: pytest.fail("must not select a policy"))
    dataset, config, frame, _ = _dataset(tmp_path)
    result = create_calibrator(dataset, config, tmp_path / "ivap")
    artifact = result["artifact"]
    assert artifact["schema_version"] == SCHEMA
    assert set(artifact["partition"]) == {"proper_training", "calibration"}
    train, cal = [set(artifact["partition"][r]["groups"]) for r in ["proper_training", "calibration"]]
    assert not train & cal
    assert train | cal == set(frame.complex_fingerprint_sha256)
    assert result["report"]["calibrator_count"] == 1
    assert all(key not in artifact for key in ["policy", "decision_capable", "runtime_gate", "comparators"])
    assert artifact["calibration"]["calibration_example_count"] == artifact["partition"]["calibration"]["row_count"]
    assert not (tmp_path / "ivap/policy_candidates.csv").exists()
    restored = load_calibrator(result["paths"]["artifact.json"])
    assert restored["artifact"] == artifact
    output = predict(result["paths"]["artifact.json"], str(dataset), str(tmp_path / "probabilities.csv"))
    assert output.p0.between(0, 1).all() and (output.p0 < output.p1).all()
    np.testing.assert_allclose(output.probability, output.p1 / (1 - output.p0 + output.p1))
    assert all(c not in output for c in ["action", "authorized_threshold", "dockq_full", "dockq_reduced", "preserved"])
    # Reference outcomes are not needed at inference time.
    reduced = tmp_path / "reduced_only.csv"
    frame.drop(columns=["dockq_full", "dockq_reduced", "runtime_full_sec", "engine_full", "scientific_full"]).to_csv(reduced, index=False)
    independent = predict(result["paths"]["artifact.json"], str(reduced), str(tmp_path / "reduced_probabilities.csv"))
    np.testing.assert_array_equal(output.probability, independent.probability)


def test_calibration_labels_do_not_train_the_base_model(tmp_path):
    dataset, config, frame, _ = _dataset(tmp_path)
    first = create_calibrator(dataset, config, tmp_path / "first")
    _, calibration = proper_calibration_partition(frame.complex_fingerprint_sha256)
    frame.loc[calibration, "dockq_reduced"] = 0.0
    frame.loc[calibration, "dockq_full"] = 0.9
    frame.to_csv(dataset, index=False)
    second = create_calibrator(dataset, config, tmp_path / "second")
    assert first["artifact"]["base_model"] == second["artifact"]["base_model"]
    assert second["artifact"]["calibration"]["calibration_positive_count"] == 0
    assert first["artifact"]["calibration"] != second["artifact"]["calibration"]


@pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 1], [0, 1, 1]])
def test_ivap_endpoints_match_paper_two_hypothetical_label_fits(labels):
    scores = np.array([-1.0, 0.0, 1.0])
    queries = np.array([-3.0, 0.0, 0.5, 4.0])
    fitted = VennAbersCalibrator().fit(scores, np.array(labels), allow_single_class=True)
    p0, p1, _ = VennAbersCalibrator.from_export(fitted.export(), allow_single_class=True).predict_interval(queries)
    for i, query in enumerate(queries):
        for label, endpoint in [(0, p0[i]), (1, p1[i])]:
            iso = IsotonicRegression(out_of_bounds="clip").fit(np.r_[scores, query], np.r_[labels, label])
            assert endpoint == pytest.approx(iso.predict([query])[0])


def test_result_zip_creates_calibrator_without_docking_or_comparison(tmp_path):
    dataset, _, _, _ = _dataset(tmp_path)
    archive_path = tmp_path / "results.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.write(dataset, "study/analise_artigo/calibrador_previsoes_e_qualidade.csv")
    result = create_from_source(archive_path, tmp_path / "ivap")
    assert result["verified"] and result["report"]["calibrator_count"] == 1
    with pytest.raises(FileExistsError):
        create_from_source(archive_path, tmp_path / "ivap")
    calibration = Path(result["paths"]["calibration_examples.csv"])
    calibration.write_bytes(calibration.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="mismatch"):
        load_calibrator(result["paths"]["artifact.json"])

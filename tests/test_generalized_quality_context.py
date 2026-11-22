import copy
import pandas as pd
import pytest

from docking_confidence.application.generalized_calibrator import create_calibrator, load_calibrator
from docking_confidence.io.config import config_sha256
from docking_confidence.io.quality_context import (quality_metric_sha256,
    validate_paired_quality_context, validate_quality_metric)
from test_generalized_creation import _dataset
from standard_quality_fixtures import bind_standard_quality_records


def metric():
    return {"schema_version":"structural_quality_metric_v1",
        "method":"official_dockq_v2_protein_protein","implementation":"DockQ",
        "implementation_version":"2.1.3","chain_mapping":"explicit_receptor_ligand",
        "capri_peptide":False,"contact_cutoff_angstrom":5,"interface_cutoff_angstrom":10}


def test_historical_labels_cannot_be_renamed_by_changing_config(tmp_path):
    path, config, frame, _ = _dataset(tmp_path)
    config["quality_metric"] = metric()
    frame["config_sha256"] = config_sha256(config)
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError, match="historical scores"):
        create_calibrator(path,config,tmp_path/"forbidden")
    assert not (tmp_path/"forbidden/artifact.json").exists()


@pytest.mark.parametrize("change", [{"capri_peptide":True},{"interface_cutoff_angstrom":8},
    {"contact_cutoff_angstrom":5.0},{"implementation_version":""}])
def test_metric_contract_rejects_convention_and_type_changes(change):
    value=metric();value.update(change)
    with pytest.raises(ValueError):validate_quality_metric(value)


def test_paired_metric_digest_must_match_every_row():
    expected=metric()
    frame=pd.DataFrame({"quality_metric_id":[expected["method"]]*2,
        "quality_metric_sha256":[quality_metric_sha256(expected),"0"*64]})
    with pytest.raises(ValueError,match="digest"):
        validate_paired_quality_context(frame,expected)


def test_new_artifact_binds_metric_and_execution_contract(tmp_path):
    path, config, frame, _ = _dataset(tmp_path)
    config["quality_metric"]=metric()
    config["execution_contract"]={"schema_version":"operational_cascade_v1",
        "reduced_evaluation_budget":5700,"full_evaluation_budget":7600,"population":100,
        "initialization":"uniform","reduced_search_scale":.5,"max_translation":5.,"max_rotation_degrees":25.}
    frame["config_sha256"]=config_sha256(config)
    frame["quality_metric_id"]=config["quality_metric"]["method"]
    frame["quality_metric_sha256"]=quality_metric_sha256(config["quality_metric"])
    frame=bind_standard_quality_records(frame,path.parent,config["quality_metric"])
    frame.to_csv(path,index=False)
    result=create_calibrator(path,config,tmp_path/"standard")
    settings=load_calibrator(result["paths"]["artifact.json"])["artifact"]["settings"]
    assert settings["quality_metric"] == config["quality_metric"]
    assert settings["execution_contract"] == config["execution_contract"]

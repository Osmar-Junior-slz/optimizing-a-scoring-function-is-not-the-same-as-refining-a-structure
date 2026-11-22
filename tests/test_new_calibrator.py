"""A usable probability estimator must never be mistaken for an authorized gate."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

import pandas as pd
import pytest

from docking_confidence.application.train import TrainingEvidenceError,train
from docking_confidence.application.predict import load_prediction_session
from docking_confidence.artifact.schema import validate_artifact
from docking_confidence.io.config import config_sha256
from test_training_context import _context_frame,_local_config


def test_development_fit_preserves_unmet_requirements_and_verified_probabilities(tmp_path):
    cfg=_local_config()
    cfg['calibration'].update(minimum_independent_complexes=100,fit_scope='all_calibration_rows')
    dataset=tmp_path/'dataset.csv';frame=_context_frame(dataset,cfg,fifty=True)
    with pytest.raises(TrainingEvidenceError) as exc:
        train(str(dataset),cfg,str(tmp_path/'strict'))
    assert exc.value.stage=='probability_calibration'
    result=train(str(dataset),cfg,str(tmp_path/'candidate'),development_only=True)
    session=load_prediction_session(result['artifact'],require_bundle=True)
    artifact=session.artifact
    assert artifact['development_only'] is True
    assert artifact['calibration_evidence_passed'] is False
    assert artifact['resolved_config']['calibration']['minimum_independent_complexes']==100
    assert artifact['decision_capable'] is False
    predictions=session.predict_bytes(dataset.read_bytes())
    assert predictions['probability'].between(0,1).all()
    assert predictions['probability_lower'].le(predictions['probability_upper']).all()
    assert set(predictions.action)=={'RUN_FULL'}
    forged=deepcopy(artifact);forged['decision_capable']=True
    with pytest.raises(ValueError,match='development-only'):
        validate_artifact(forged)
    forged=deepcopy(artifact);forged['development_only']=False
    with pytest.raises(ValueError,match='calibration complex-level'):
        validate_artifact(forged)


def test_development_fit_cannot_enable_automatic_acceptance(tmp_path):
    cfg=_local_config();cfg['authorization']['allow_automatic_decision']=True
    with pytest.raises(ValueError):
        train(str(tmp_path/'unused.csv'),cfg,str(tmp_path/'out'),development_only=True)


def test_source_contract_merges_disjoint_verified_batches_and_rejects_tampering(tmp_path):
    from docking_confidence.application.calibrated_batch import CalibratedBatchGate
    from docking_confidence.application.batch import _atomic_write_json,_manifest_for_plan
    parts=[];frames=[];batch_hashes=[];config_hash='c'*64
    for index in range(2):
        folder=tmp_path/f'part{index}';folder.mkdir()
        frame=pd.DataFrame([{'complex_fingerprint_sha256':'a'*63+str(index),'algorithm':'ag','seed':1}])
        csv=folder/'data.csv';frame.to_csv(csv,index=False);frames.append(pd.read_csv(csv))
        manifest=_manifest_for_plan({'engine':{'name':'fixture'},'jobs':[{'same':'profile'}],'mobile_partner_domain':{}})
        _atomic_write_json(folder/'batch_manifest.json',manifest);batch_hashes.append(manifest['manifest_sha256'])
        report={'valid':True,'config_sha256':config_hash,'dataset_path':str(csv),
            'dataset_sha256':hashlib.sha256(csv.read_bytes()).hexdigest(),'batch_dir':str(folder),
            'batch_manifest_sha256':manifest['manifest_sha256']}
        report_path=folder/'dataset_build_report.json';_atomic_write_json(report_path,report);parts.append(report_path)
    destination=tmp_path/'merged';destination.mkdir()
    dataset=destination/'dataset.csv'
    pd.concat(frames,ignore_index=True).sort_values(['complex_fingerprint_sha256','algorithm','seed'],kind='stable').to_csv(dataset,index=False,lineterminator='\n')
    digest=hashlib.sha256(dataset.read_bytes()).hexdigest()
    _atomic_write_json(destination/'dataset_build_report.json',{'valid':True,'dataset_sha256':digest,'config_sha256':config_hash,
        'source_kind':'multiple_hash_verified_training_batches_v1','source_preparations':list(map(str,parts)),
        'source_preparation_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in parts},
        'source_batch_manifest_sha256':batch_hashes})
    gate=CalibratedBatchGate.__new__(CalibratedBatchGate)
    gate.artifact={'training_dataset_snapshot':{'path':str(dataset),'sha256':digest},'training_config_source':{'sha256':config_hash}}
    contract,issue=gate._execution_contract()
    assert issue is None and contract['source_batch_manifest_sha256']==batch_hashes
    parts[0].write_bytes(parts[0].read_bytes()+b' ')
    contract,issue=gate._execution_contract()
    assert contract is None and 'preparation changed' in issue

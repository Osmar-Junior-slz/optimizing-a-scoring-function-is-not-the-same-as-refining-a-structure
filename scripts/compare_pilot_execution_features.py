"""Diagnose whether available REDUCED execution evidence adds predictive signal."""
from pathlib import Path
import os
for _key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_key,"1")
import argparse
import json
import hashlib
import sys
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from docking_confidence.features.pilot_execution import (
    extract_pilot_execution_features,PILOT_EXECUTION_FEATURE_COLUMNS,
)
from docking_confidence.application.pilot_diagnostics import (
    PilotCandidate,compare_pilot_candidates,PILOT_ALGORITHM_COLUMNS,
)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--preparation-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    source,out=Path(args.preparation_dir),Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    preparation=json.loads((source/"dataset_build_report.json").read_text(encoding="utf-8"))
    if not preparation["valid"]:
        raise ValueError("pilot requires a verified dataset")
    snapshot=Path(preparation["dataset_path"]).read_bytes()
    if hashlib.sha256(snapshot).hexdigest()!=preparation["dataset_sha256"]:
        raise ValueError("dataset snapshot changed")
    data=pd.read_csv(preparation["dataset_path"])
    root=Path(preparation["batch_dir"])
    traces=[]
    for index,row in data.iterrows():
        directory=root/"targets"/row.complex_id/"jobs"/row.run_id
        receipt=json.loads((directory/"receipt.json").read_text(encoding="utf-8-sig"))
        features,trace=extract_pilot_execution_features(directory,receipt)
        for name,value in features.items():
            data.loc[index,name]=value
        traces.append({"complex_id":row.complex_id,"run_id":row.run_id,**trace})
    for column in PILOT_ALGORITHM_COLUMNS:
        data[column]=(data.algorithm==column.removeprefix("algorithm_")).astype(int)
    legacy=preparation["feature_columns"]
    families={
        "input_retention_plus_algorithm":legacy+list(PILOT_ALGORITHM_COLUMNS)+["pilot_input_pose_retention"],
        "execution_plus_algorithm":legacy+list(PILOT_ALGORITHM_COLUMNS)+list(PILOT_EXECUTION_FEATURE_COLUMNS),
    }
    candidates=(
        PilotCandidate("balanced_C1",1,"balanced"),
        PilotCandidate("unweighted_C0.1",.1,None),
        PilotCandidate("unweighted_C1",1,None),
        PilotCandidate("forest_depth3",1,None,model="random_forest"),
        PilotCandidate("boosting_depth2",1,None,model="hist_gradient_boosting"),
    )
    plan={
        "scope":"exploratory_third_stage_execution_representation_diagnostics",
        "families":families,"models":[v.__dict__ for v in candidates],
        "descriptor_reference_warning":"Input-pose retention is a predictor relative to the provided peptide. It is NEVER the experimental-reference outcome.",
        "source_traces":traces,"domain_features":legacy,
        "authorization_changed":False,"dataset_sha256":preparation["dataset_sha256"],
    }
    (out/"comparison_plan.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
    for family,columns in families.items():
        report=compare_pilot_candidates(data,columns,candidates=candidates,domain_columns=legacy)
        report["feature_family"]=family
        report["dataset_sha256"]=preparation["dataset_sha256"]
        (out/f"{family}.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
        for name,result in report["candidates"].items():
            before,after=result["before"],result["after_on_common_rows"]
            print(json.dumps({"family":family,"model":name,
                "raw_brier":before["preservation"]["brier_score"],"raw_auc":before["preservation"]["roc_auc"],
                "VA_brier":after["preservation"]["brier_score"],"VA_auc":after["preservation"]["roc_auc"],
                "violation_ap":after["violation"]["average_precision"]}),flush=True)


if __name__=="__main__":
    main()

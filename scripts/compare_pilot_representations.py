"""Second diagnostic stage: fixed capacity/representation comparisons."""
from pathlib import Path
import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_name, "1")
import argparse
import json
import hashlib
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docking_confidence.application.pilot_diagnostics import (
    PilotCandidate, compare_pilot_candidates, PILOT_ALGORITHM_COLUMNS,
)
from docking_confidence.features.contact_graph import CONTACT_GRAPH_FEATURE_COLUMNS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--preparation-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    p,out=Path(args.preparation_dir),Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    prep=json.loads((p/"dataset_build_report.json").read_text(encoding="utf-8"))
    if not prep["valid"]:
        raise ValueError("invalid dataset preparation")
    snapshot=Path(prep["dataset_path"]).read_bytes()
    if hashlib.sha256(snapshot).hexdigest()!=prep["dataset_sha256"]:
        raise ValueError("dataset snapshot changed")
    data=pd.read_csv(prep["dataset_path"])
    legacy=prep["feature_columns"]
    for column in PILOT_ALGORITHM_COLUMNS:
        data[column]=(data.algorithm==column.removeprefix("algorithm_")).astype(int)
    families={
        "legacy_11":legacy,
        "legacy_plus_algorithm":list(legacy)+list(PILOT_ALGORITHM_COLUMNS),
        "full_graph_plus_algorithm":list(dict.fromkeys(
            list(legacy)+list(CONTACT_GRAPH_FEATURE_COLUMNS)+list(PILOT_ALGORITHM_COLUMNS))),
    }
    candidates=(
        PilotCandidate("balanced_C1",1.0,"balanced"),
        PilotCandidate("unweighted_C0.1",.1,None),
        PilotCandidate("unweighted_C1",1.0,None),
        PilotCandidate("forest_depth3",1.0,None,model="random_forest"),
        PilotCandidate("boosting_depth2",1.0,None,model="hist_gradient_boosting"),
    )
    plan={"scope":"exploratory_second_stage_after_simple_candidate_diagnostics",
          "feature_families":families,"models":[c.__dict__ for c in candidates],
          "domain":"unchanged original 11-feature domain for all candidate representations",
          "selection_warning":"No production model is selected or authorized by this diagnostic",
          "reference_data_for_features":False,"dataset_sha256":prep["dataset_sha256"]}
    (out/"comparison_plan.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
    for family,columns in families.items():
        report=compare_pilot_candidates(data,columns,candidates=candidates,domain_columns=legacy)
        report["dataset_sha256"]=prep["dataset_sha256"]
        report["feature_family"]=family
        (out/f"{family}.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
        for name,value in report["candidates"].items():
            b,a=value["before"],value["after_on_common_rows"]
            print(json.dumps({"family":family,"model":name,"n":report["counts"]["common_calibrated_rows"],
                "raw_brier":b["preservation"]["brier_score"],"raw_auc":b["preservation"]["roc_auc"],
                "raw_violation_ap":b["violation"]["average_precision"],
                "VA_brier":a["preservation"]["brier_score"],"VA_auc":a["preservation"]["roc_auc"],
                "VA_violation_ap":a["violation"]["average_precision"]}),flush=True)


if __name__=="__main__":
    main()

"""Post-hoc experimental quality of unchanged input poses; NEVER model features."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
for _name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_name,"1")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from docking_confidence.evaluation.reference_dockq import evaluate_reference_pose,read_engine_order_atoms


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--input-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    manifest=json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    root,out=Path(args.input_dir).resolve(),Path(args.output_dir).resolve()
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for target in manifest["plan"]["targets"]:
        paths={role:root/target[role+"_path"] for role in ["receptor","ligand","reference"]}
        for role,path in paths.items():
            if hashlib.sha256(path.read_bytes()).hexdigest()!=target[role+"_sha256"]:
                raise ValueError("supplied input changed: "+str(path))
        directory=out/target["complex_id"];directory.mkdir()
        pose=directory/"unchanged_input_pose.json"
        pose.write_text(json.dumps({
            "schema_version":"1.0","pose_role":"best_computed_pose",
            "engine_name":"input_identity_diagnostic_no_search",
            "search_space_mode":"unchanged_supplied_input",
            "coords":[atom.coord for atom in read_engine_order_atoms(paths["ligand"])],
            "meta":{"diagnostic_role":"unchanged_input_pose","search_executed":False,
                    "engine_pose_schema_used_only_for_evaluator_compatibility":True},
        }),encoding="utf-8")
        metric=evaluate_reference_pose(
            receptor_model_pdb=paths["receptor"],ligand_template_pdb=paths["ligand"],
            reference_complex_pdb=paths["reference"],receptor_chain=target["receptor_chain"],
            ligand_chain=target["ligand_chain"],pose_json=pose,output_dir=directory,
        )
        rows.append({"complex_id":target["complex_id"],
                     "fingerprint":target["complex_fingerprint_sha256"],
                     "input_experimental_dockq":metric.dockq,
                     "input_experimental_irmsd":metric.irmsd,
                     "input_experimental_lrmsd":metric.lrmsd})
        print(target["complex_id"],round(metric.dockq,4),flush=True)
    summary={
        "scope":"post_hoc_input_quality_audit_NOT_A_MODEL_FEATURE_OR_SEARCH",
        "experimental_reference_used_for_audit":True,"experimental_reference_used_for_prediction":False,
        "rows":rows,"n":len(rows),
        "input_above_023":sum(row["input_experimental_dockq"]>=.23 for row in rows),
        "mean_input_dockq":sum(row["input_experimental_dockq"] for row in rows)/len(rows),
    }
    (out/"input_quality_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="rows"}),flush=True)


if __name__=="__main__":
    main()

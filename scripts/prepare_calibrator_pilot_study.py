"""Verify completed pilot shards and prepare both matched initialization datasets."""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import sys
import time

for _name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_name,"1")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))


def prepare_one(arguments):
    from docking_confidence.application.training_dataset import build_training_dataset
    batch,out,config,initialization=arguments
    directory=Path(out)/Path(batch).name/initialization
    report_path=directory/"dataset_build_report.json"
    if report_path.exists():
        report=json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report=build_training_dataset(
            batch,directory,profile="pilot",fingerprint_alias_policy="first_complex_id",
            factorial_initialization=initialization,config_path=config,
        )
    if not report["valid"]:
        raise ValueError(f"pilot preparation failed: {report_path}: {report['issues'][:3]}")
    return report


def merge_reports(reports,out,config,initialization):
    import pandas as pd
    from docking_confidence.io.config import load_config
    from docking_confidence.io.dataset import load_paired_dataset
    from io import BytesIO
    destination=Path(out)/f"merged_{initialization}"
    destination.mkdir(parents=True,exist_ok=False)
    frames=[];seen=set()
    for report in reports:
        path=Path(report["dataset_path"]);payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=report["dataset_sha256"]:
            raise ValueError("source pilot dataset changed after preparation")
        frame=pd.read_csv(path)
        groups=set(frame.complex_fingerprint_sha256)
        if seen&groups:
            raise ValueError("a molecular fingerprint appears in two pilot shards")
        seen|=groups;frames.append(frame)
    frame=pd.concat(frames,ignore_index=True)
    frame=frame.sort_values(["complex_fingerprint_sha256","algorithm","seed"],kind="stable")
    csv=frame.to_csv(index=False,lineterminator="\n").encode("utf-8")
    resolved=load_config(config)
    checked=load_paired_dataset(
        BytesIO(csv),resolved["features"]["columns"],resolved["contact_graph"],
        docking_task="local_refinement",feature_schema=resolved["features"]["schema_version"],
    )
    path=destination/"paired_dataset.csv";path.write_bytes(csv)
    report={
        "schema_version":"paired_batch_training_dataset_v1","valid":True,
        "source_kind":"multiple_hash_verified_pilot_batches",
        "production_execution_contract_eligible":False,
        "scope":"50_complex_development_pilot_not_the_definitive_model",
        "batch_dir":str(Path(reports[0]["batch_dir"]).parent),
        "dataset_path":str(path.resolve()),"dataset_sha256":hashlib.sha256(csv).hexdigest(),
        "feature_columns":resolved["features"]["columns"],
        "feature_schema":resolved["features"]["schema_version"],
        "config_path":str(Path(config).resolve()),"config_sha256":reports[0]["config_sha256"],
        "docking_task":"local_refinement","experimental_initialization":initialization,
        "source_preparations":[r["report_path"] for r in reports],
        "source_dataset_hashes":[r["dataset_sha256"] for r in reports],
        "counts":{"usable_rows":len(checked),"independent_complexes":len(seen),
                  "planned_complex_ids":sum(r["counts"]["complexes_found"] for r in reports),
                  "excluded_alias_targets":sum(r["counts"]["excluded_alias_targets"] for r in reports)},
        "failures":[x for r in reports for x in r["failures"]],
        "issues":[],"loader_passed":True,
    }
    (destination/"dataset_build_report.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    (destination/"dataset_config.json").write_text(json.dumps(resolved,indent=2),encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--study-root",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--config",required=True)
    parser.add_argument("--wait",action="store_true")
    args=parser.parse_args()
    root,out=Path(args.study_root).resolve(),Path(args.output_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    batches=[root/f"shard_{i:02d}" for i in range(1,9)]
    submitted=set();futures={};reports={"uniform":[],"seeded":[]}
    with ProcessPoolExecutor(max_workers=2) as pool:
        while len(submitted)<len(batches):
            for batch in batches:
                if batch in submitted:continue
                try:status=json.loads((batch/"batch_status.json").read_text(encoding="utf-8"))
                except (OSError,ValueError):continue
                if status["state"] in {"COMPLETED","COMPLETED_WITH_ERRORS"}:
                    # Receipts must attest the same effective budget, including
                    # successful arms that are not chosen for a training pair.
                    for receipt_path in batch.glob("targets/*/jobs/*/receipt.json"):
                        receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
                        result=receipt["engine_result"]
                        if result.get("actual_eval_count")!=7600 or result.get("stop_reason")!="objective_evaluation_ceiling":
                            raise ValueError(f"unverified effective budget: {receipt_path}")
                    for initialization in reports:
                        future=pool.submit(prepare_one,(str(batch),str(out),args.config,initialization))
                        futures[future]=initialization
                    submitted.add(batch)
                    print(f"Preparing completed {batch.name}",flush=True)
            if len(submitted)<len(batches):
                if not args.wait:
                    raise RuntimeError("pilot batch is not yet complete")
                time.sleep(5)
        for future,initialization in futures.items():
            reports[initialization].append(future.result())
    for initialization,items in reports.items():
        merged=merge_reports(items,out,args.config,initialization)
        print(json.dumps({"initialization":initialization,**merged["counts"]}),flush=True)


if __name__=="__main__":
    main()

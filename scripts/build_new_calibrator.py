"""Collect new operational training pairs; never runs the 100-complex experiment."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
from docking_confidence.application.batch import _atomic_write_json,run_batch
from docking_confidence.application.operational import load_operational_config
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm,ProtocolMode
from docking_confidence.io.batch_discovery import discover_ppi4dock


def fit_and_install(output: Path):
    import hashlib
    from docking_confidence.application.training_dataset import build_training_dataset
    from docking_confidence.application.train import train
    from docking_confidence.application.predict import load_prediction_session
    from docking_confidence.application.calibrated_batch import CalibratedBatchGate
    from docking_confidence.io.config import load_config
    from prepare_calibrator_pilot_study import merge_reports
    root=Path(__file__).resolve().parents[1]
    config_path=root/"configs/local_refinement_operational_candidate.yaml"
    config=load_config(config_path)
    preparations=[];preparation_root=output/"training_preparations"
    from collections import Counter
    collection_counts=Counter()
    effective_evaluation_receipts=Counter()
    selected_complexes=[]
    for folder in sorted(output.glob("shard_*")):
        batch=folder/"batch"
        status=json.loads((batch/"batch_status.json").read_text(encoding="utf-8"))
        if status["state"] not in {"COMPLETED","COMPLETED_WITH_ERRORS"}:
            raise ValueError("finish the training collection before fitting")
        collection_counts.update(status["counts"])
        selected_complexes.extend(json.loads((folder/"selection.json").read_text(encoding="utf-8"))["complexes"])
        for receipt_path in batch.glob("targets/*/jobs/*/receipt.json"):
            receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
            engine_result=receipt["engine_result"]
            mode=engine_result["protocol"]
            expected=config["execution_contract"][f"{mode}_evaluation_budget"]
            actual=engine_result.get("actual_eval_count")
            if actual != expected or engine_result.get("stop_reason") != "objective_evaluation_ceiling":
                raise ValueError(f"training evaluation budget was not met: {receipt_path}")
            effective_evaluation_receipts[f"{mode}_{actual}"]+=1
        report_path=preparation_root/folder.name/"dataset_build_report.json"
        if report_path.exists():
            report=json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report=build_training_dataset(batch,report_path.parent,profile="operational",
                fingerprint_alias_policy="first_complex_id",config_path=config_path)
        if not report["valid"]:raise ValueError(f"invalid training pairs: {report_path}: {report['issues']}")
        preparations.append(report)
    merged_path=preparation_root/"merged_uniform/dataset_build_report.json"
    if merged_path.exists():
        merged=json.loads(merged_path.read_text(encoding="utf-8"))
    else:
        merged=merge_reports(preparations,preparation_root,config_path,"uniform")
        merged.update(source_kind="multiple_hash_verified_training_batches_v1",
            source_preparation_hashes={r["report_path"]:hashlib.sha256(Path(r["report_path"]).read_bytes()).hexdigest() for r in preparations},
            source_batch_manifest_sha256=[r["batch_manifest_sha256"] for r in preparations],
            scope="new_5700_7600_calibrator_development_only; no independent authorization claimed")
        _atomic_write_json(merged_path,merged)
    destination=root/"models/calibrador_50_20261115"
    artifact_path=destination/"artifact.json"
    if not artifact_path.exists():
        result=train(merged["dataset_path"],config,str(destination),expected_dataset_sha256=merged["dataset_sha256"],
            development_only=True,progress_callback=lambda f,m:print(f"{f:.0%} {m}",flush=True))
    else:result={"artifact":str(artifact_path)}
    session=load_prediction_session(artifact_path,require_bundle=True)
    if session.artifact["training_dataset_snapshot"]["sha256"] != merged["dataset_sha256"]:
        raise ValueError("existing calibrator was fitted on a different dataset")
    from docking_confidence.io.config import config_sha256
    if session.artifact["training_config_source"]["sha256"] != config_sha256(config):
        raise ValueError("existing calibrator was fitted with a different configuration")
    gate=CalibratedBatchGate(artifact_path)
    jobs=build_execution_plan(algorithms=[Algorithm.AG,Algorithm.PSO,Algorithm.HYBRID],protocol_mode=ProtocolMode.OPERATIONAL,
        base_config=load_operational_config(),seeds=[1])
    gate.validate_jobs(jobs,DockingPPLocalEngine(timeout_seconds=600))
    if gate.protocol_issues:raise ValueError(f"new calibrator execution contract failed: {gate.protocol_issues}")
    import yaml
    binding={"schema_version":"docking_calibrator_binding_v1","artifact_path":artifact_path.relative_to(root).as_posix()}
    (root/"configs/calibrator.yaml").write_text(yaml.safe_dump(binding,sort_keys=False),encoding="utf-8",newline="\n")
    summary={"artifact":str(artifact_path),"artifact_sha256":session.artifact_sha256,
        "selected_complex_count":len(selected_complexes),"collection_counts":dict(collection_counts),
        "effective_evaluation_receipts":dict(effective_evaluation_receipts),
        "budget":{"reduced":config["execution_contract"]["reduced_evaluation_budget"],
                  "full":config["execution_contract"]["full_evaluation_budget"]},
        "probability_estimator_ready":True,"automatic_decisions_authorized":session.artifact["decision_capable"],
        "development_only":session.artifact.get("development_only"),
        "calibration_evidence_passed":session.artifact.get("calibration_evidence_passed"),
        "execution_contract_issues":gate.protocol_issues,"development_summary":session.artifact["development_summary"]}
    _atomic_write_json(output/"new_calibrator_summary.json",summary)
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
    return result


def selected_shards(report,workers):
    # Select molecular representatives before any new outcomes are observed.
    canonical={};aliases=[]
    for target in sorted(report.eligible_targets,key=lambda t:t.complex_id):
        key=target.complex_fingerprint_sha256
        if key in canonical:
            aliases.append({"selected":canonical[key].complex_id,"excluded":target.complex_id,"fingerprint":key})
        else:canonical[key]=target
    partitions=[[] for _ in range(workers)];loads=[0]*workers
    for target in sorted(canonical.values(),key=lambda t:(-max(1,t.receptor_atom_count or 1)*max(1,t.ligand_atom_count or 1),t.complex_id)):
        index=min(range(workers),key=lambda i:(loads[i],i))
        partitions[index].append(target)
        loads[index]+=max(1,target.receptor_atom_count or 1)*max(1,target.ligand_atom_count or 1)
    return partitions,aliases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--worker-index",type=int)
    parser.add_argument("--fit-only",action="store_true")
    args=parser.parse_args()
    if not 1<=args.workers<=8:parser.error("workers must be between 1 and 8")
    output=Path(args.output_dir).resolve()
    if args.fit_only:
        fit_and_install(output)
        return
    if args.worker_index is None:
        output.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy()
        limits={name:"1" for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")}
        env.update(limits);env["PYTHONDONTWRITEBYTECODE"]="1";env["PYTHONUTF8"]="1"
        children=[];handles=[]
        try:
            for index in range(args.workers):
                handle=(output/f"worker_{index+1:02d}.log").open("a",encoding="utf-8");handles.append(handle)
                child=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"--input-dir",args.input_dir,
                    "--output-dir",str(output),"--workers",str(args.workers),"--worker-index",str(index)],
                    stdout=handle,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
                children.append(child)
            _atomic_write_json(output/"workers.json",{"scope":"new_calibrator_training_50_only","thread_limits":limits,
                "workers":[{"pid":p.pid,"index":i} for i,p in enumerate(children)]})
            print("Collecting 5,700 / 7,600 training pairs on up to 50 input complexes; eight workers, one thread each.",flush=True)
            codes=[child.wait() for child in children]
            _atomic_write_json(output/"worker_completion.json",{"exit_codes":codes,"completed":all(code==0 for code in codes)})
            if any(codes):raise RuntimeError(f"Collection worker failure: {codes}; inspect the worker logs")
            fit_and_install(output)
        finally:
            for handle in handles:handle.close()
        return
    if not 0<=args.worker_index<args.workers:parser.error("invalid worker index")
    report=discover_ppi4dock(args.input_dir,validate_contents=True)
    if report.has_errors or len(report.targets)>50:raise ValueError("expected the authorized folder with at most 50 valid inputs")
    partitions,aliases=selected_shards(report,args.workers)
    folder=output/f"shard_{args.worker_index+1:02d}";folder.mkdir(parents=True,exist_ok=True)
    _atomic_write_json(folder/"selection.json",{"aliases":aliases,"complexes":[t.complex_id for t in partitions[args.worker_index]],
        "rule":"first complex_id per fingerprint; size-balanced shards; no labels used"})
    report=replace(report,targets=tuple(sorted(partitions[args.worker_index],key=lambda t:t.complex_id)),issues=())
    jobs=build_execution_plan(algorithms=[Algorithm.AG,Algorithm.PSO,Algorithm.HYBRID],protocol_mode=ProtocolMode.OPERATIONAL,
        base_config=load_operational_config(),seeds=[1])
    result=run_batch(args.input_dir,folder/"batch",jobs=jobs,engine=DockingPPLocalEngine(timeout_seconds=600),
        report=report,collect_operational_pairs=True)
    print(json.dumps({"state":result.state,"succeeded":result.succeeded,"failed":result.failed}),flush=True)


if __name__=="__main__":main()

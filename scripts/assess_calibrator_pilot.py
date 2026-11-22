"""Paired pilot comparison of frozen legacy and optional input-context logic."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
for _name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_name,"1")
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from docking_confidence.application.pilot_diagnostics import PilotCandidate,compare_pilot_candidates
from docking_confidence.application.probability_reporting import _resampled_metrics
from docking_confidence.io.config import LOCAL_REFINEMENT_FEATURE_COLUMNS,LOCAL_REFINEMENT_CONTEXT_FEATURE_COLUMNS


def source_clusters(frame):
    fingerprints=sorted(frame.complex_fingerprint_sha256.unique())
    parent={value:value for value in fingerprints}
    def find(value):
        while parent[value]!=value:
            parent[value]=parent[parent[value]]
            value=parent[value]
        return value
    seen={}
    for row in frame.drop_duplicates("complex_id").itertuples():
        group=row.complex_fingerprint_sha256
        keys=["pdb:"+row.complex_id.split("_")[0].lower(),
              "sequence:"+row.receptor_sequence,"sequence:"+row.ligand_sequence]
        for key in keys:
            if key in seen:
                left,right=find(group),find(seen[key])
                low,high=sorted([left,right]);parent[high]=low
            seen[key]=group
    return {value:find(value) for value in fingerprints}


def paired_uncertainty(legacy,context,reference,y,groups,repetitions=2000):
    unique=np.unique(groups)
    members=[np.flatnonzero(groups==group) for group in unique]
    rng=np.random.default_rng(20261115)
    samples={key:[] for key in ["brier_vs_legacy","brier_vs_prevalence","auc_vs_legacy","violation_ap_vs_legacy"]}
    for _ in range(repetitions):
        idx=np.concatenate([members[i] for i in rng.integers(0,len(unique),len(unique))])
        yy=y[idx]
        old=_resampled_metrics(yy,legacy[idx]);new=_resampled_metrics(yy,context[idx])
        samples["brier_vs_legacy"].append(new["brier_score"]-old["brier_score"])
        samples["brier_vs_prevalence"].append(new["brier_score"]-float(np.mean((yy-reference[idx])**2)))
        if old["roc_auc"] is not None:
            samples["auc_vs_legacy"].append(new["roc_auc"]-old["roc_auc"])
            old_ap=_resampled_metrics(1-yy,1-legacy[idx])["average_precision"]
            new_ap=_resampled_metrics(1-yy,1-context[idx])["average_precision"]
            samples["violation_ap_vs_legacy"].append(new_ap-old_ap)
    return {key:{"ci95":np.quantile(values,[.025,.975]).tolist() if values else None,
                "valid_repetitions":len(values)} for key,values in samples.items()}


def assess(frame,output,*,group_column="complex_fingerprint_sha256",suffix="primary"):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    variants={}
    for name,columns in [("legacy",LOCAL_REFINEMENT_FEATURE_COLUMNS),
                         ("context",LOCAL_REFINEMENT_CONTEXT_FEATURE_COLUMNS)]:
        candidate=PilotCandidate(name+"_balanced_C1",1.0,"balanced")
        variants[name]=compare_pilot_candidates(
            frame,list(columns),candidates=(candidate,),group_column=group_column)
        (out/f"{suffix}_{name}.json").write_text(
            json.dumps(variants[name],indent=2,allow_nan=False),encoding="utf-8")
    old=pd.DataFrame(variants["legacy"]["predictions"])
    new=pd.DataFrame(variants["context"]["predictions"])
    assert old[["complex_id","algorithm","seed"]].equals(new[["complex_id","algorithm","seed"]])
    op=old["legacy_balanced_C1__after"].to_numpy(dtype=float)
    np_=new["context_balanced_C1__after"].to_numpy(dtype=float)
    common=np.isfinite(op)&np.isfinite(np_)
    from docking_confidence.validation.metrics import probability_metrics,probability_curves
    y=old.preserved.to_numpy(dtype=int)[common]
    groups=old[group_column].to_numpy(dtype=str)[common]
    reference=old.training_prevalence_probability.to_numpy(dtype=float)[common]
    summary={
        "scope":"exploratory_paired_development_comparison_not_independent_confirmation",
        "grouping":group_column,"total_pairs":len(old),"common_calibrated_pairs":int(common.sum()),
        "evaluated_groups":len(np.unique(groups)),"violations":int((y==0).sum()),
        "legacy":probability_metrics(y,op[common]),
        "context":probability_metrics(y,np_[common]),
        "legacy_violation":probability_metrics(1-y,1-op[common]),
        "context_violation":probability_metrics(1-y,1-np_[common]),
        "prevalence_reference":probability_metrics(y,reference),
        "paired_bootstrap":paired_uncertainty(op[common],np_[common],reference,y,groups),
        "uncertainty_note":"Whole-group percentile intervals conditional on the fitted fold models; no correction for prior exploratory candidate selection.",
        "automatic_authorization":False,"production_gates_changed":False,
        "policies":{},
    }
    for name,report,predictions in [("legacy",variants["legacy"],old),("context",variants["context"],new)]:
        lower=predictions[name+"_balanced_C1__lower"].to_numpy(dtype=float)[common]
        eligible=predictions[name+"_balanced_C1__in_domain"].to_numpy(dtype=bool)[common]
        accepted=eligible&(lower>=.9)
        count=int(accepted.sum());errors=int(((y==0)&accepted).sum())
        summary["policies"][name]={
            "threshold":.9,"hypothetical_accepted":count,"false_accepted":errors,
            "coverage_of_common_valid_pairs":count/len(y) if len(y) else None,
            "risk":errors/count if count else None,
            "false_acceptance_ids":predictions.loc[np.flatnonzero(common)[accepted&(y==0)],"complex_id"].tolist(),
        }
    if len(y):
        make_plot(y,op[common],np_[common],reference,out/f"{suffix}_diagnostics.png")
    (out/f"{suffix}_summary.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"analysis":suffix,**{key:summary[key] for key in
        ["total_pairs","common_calibrated_pairs","evaluated_groups","violations","legacy","context","paired_bootstrap","policies"]}}),flush=True)
    return summary


def make_plot(y,old,new,reference,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from docking_confidence.validation.metrics import probability_curves
    fig,axes=plt.subplots(1,3,figsize=(12,3.9),layout="constrained")
    for p,name,color in [(old,"Lógica anterior","#777777"),(new,"Contexto da execução","#297b98")]:
        curve=probability_curves(y,p)
        axes[0].plot([r["false_positive_rate"] for r in curve["roc"]],
                     [r["true_positive_rate"] for r in curve["roc"]],label=name,color=color)
        pr=probability_curves(1-y,1-p)["precision_recall"]
        axes[1].plot([r["recall"] for r in pr],[r["precision"] for r in pr],color=color,label=name)
        bins=[r for r in curve["reliability_bins"] if r["count"]]
        axes[2].plot([r["mean_probability"] for r in bins],
                     [r["observed_positive_fraction"] for r in bins],"o-",color=color,label=name)
    axes[0].plot([0,1],[0,1],"--",color="#bbbbbb")
    axes[1].axhline(float((y==0).mean()),ls="--",color="#bbbbbb",label="Prevalência de violações")
    axes[2].plot([0,1],[0,1],"--",color="#bbbbbb")
    for ax,title,xlabel,ylabel in zip(axes,["ROC: preservação","Precisão–sensibilidade: violações","Confiabilidade"],
        ["Taxa de falsos positivos","Sensibilidade","Probabilidade prevista"],
        ["Sensibilidade","Precisão","Frequência de preservação"]):
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel,xlim=(0,1),ylim=(0,1.02))
        ax.grid(alpha=.15);ax.legend(fontsize=7,loc="best")
    fig.suptitle("Piloto de desenvolvimento: previsões fora do ajuste por grupo molecular",fontsize=12)
    fig.savefig(path,dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--preparation-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--wait",action="store_true")
    args=parser.parse_args()
    source=Path(args.preparation_dir)
    while True:
        try:preparation=json.loads((source/"dataset_build_report.json").read_text(encoding="utf-8"))
        except (OSError,ValueError):
            if not args.wait:raise
            time.sleep(5);continue
        break
    if not preparation["valid"]:
        raise ValueError("assessment requires an official valid preparation")
    snapshot=Path(preparation["dataset_path"]).read_bytes()
    if hashlib.sha256(snapshot).hexdigest()!=preparation["dataset_sha256"]:
        raise ValueError("prepared dataset changed")
    frame=pd.read_csv(preparation["dataset_path"])
    output=Path(args.output_dir)
    if (output/"primary_summary.json").exists():
        raise FileExistsError("pilot assessment output already exists")
    assess(frame,output)
    grouping=source_clusters(frame)
    frame["pilot_source_cluster"]=frame.complex_fingerprint_sha256.map(grouping)
    (output/"source_cluster_audit.json").write_text(
        json.dumps({"complex_fingerprints":len(grouping),"source_clusters":len(set(grouping.values())),
                    "rule":"same source PDB ID or an exactly shared component sequence; no experimental outcomes used",
                    "mapping":grouping},indent=2),encoding="utf-8")
    try:
        assess(frame,output,group_column="pilot_source_cluster",suffix="source_blocked")
    except ValueError as exc:
        (output/"source_blocked_unavailable.json").write_text(
            json.dumps({"unavailable_reason":str(exc),"fallback_split_attempted":False}),encoding="utf-8")
        print("Source-blocked assessment unavailable:",str(exc),flush=True)


if __name__=="__main__":
    main()

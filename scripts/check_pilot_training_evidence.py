"""Try the official training gates once, without activating a pilot artifact."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docking_confidence.application.train import TrainingEvidenceError, train
from docking_confidence.io.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    source = Path(args.preparation_dir)
    while not (source / "dataset_build_report.json").exists():
        if not args.wait:
            raise ValueError("pilot preparation is not yet complete")
        time.sleep(5)
    report = json.loads((source / "dataset_build_report.json").read_text(encoding="utf-8"))
    if not report["valid"]:
        raise ValueError("invalid pilot preparation")
    payload = Path(report["dataset_path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != report["dataset_sha256"]:
        raise ValueError("prepared pilot dataset changed")
    config = load_config(report["config_path"])
    if config["authorization"]["allow_automatic_decision"] is not False:
        raise ValueError("this diagnostic requires automatic decisions disabled")
    output = Path(args.output_dir)
    evidence_path = output.parent / (output.name + "_evidence.json")
    if evidence_path.exists():
        raise FileExistsError("official pilot gate diagnostic already exists")
    try:
        result = train(report["dataset_path"], config, str(output))
        artifact = json.loads(Path(result["artifact"]).read_text(encoding="utf-8"))
        if artifact["decision_capable"]:
            raise RuntimeError("a pilot artifact must not authorize decisions")
        evidence = {"status": "diagnostic_artifact_only", "automatic_authorization": False,
                    "artifact": result["artifact"]}
    except TrainingEvidenceError as exc:
        evidence = {"status": "insufficient_evidence", "stage": exc.stage,
                    "message": str(exc), "evidence": exc.evidence,
                    "automatic_authorization": False}
    evidence["dataset_sha256"] = report["dataset_sha256"]
    evidence_path.write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in evidence.items() if key != "evidence"}), flush=True)


if __name__ == "__main__":
    main()

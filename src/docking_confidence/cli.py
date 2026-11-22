from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys

from docking_confidence.application.demo import make_demo
from docking_confidence.application.predict import predict
from docking_confidence.application.generalized_calibrator import create_calibrator
from docking_confidence.artifact.store import write_json
from docking_confidence.domain.execution import DockingTask
from docking_confidence.domain.mobile_partner import (
    DEFAULT_MOBILE_PARTNER_DOMAIN,
    MobilePartnerResidueDomain,
)
from docking_confidence.domain.structure import (
    CanonicalComplexFingerprint,
    RunBinding,
)
from docking_confidence.features.contact_graph import (
    ContactGraphConfig,
    describe_interface_contact_graph,
)
from docking_confidence.io.config import load_config
from docking_confidence.io.pdb import read_complex_pose


def _positive_int_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _nonnegative_int_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _positive_float_argument(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be numeric") from exc
    if not math.isfinite(parsed) or not (parsed > 0.0):
        raise argparse.ArgumentTypeError("must be finite and > 0")
    return parsed


def _build_dockingpp_archive_engine(
    *,
    archive_path: Path,
    config_entry: str,
    archive_sha256: str,
    config_sha256: str,
    timeout_seconds: float,
) -> object:
    from docking_confidence.adapters.dockingpp_archive_engine import (
        DockingPPArchiveEngine,
    )
    from docking_confidence.application.batch_launcher import (
        default_dockingpp_snapshot_dir,
    )

    return DockingPPArchiveEngine(
        archive_path,
        default_dockingpp_snapshot_dir(),
        config_entry=config_entry,
        expected_archive_sha256=archive_sha256,
        expected_config_sha256=config_sha256,
        timeout_seconds=timeout_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="docking-confidence")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ui", help="open the Streamlit interface")

    p_demo = sub.add_parser("make-demo", help="create a synthetic paired dataset")
    p_demo.add_argument("--out", required=True)
    p_demo.add_argument("--complexes", type=int, default=30)
    p_demo.add_argument("--runs", type=int, default=12)

    p_train = sub.add_parser("train", help="create a Generalized Venn-Abers (2025) quality-margin calibrator")
    p_train.add_argument("--dataset", required=True)
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--out-dir", required=True)
    p_train.add_argument("--calibration-fraction", type=float, default=0.3)
    p_train.add_argument("--random-seed", type=_nonnegative_int_argument, default=2026)

    p_predict = sub.add_parser("predict", help="apply a frozen artifact to REDUCED-only features")
    p_predict.add_argument("--artifact", required=True)
    p_predict.add_argument("--input", required=True)
    p_predict.add_argument("--out", required=True)

    p_quality = sub.add_parser("quality-check", help="check the isolated official DockQ runtime without scoring")
    p_quality.add_argument("--python", default=sys.executable)
    p_quality.add_argument("--dependency-path")

    p_standard = sub.add_parser("rescore-standard", help="create a separate officially rescored paired dataset; preserve historical outputs")
    p_standard.add_argument("--dataset", required=True)
    p_standard.add_argument("--batch", required=True, nargs="+")
    p_standard.add_argument("--config", required=True, help="original configuration of the source dataset")
    p_standard.add_argument("--references", required=True, nargs="+")
    p_standard.add_argument("--out-dir", required=True)
    p_standard.add_argument("--python", default=sys.executable)
    p_standard.add_argument("--dependency-path")
    p_standard.add_argument("--workers", type=_positive_int_argument, default=1)

    p_decision_prepare = sub.add_parser("decision-prepare", help="commit model, group roles, candidate cutoffs and empirical criteria before outcomes")
    p_decision_prepare.add_argument("--artifact", required=True)
    p_decision_prepare.add_argument("--config", required=True)
    p_decision_prepare.add_argument("--split", required=True, help="decision_group_split_v1 JSON with group identities, without labels")
    p_decision_prepare.add_argument("--out-dir", required=True)
    p_decision_prepare.add_argument("--minimum-preservation", type=float, default=.90)
    p_decision_prepare.add_argument("--minimum-coverage", type=float, default=.75)
    p_decision_prepare.add_argument("--minimum-accepted-complexes", type=_positive_int_argument, default=20)
    p_decision_prepare.add_argument("--score", choices=["prediction", "base_raw_score"], default="prediction")
    p_decision_prepare.add_argument("--thresholds", help="JSON file of predeclared numeric margin cutoffs; never derived from final labels")
    p_decision_prepare.add_argument("--exploratory", action="store_true")
    p_decision_prepare.add_argument("--require-runtime-improvement", action="store_true")
    p_decision_prepare.add_argument("--bootstrap-repetitions", type=_nonnegative_int_argument, default=2000)
    p_decision_prepare.add_argument("--random-seed", type=_nonnegative_int_argument, default=2026)
    for command, help_text, first_option in [
        ("decision-select", "select and freeze a rule on the committed policy-validation role", "protocol"),
        ("decision-evaluate", "evaluate a frozen rule once on its independent final-test role", "policy"),
    ]:
        stage = sub.add_parser(command, help=help_text)
        stage.add_argument("--" + first_option, required=True)
        stage.add_argument("--dataset", required=True)
        stage.add_argument("--out-dir", required=True)
        stage.add_argument("--expected-dataset-sha256")
    p_decision_predict = sub.add_parser("decision-predict", help="apply a frozen rule to REDUCED-only features with linked final assessment")
    p_decision_predict.add_argument("--policy", required=True)
    p_decision_predict.add_argument("--evaluation")
    p_decision_predict.add_argument("--input", required=True)
    p_decision_predict.add_argument("--out", required=True)

    p_batch_scan = sub.add_parser(
        "batch-scan",
        help="validate PPI4DOCK inputs and audit the configured mobile-partner range",
    )
    p_batch_scan.add_argument("--input-dir", required=True)
    p_batch_scan.add_argument("--fast", action="store_true", help="check layout only")
    p_batch_scan.add_argument(
        "--mobile-partner-min-residues",
        "--peptide-min-residues",
        dest="mobile_partner_min_residues",
        type=int,
        default=DEFAULT_MOBILE_PARTNER_DOMAIN.minimum_residues,
    )
    p_batch_scan.add_argument(
        "--mobile-partner-max-residues",
        "--peptide-max-residues",
        dest="mobile_partner_max_residues",
        type=int,
        default=DEFAULT_MOBILE_PARTNER_DOMAIN.maximum_residues,
    )
    p_batch_scan.add_argument("--out-json")
    p_batch_scan.add_argument("--out-csv")

    p_batch_real = sub.add_parser(
        "batch-run",
        help="run/resume the hash-pinned DockingPP engine on eligible PPI4DOCK targets",
    )
    p_batch_real.add_argument("--input-dir", required=True)
    p_batch_real.add_argument("--output-dir", required=True)
    p_batch_real.add_argument(
        "--engine-archive",
        help="optional legacy archive; by default uses the engine bundled with this project",
    )
    p_batch_real.add_argument(
        "--engine-config-entry",
        help="config inside the ZIP; defaults to the versioned V5 pin contract",
    )
    p_batch_real.add_argument(
        "--engine-archive-sha256",
        help="alternate archive pin; requires --engine-config-sha256",
    )
    p_batch_real.add_argument(
        "--engine-config-sha256",
        help="alternate config pin; requires --engine-archive-sha256",
    )
    p_batch_real.add_argument(
        "--algorithms",
        nargs="+",
        type=str.lower,
        choices=["ag", "pso", "hybrid"],
        default=["pso"],
    )
    p_batch_real.add_argument(
        "--protocol",
        type=str.lower,
        choices=["full", "reduced", "reduced_vs_full", "factorial_2x2", "operational_cascade"],
        default="reduced",
    )
    p_batch_real.add_argument(
        "--docking-task",
        type=str.lower,
        choices=[task.value for task in DockingTask],
        default=DockingTask.LOCAL_REFINEMENT.value,
        help=(
            "local_refinement: FULL and REDUCED refine the supplied relative pose; "
            "surface_discovery: reproduce legacy REDUCED surface search with local FULL control"
        ),
    )
    p_batch_real.add_argument(
        "--confidence-artifact",
        help=(
            "verified calibrator artifact.json; requires reduced_vs_full or operational_cascade; "
            "runs REDUCED first and skips FULL only when the predictor authorizes acceptance"
        ),
    )
    p_batch_real.add_argument(
        "--seed",
        nargs="+",
        type=_nonnegative_int_argument,
        default=[1],
        help="one or more deterministic seeds",
    )
    p_batch_real.add_argument("--generations", type=_positive_int_argument, default=100)
    p_batch_real.add_argument("--pop-size", type=_positive_int_argument, default=100)
    p_batch_real.add_argument("--top-pockets", type=_positive_int_argument, default=32)
    p_batch_real.add_argument("--evaluation-budget", type=_positive_int_argument, help="exact number of effective evaluations per factorial run (default: 7600)")
    p_batch_real.add_argument("--operational-config", help="separate operational YAML (default: configs/operational_cascade.yaml)")
    p_batch_real.add_argument("--collect-operational-pairs", action="store_true", help="article data collection: always execute both arms; do not consult a deployment model")
    p_batch_real.add_argument("--reduced-search-scale", type=_positive_float_argument)
    p_batch_real.add_argument("--max-translation", type=_positive_float_argument, help="expanded translational half-width in angstrom")
    p_batch_real.add_argument("--max-rotation-degrees", type=_positive_float_argument)
    p_batch_real.add_argument("--pose-source", help="origin of supplied pose and relationship to structural reference")
    p_batch_real.add_argument("--execution-order-seed", type=_nonnegative_int_argument)
    p_batch_real.add_argument("--analysis-role", choices=["exploratory", "confirmatory"])
    p_batch_real.add_argument("--prepare-only", action="store_true", help="freeze factorial protocol without executing docking")
    p_batch_real.add_argument("--experimental-protocol", help="previously frozen experimental_protocol.json, required for factorial execution")
    p_batch_real.add_argument(
        "--timeout-seconds",
        "--timeout",
        dest="timeout_seconds",
        type=_positive_float_argument,
        default=24 * 60 * 60,
    )
    p_batch_real.add_argument(
        "--resume-after-cancel",
        action="store_true",
        help="clear a previous cooperative cancellation marker before resuming",
    )

    p_factorial_report = sub.add_parser("analyze-factorial", help="analyze existing factorial receipts without executing docking")
    p_factorial_report.add_argument("--batch", required=True)
    p_factorial_report.add_argument("--output-dir", required=True)
    p_factorial_report.add_argument("--no-plots", action="store_true")

    p_batch_demo = sub.add_parser(
        "batch-run-demo",
        help="run/resume the explicitly non-scientific demo engine in batch",
    )
    p_batch_demo.add_argument("--input-dir", required=True)
    p_batch_demo.add_argument("--output-dir", required=True)
    p_batch_demo.add_argument(
        "--algorithms",
        nargs="+",
        choices=["ag", "pso", "hybrid"],
        default=["pso"],
    )
    p_batch_demo.add_argument(
        "--protocol",
        choices=["full", "reduced", "reduced_vs_full"],
        default="reduced_vs_full",
    )
    p_batch_demo.add_argument("--generations", type=int, default=100)
    p_batch_demo.add_argument("--population", type=int, default=100)
    p_batch_demo.add_argument("--top-pockets", type=int, default=32)
    p_batch_demo.add_argument("--seed-start", type=int, default=1)
    p_batch_demo.add_argument("--seed-count", type=int, default=1)
    p_batch_demo.add_argument(
        "--include-outside-domain",
        action="store_true",
        help="demo only: include targets outside the configured operational range",
    )
    p_batch_demo.add_argument(
        "--resume-after-cancel",
        action="store_true",
        help="clear a previous cooperative cancellation marker before resuming",
    )
    p_batch_demo.add_argument(
        "--acknowledge-non-scientific-demo",
        action="store_true",
        help="required acknowledgement that no molecular docking is performed",
    )

    p_batch_cancel = sub.add_parser(
        "batch-cancel", help="request cooperative cancellation between batch jobs"
    )
    p_batch_cancel.add_argument("--output-dir", required=True)

    p_batch_status = sub.add_parser(
        "batch-status", help="read the durable status of a batch"
    )
    p_batch_status.add_argument("--output-dir", required=True)

    p_graph = sub.add_parser(
        "graph-describe",
        help="calculate auditable contact-graph descriptors from one complex pose",
    )
    p_graph.add_argument("--reduced-complex-pdb", required=True)
    p_graph.add_argument("--complex-id", required=True)
    p_graph.add_argument("--run-id", required=True)
    p_graph.add_argument("--algorithm", required=True)
    p_graph.add_argument("--seed", required=True, type=int)
    p_graph.add_argument("--engine", required=True)
    p_graph.add_argument("--receptor-sequence", required=True)
    p_graph.add_argument("--ligand-sequence", required=True)
    p_graph.add_argument("--receptor-chains", required=True, help="comma-separated chain IDs")
    p_graph.add_argument("--ligand-chains", required=True, help="comma-separated chain IDs")
    p_graph.add_argument("--model-id", type=int, default=1)
    p_graph.add_argument("--pose-id")
    p_graph.add_argument("--pose-rank", type=int, default=1)
    p_graph.add_argument("--include-hetero", action="store_true")
    p_graph.add_argument("--cutoff", type=float, default=4.5)
    p_graph.add_argument("--clash-cutoff", type=float, default=2.0)
    p_graph.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.command == "ui":
        app_path = Path(__file__).parent / "ui" / "app.py"
        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)]))
    if args.command == "make-demo":
        make_demo(args.out, n_complexes=args.complexes, runs_per_complex=args.runs)
        print(args.out)
    elif args.command == "train":
        result = create_calibrator(args.dataset, load_config(args.config), args.out_dir,
            calibration_fraction=args.calibration_fraction, random_seed=args.random_seed)
        print(json.dumps(result["paths"], indent=2, ensure_ascii=False))
    elif args.command == "predict":
        result = predict(args.artifact, args.input, args.out)
        print(f"wrote {len(result)} predictions to {args.out}")
    elif args.command == "quality-check":
        from docking_confidence.evaluation.official_dockq import official_dockq_preflight
        result = official_dockq_preflight(python_executable=args.python, dependency_path=args.dependency_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result.get("available"):
            raise SystemExit(2)
    elif args.command == "rescore-standard":
        from docking_confidence.application.standard_dataset import build_standard_paired_dataset
        result = build_standard_paired_dataset(args.dataset, args.batch, args.out_dir,
            source_config_path=args.config, reference_roots=args.references,
            python_executable=args.python, dependency_path=args.dependency_path, max_workers=args.workers)
        print(json.dumps({key: result.get(key) for key in ["valid", "dataset_path", "config_path", "counts", "issues"]},
                         indent=2, ensure_ascii=False))
        if not result.get("valid"):
            raise SystemExit(2)
    elif args.command == "decision-prepare":
        from docking_confidence.application.decision_validation import prepare_decision_protocol
        split = json.loads(Path(args.split).read_text(encoding="utf-8-sig"))
        if split.get("schema_version") != "decision_group_split_v1":
            parser.error("split file must use decision_group_split_v1")
        thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8-sig")) if args.thresholds else None
        result = prepare_decision_protocol(args.artifact, split["validation_groups"], split["test_groups"], args.out_dir,
            config_path=args.config, minimum_preservation=args.minimum_preservation,
            minimum_coverage=args.minimum_coverage, minimum_accepted_complexes=args.minimum_accepted_complexes,
            score_column=args.score, candidate_thresholds=thresholds,
            prior_exposure_groups=split.get("prior_exposure_groups", ()),
            analysis_role="exploratory" if args.exploratory else "prospective",
            planned_runs=split.get("planned_runs"), algorithms=tuple(split.get("algorithms", ["ag","pso","hybrid"])),
            seeds=tuple(split.get("seeds", [1])), bootstrap_repetitions=args.bootstrap_repetitions,
            random_seed=args.random_seed,
            require_runtime_improvement=args.require_runtime_improvement)
        print(json.dumps({"protocol_path":result["protocol_path"], "criteria":result["protocol"]["criteria"],
                          "analysis_role":result["protocol"]["analysis_role"]}, indent=2,ensure_ascii=False))
    elif args.command == "decision-select":
        from docking_confidence.application.decision_validation import select_decision_policy
        result = select_decision_policy(args.protocol,args.dataset,args.out_dir,
                                       expected_dataset_sha256=args.expected_dataset_sha256)
        print(json.dumps({"policy_path":result["policy_path"], "status":result["policy"]["status"],
                          "threshold":result["policy"].get("threshold")}, indent=2,ensure_ascii=False))
    elif args.command == "decision-evaluate":
        from docking_confidence.application.decision_validation import evaluate_frozen_policy
        result = evaluate_frozen_policy(args.policy,args.dataset,args.out_dir,
                                       expected_dataset_sha256=args.expected_dataset_sha256)
        print(json.dumps({"report_path":result.get("report_path"),"status":result["report"]["status"],
                          "selected":result["report"]["selected"],"baseline":result["report"]["baseline"],
                          "decision_capable":result["report"]["decision_capable"]},indent=2,ensure_ascii=False))
    elif args.command == "decision-predict":
        from docking_confidence.application.decision_validation import predict_decisions
        result = predict_decisions(args.policy,args.input,args.out,evaluation_path=args.evaluation)
        print(f"wrote {len(result)} auditable decisions to {args.out}")
    elif args.command == "batch-scan":
        from docking_confidence.application.batch_reporting import (
            discovery_payload,
            write_discovery_reports,
        )
        from docking_confidence.io.batch_discovery import discover_ppi4dock

        report = discover_ppi4dock(
            args.input_dir,
            validate_contents=not bool(args.fast),
            mobile_partner_domain=MobilePartnerResidueDomain(
                minimum_residues=int(args.mobile_partner_min_residues),
                maximum_residues=int(args.mobile_partner_max_residues),
            ),
        )
        write_discovery_reports(
            report,
            json_path=args.out_json,
            csv_path=args.out_csv,
        )
        print(
            json.dumps(
                discovery_payload(report)["counts"],
                indent=2,
                ensure_ascii=False,
            )
        )
        if report.has_errors:
            raise SystemExit(2)
    elif args.command == "batch-run":
        from docking_confidence.application.batch import (
            clear_batch_cancel_request,
            run_batch,
        )
        from docking_confidence.application.batch_launcher import (
            discover_dockingpp_archive,
            load_dockingpp_archive_contract,
        )
        from docking_confidence.application.planning import build_execution_plan
        from docking_confidence.domain.execution import (
            Algorithm,
            BaseSearchConfig,
            ProtocolMode,
            FactorialExperimentConfig,
            FACTORIAL_EVALUATION_BUDGET,
        )

        if args.confidence_artifact is not None and args.protocol not in {"reduced_vs_full", "operational_cascade"}:
            parser.error("--confidence-artifact requires --protocol reduced_vs_full or operational_cascade")
        if args.operational_config is not None and args.protocol != "operational_cascade":
            parser.error("--operational-config requires --protocol operational_cascade")
        if args.collect_operational_pairs and (args.protocol != "operational_cascade" or args.confidence_artifact is not None):
            parser.error("--collect-operational-pairs requires operational_cascade without --confidence-artifact")
        if args.protocol != "factorial_2x2" and any((args.prepare_only, args.experimental_protocol, args.evaluation_budget is not None,
            args.pose_source is not None, args.reduced_search_scale is not None, args.max_translation is not None,
            args.max_rotation_degrees is not None, args.execution_order_seed is not None, args.analysis_role is not None)):
            parser.error("experimental controls require --protocol factorial_2x2")
        if args.protocol == "factorial_2x2" and not args.prepare_only and not args.experimental_protocol:
            parser.error("first freeze the protocol with --prepare-only, then provide --experimental-protocol")
        if (args.engine_archive_sha256 is None) != (args.engine_config_sha256 is None):
            parser.error(
                "--engine-archive-sha256 and --engine-config-sha256 must be provided together"
            )
        try:
            if args.engine_archive is None:
                if args.engine_archive_sha256 or args.engine_config_entry:
                    parser.error("archive options require --engine-archive")
                from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
                engine = DockingPPLocalEngine(timeout_seconds=float(args.timeout_seconds))
            else:
                contract = load_dockingpp_archive_contract()
                archive_path = discover_dockingpp_archive(args.engine_archive)
                config_entry = args.engine_config_entry or contract.config_entry
                archive_sha256 = args.engine_archive_sha256 or contract.archive_sha256
                config_sha256 = args.engine_config_sha256 or contract.config_sha256
                engine = _build_dockingpp_archive_engine(
                    archive_path=archive_path,
                    config_entry=config_entry,
                    archive_sha256=archive_sha256,
                    config_sha256=config_sha256,
                    timeout_seconds=float(args.timeout_seconds),
                )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            parser.error(str(exc))

        experiment = None
        if args.protocol == "factorial_2x2":
            experiment = FactorialExperimentConfig(
                evaluation_budget=args.evaluation_budget or FACTORIAL_EVALUATION_BUDGET,
                reduced_search_scale=args.reduced_search_scale if args.reduced_search_scale is not None else 0.5,
                max_translation=args.max_translation if args.max_translation is not None else 5.0,
                max_rotation_degrees=args.max_rotation_degrees if args.max_rotation_degrees is not None else 25.0,
                pose_source=args.pose_source or "",
                execution_order_seed=args.execution_order_seed if args.execution_order_seed is not None else 20260925,
            )
        base = BaseSearchConfig(
            generations=int(args.generations),
            population=int(args.pop_size),
            top_pockets=int(args.top_pockets),
            docking_task=DockingTask(args.docking_task),
            experiment=experiment,
        )
        if args.protocol == "operational_cascade":
            from docking_confidence.application.operational import load_operational_config
            if args.docking_task != "local_refinement":
                parser.error("operational_cascade requires local_refinement")
            base = load_operational_config(args.operational_config)
            if (args.generations != base.generations or args.pop_size != base.population
                    or args.top_pockets not in {32, 1}):
                parser.error("operational population/generations must match --operational-config; one pocket is required")
        plan = build_execution_plan(
            algorithms=[Algorithm(value) for value in args.algorithms],
            protocol_mode=ProtocolMode(args.protocol),
            base_config=base,
            seeds=args.seed,
        )
        if args.prepare_only:
            from docking_confidence.application.experimental_protocol import prepare_experimental_protocol
            prepared = prepare_experimental_protocol(args.input_dir, args.output_dir, jobs=plan, engine=engine, analysis_role=args.analysis_role or "exploratory")
            print(json.dumps(prepared, indent=2, ensure_ascii=False))
            return
        if args.resume_after_cancel:
            clear_batch_cancel_request(args.output_dir)
        summary = run_batch(
            args.input_dir,
            args.output_dir,
            jobs=plan,
            engine=engine,
            eligible_only=True,
            confidence_artifact=args.confidence_artifact,
            experimental_protocol=args.experimental_protocol,
            collect_operational_pairs=args.collect_operational_pairs,
        )
        payload = asdict(summary)
        payload["manifest_path"] = str(summary.manifest_path)
        payload["status_path"] = str(summary.status_path)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif args.command == "analyze-factorial":
        from docking_confidence.application.factorial_reporting import analyze_factorial_batch, write_factorial_report
        report = analyze_factorial_batch(args.batch)
        paths = write_factorial_report(report, args.output_dir, plots=not args.no_plots)
        print(json.dumps(paths, indent=2, ensure_ascii=False, default=str))
    elif args.command == "batch-run-demo":
        from docking_confidence.adapters.demo_engine import DemoDockingEngine
        from docking_confidence.application.batch import (
            clear_batch_cancel_request,
            run_batch,
        )
        from docking_confidence.application.planning import build_execution_plan
        from docking_confidence.domain.execution import (
            Algorithm,
            BaseSearchConfig,
            ProtocolMode,
        )

        if not args.acknowledge_non_scientific_demo:
            parser.error(
                "batch-run-demo requires --acknowledge-non-scientific-demo"
            )
        if args.seed_count < 1:
            parser.error("--seed-count must be >= 1")
        base = BaseSearchConfig(
            generations=int(args.generations),
            population=int(args.population),
            top_pockets=int(args.top_pockets),
        )
        plan = build_execution_plan(
            algorithms=[Algorithm(value) for value in args.algorithms],
            protocol_mode=ProtocolMode(args.protocol),
            base_config=base,
            seeds=range(args.seed_start, args.seed_start + args.seed_count),
        )
        if args.resume_after_cancel:
            clear_batch_cancel_request(args.output_dir)
        summary = run_batch(
            args.input_dir,
            args.output_dir,
            jobs=plan,
            engine=DemoDockingEngine(),
            eligible_only=not bool(args.include_outside_domain),
        )
        payload = asdict(summary)
        payload["manifest_path"] = str(summary.manifest_path)
        payload["status_path"] = str(summary.status_path)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif args.command == "batch-cancel":
        from docking_confidence.application.batch import request_batch_cancel

        print(request_batch_cancel(args.output_dir))
    elif args.command == "batch-status":
        from docking_confidence.application.batch import load_batch_status

        status = load_batch_status(args.output_dir)
        if status is None:
            print("no batch status found", file=sys.stderr)
            raise SystemExit(1)
        print(json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True))
    elif args.command == "graph-describe":
        receptor_chains = [item.strip() for item in args.receptor_chains.split(",")]
        ligand_chains = [item.strip() for item in args.ligand_chains.split(",")]
        graph_config = ContactGraphConfig(
            model_id=int(args.model_id),
            cutoff_angstrom=float(args.cutoff),
            clash_cutoff_angstrom=float(args.clash_cutoff),
            include_hetero=bool(args.include_hetero),
        )
        run_binding = RunBinding(
            complex_id=args.complex_id,
            run_id=args.run_id,
            algorithm=args.algorithm,
            seed=int(args.seed),
            protocol="reduced",
            engine=args.engine,
        )
        complex_fingerprint = CanonicalComplexFingerprint.from_sequences(
            args.receptor_sequence,
            args.ligand_sequence,
        )
        pose = read_complex_pose(
            args.reduced_complex_pdb,
            receptor_chains=receptor_chains,
            ligand_chains=ligand_chains,
            run_binding=run_binding,
            complex_fingerprint=complex_fingerprint,
            include_hetero=graph_config.include_hetero,
            heavy_atoms_only=graph_config.heavy_atoms_only,
            exclude_waters=graph_config.exclude_waters,
            model_id=graph_config.model_id,
            pose_id=args.pose_id,
            pose_rank=int(args.pose_rank),
        )
        descriptors = describe_interface_contact_graph(pose, graph_config)
        write_json(args.out, descriptors)
        print(args.out)

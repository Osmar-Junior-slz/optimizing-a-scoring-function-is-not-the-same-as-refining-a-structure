"""Reanalyse the verified original risk role; no docking, model refitting or test selection."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docking_confidence.application.article_study import freeze_comparators, study_software
from docking_confidence.application.article_reporting import write_risk_outputs
from docking_confidence.application.batch import _atomic_write_json, _file_sha256
from docking_confidence.application.calibrated_batch import CalibratedBatchGate, configured_calibrator_path
from docking_confidence.risk.scientific import structural_outcomes, zero_error_sample_requirements
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new audit output directory; historical results are immutable")
    gate = CalibratedBatchGate(configured_calibrator_path())
    before = _file_sha256(gate.artifact_path)
    frozen = freeze_comparators(gate)
    profile = frozen["validation_risk_profile"]
    frame = pd.read_csv(gate.artifact_path.parent / "risk_calibration_predictions.csv", float_precision="round_trip")
    outcome = structural_outcomes(frame)
    summary = {"scope": "exploratory reanalysis of previously observed development risk role; no new docking or independent validation",
        "artifact": str(gate.artifact_path), "artifact_sha256": before,
        "risk_rows": len(frame), "independent_risk_complexes": int(frame.complex_fingerprint_sha256.nunique()),
        "legacy_probability_target": gate.artifact["preservation_rule"],
        "legacy_negative_rows": int((frame.preserved == 0).sum()),
        "structural_class_adverse_rows": int((~outcome.structural_quality_preserved).sum()),
        "old_target_positive_but_structural_class_adverse_rows": int(((frame.preserved == 1) & ~outcome.structural_quality_preserved).sum()),
        "maximum_accepted_error_rate": None, "selected_threshold": None, "automatic_decisions_authorized": False,
        "family_size": profile["family_size"], "error_probability_per_bound": profile["error_probability_per_bound"],
        "lowest_grid_point_diagnostics_not_a_selected_rule": {method: curve[0] for method, curve in profile["curves"].items()},
        "sample_planning_illustrative_not_approved": zero_error_sample_requirements(frozen["risk_protocol"]),
        "software": study_software()}
    out.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out / "risk_profile.json", profile)
    _atomic_write_json(out / "audit_summary.json", summary)
    write_risk_outputs(profile, out, "validacao")
    rows = ["# Revisão científica do risco — dados salvos", "",
        "Esta é uma reanálise exploratória da partição de risco original. Nenhum docking foi executado, nenhum peso do Venn–Abers foi alterado e nenhum limiar foi aprovado.", "",
        f"Foram analisadas {len(frame)} execuções em {summary['independent_risk_complexes']} complexos distintos. "
        f"O alvo histórico tem {summary['legacy_negative_rows']} resultados negativos; "
        f"o desfecho estrutural por classes tem {summary['structural_class_adverse_rows']} casos adversos. "
        "Esses alvos são distintos; as probabilidades não foram renomeadas.", "",
        f"Família declarada: {profile['family_size']} elementos. Erro por limite: {profile['error_probability_per_bound']:.8g}.", "",
        "| Método | Limiar mínimo da grade (ilustrativo) | Complexos aceitos hipoteticamente | Adversos | Limite superior simultâneo |",
        "|---|---:|---:|---:|---:|"]
    for method, curve in profile["curves"].items():
        row = curve[0]
        upper = row["simultaneous_complex_risk_upper_bound"]
        rows.append(f"| {method} | {row['threshold']:.2f} | {row['accepted_complexes']} | {row['worst_case_adverse_complexes']} | "
                    + (f"{upper:.6f} |" if upper is not None else "indisponível |"))
    rows += ["", "O ponto mínimo é exibido para descrever os dados, não como decisão recomendada. Zero erros observados não significa risco zero.", "",
        "![Curva de risco e cobertura](risco_cobertura_validacao.png)", "",
        "A taxa máxima de erro continua indefinida. As necessidades de amostra no JSON são exemplos de planejamento, não tolerâncias aprovadas. "
        "Os limites dependem das hipóteses de independência e protocolo fixo; esta revisão de dados já observados não constitui validação prospectiva.", "",
        "Método: [Learn then Test, seções 2.1/2.3/3.2](https://arxiv.org/abs/2110.01052v5). "
        "Classes: [DockQ](https://doi.org/10.1371/journal.pone.0161879). "
        "Escores: [Venn–Abers indutivo](https://arxiv.org/abs/1511.00213)."]
    (out / "RELATORIO_CIENTIFICO.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert _file_sha256(gate.artifact_path) == before
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

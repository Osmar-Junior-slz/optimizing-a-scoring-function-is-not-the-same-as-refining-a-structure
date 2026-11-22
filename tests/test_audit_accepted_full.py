"""Only invariants needed for the counterfactual FULL procedure."""
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    import audit_accepted_full as audit
finally:
    sys.path.remove(str(SCRIPTS))


def case(score=.4, algorithm="AG", seed=7, complex_id="1abc_AB"):
    return dict(complex_id=complex_id, algorithm=algorithm, seed=seed,
                action="ACCEPT_REDUCED", dockq_reduced=score, dockq_full=None,
                dockq_full_minus_reduced=None, Y=None, reduced_state="SUCCEEDED")


def original_job(algorithm="ag", seed=7):
    return {"algorithm": algorithm, "seed": seed, "protocol": "full",
            "job_id": f"{algorithm}-full-seed{seed}",
            "profile": {"mode": "full", "docking_task": "local_refinement",
                        "generations": 100, "population": 100, "top_pockets": 32,
                        "evaluation_budget": 10100, "generation_factor": 1.0, "pocket_factor": 1.0}}


@pytest.mark.parametrize("algorithm,seed", [("AG", 7), ("PSO", 0), ("Hybrid", 42)])
def test_exact_original_full_job_only(algorithm, seed):
    original = original_job(algorithm.lower(), seed)
    job = audit.full_job(case(algorithm=algorithm, seed=seed), [original])
    assert audit._job_snapshot(job) == original
    assert job.protocol.value == "full"


def test_only_eligible_acceptances_selected():
    rows = [case(.23), case(.229, seed=8), case(seed=9), case(seed=10)]
    rows[2]["action"] = "RUN_FULL"
    rows[3]["dockq_full"] = .4
    assert audit.select_cases(rows) == [rows[0]]


@dataclass
class Report:
    input_root: Path
    targets: tuple
    issues: tuple = ()


def test_runner_gets_one_original_complex_full_only_and_preserves_files(tmp_path, monkeypatch):
    inputs, batch, output = (tmp_path / n for n in ("inputs", "original_batch", "audit"))
    inputs.mkdir()
    batch.mkdir()
    source = inputs / "input.pdb"
    source.write_bytes(b"original coordinates")
    decision = batch / "decision.json"
    decision.write_bytes(b"original decision")
    summary = tmp_path / "scientific_summary.json"
    summary.write_bytes(b"original summary")
    before = {p: p.read_bytes() for p in (source, decision, summary)}
    target = SimpleNamespace(complex_id="1abc_AB")
    report = Report(inputs, (target, SimpleNamespace(complex_id="2abc_AB")))
    snapshot = {"complex_id": "1abc_AB", "input_sha256": "original"}
    monkeypatch.setattr(audit, "_target_snapshot", lambda t, root: snapshot)
    output = audit.separate_output(output, batch, inputs, summary)
    calls = []

    def runner(source_root, destination, **kwargs):
        calls.append(kwargs)
        assert source_root == inputs
        assert kwargs["report"].targets == (target,)
        assert kwargs["confidence_artifact"] is None
        assert len(kwargs["jobs"]) == 1
        assert audit._job_snapshot(kwargs["jobs"][0]) == original_job()
        assert destination.is_relative_to(output)
        destination.mkdir(parents=True)
        (destination / "result.json").write_bytes(b"audit FULL only")

    audit.execute_case(case(), audit.full_job(case(), [original_job()]), report,
                       snapshot, output, object(), runner=runner)
    assert len(calls) == 1
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize("location", ["original", "original/nested", ".", "inputs", "inputs/nested"])
def test_refuses_original_output_overlap(tmp_path, location):
    with pytest.raises(ValueError):
        audit.separate_output(tmp_path / location, tmp_path / "original", tmp_path / "inputs",
                              tmp_path / "scientific_summary.json")


def test_changed_original_target_cannot_execute(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "_target_snapshot", lambda *args: {"hash": "changed"})
    report = Report(tmp_path, (SimpleNamespace(complex_id="1abc_AB"),))
    with pytest.raises(ValueError, match="Complex/input"):
        audit.execute_case(case(), audit.full_job(case(), [original_job()]), report,
                           {"hash": "original"}, tmp_path / "out", object(),
                           runner=lambda *args, **kwargs: pytest.fail("Must not execute"))


def test_y_ppv_risk_and_original_decisions_unchanged():
    rows = [case(.23), case(.5, seed=8), case(.1, seed=9)]
    original = deepcopy(rows)
    merged = audit.merge_audit(rows, {audit.case_key(rows[0]): (.28, "audit/a"),
                                     audit.case_key(rows[1]): (.55000001, "audit/b")})
    assert rows == original
    assert [c["Y"] for c in merged] == [1, 0, 0]
    assert merged[2]["dockq_full"] is None
    assert merged[2]["dockq_full_minus_reduced"] is None
    assert [(c["action"], c["dockq_reduced"], audit.case_key(c)) for c in merged] == [
        (c["action"], c["dockq_reduced"], audit.case_key(c)) for c in original]
    metrics = audit.acceptance_metrics(merged)
    assert metrics["accepted"] == 3
    assert metrics["Y_1"] == 1 and metrics["Y_0"] == 2
    assert metrics["PPV"] == 1 / 3
    assert metrics["false_accept_risk"] == 2 / 3


def test_missing_full_does_not_become_false_accept():
    with pytest.raises(ValueError):
        audit.merge_audit([case()], {})
    with pytest.raises(ValueError):
        audit.merge_audit([case()], {audit.case_key(case()): (None, "audit")})

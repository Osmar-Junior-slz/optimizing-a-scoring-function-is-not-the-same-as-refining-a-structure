"""Check that lineage claims use reproductions with the appropriate denominator."""
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from revision_seed_replay import export_results


def test_parallel_replays_preserve_historical_order(monkeypatch, tmp_path):
    from threading import Event
    import revision_seed_replay as replay
    second_finished = Event()
    rows = [{"batch_shard": "s", "complex_id": str(i), "job_id": str(i)} for i in range(4)]
    inputs = {("s", row["complex_id"]): [] for row in rows}

    def job(study, row, paths, out, identity):
        if row["job_id"] == "0":
            assert second_finished.wait(5)
        if row["job_id"] == "1":
            second_finished.set()
        return row["job_id"]

    monkeypatch.setattr(replay, "replay_run", job)
    assert replay.replay_selection(rows, inputs, tmp_path, tmp_path, "engine", 2) == ["0", "1", "2", "3"]


def test_serial_receipts_require_same_inputs_and_known_replay_implementation(monkeypatch):
    import revision_seed_replay as replay
    current = {"script": "new", "config": "frozen"}
    old = {**current, "script": replay._SERIAL_SCRIPT_HASH}
    assert replay.binding_matches(old, current)
    assert not replay.binding_matches({**old, "config": "different"}, current)
    assert not replay.binding_matches({**old, "script": "unknown"}, current)
    monkeypatch.setattr(replay, "replay_core_hash", lambda: "changed")
    assert not replay.binding_matches(old, current)


def test_rejected_and_divergent_replays_have_distinct_denominators(tmp_path):
    def result(target, *, matches, valid, last_generation):
        return {
            "complex_id": target, "algorithm": "hybrid", "condition": "reduced_seeded",
            "job_id": target, "matches_original": matches, "valid_final_pose": valid,
            "final_seed_lineage": True if valid else None, "final_unchanged_seed": False if valid else None,
            "historical_dockq": None, "differences": [] if matches else ["vector"],
            "trace": {"generations": [
                {"generation": g, "n_eval_cumulative": 100 * (1 + g), "population_size": 100,
                 "unchanged_seed_count": int(g == 0), "seed_lineage_count": 1}
                for g in range(last_generation + 1)]},
        }

    export_results(tmp_path, [
        result("ok", matches=True, valid=True, last_generation=1),
        result("rejected", matches=True, valid=False, last_generation=2),
        result("different", matches=False, valid=True, last_generation=3),
    ])

    def read(name):
        with (tmp_path / name).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    group, = read("resumo.csv")
    assert (group["replayed"], group["matched"], group["matched_valid_final_poses"]) == ("3", "2", "1")
    assert (group["final_from_seed"], group["final_unchanged"]) == ("1", "0")
    survival = read("sobrevivencia.csv")
    assert [r["verified_runs_reaching_generation"] for r in survival] == ["2", "2", "1"]
    assert [r["with_unchanged_seed"] for r in survival] == ["2", "0", "0"]
    assert len(read("por_execucao.csv")) == 3

"""A temporary Windows lock must not corrupt state or rerun docking jobs."""
import errno
import json
from pathlib import Path

import pytest

from docking_confidence.application import batch


def _locked(winerror):
    if winerror is None:
        return PermissionError(errno.EACCES, "temporarily locked")
    error = OSError(errno.EBUSY, "Windows file lock")
    error.winerror = winerror
    return error


@pytest.mark.parametrize("winerror", [None, 5, 32, 33])
def test_transient_lock_preserves_previous_json_until_atomic_publication(tmp_path, monkeypatch, winerror):
    path = tmp_path / "batch_status.json"
    original = b'{"state":"RUNNING"}'
    path.write_bytes(original)
    payload = {"state": "COMPLETED", "details": "execução concluída"}
    replace = batch.os.replace
    attempted_sources = []
    waits = []

    def temporarily_locked(source, destination):
        assert Path(destination) == path
        assert path.read_bytes() == original
        assert Path(source).read_bytes() == batch._canonical_json_bytes(payload)
        attempted_sources.append(Path(source))
        if len(attempted_sources) < 3:
            raise _locked(winerror)
        replace(source, destination)

    monkeypatch.setattr(batch.os, "replace", temporarily_locked)
    monkeypatch.setattr(batch, "sleep", waits.append)
    batch._atomic_write_json(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert len(attempted_sources) == 3 and len(set(attempted_sources)) == 1
    assert len(waits) == 2 and sum(waits) < 2
    assert list(tmp_path.iterdir()) == [path]


def test_persistent_lock_has_bounded_retries_and_keeps_previous_state(tmp_path, monkeypatch):
    path = tmp_path / "batch_status.json"
    original = b'{"state":"RUNNING"}'
    path.write_bytes(original)
    error = _locked(5)
    attempts = []
    waits = []

    def still_locked(source, destination):
        attempts.append(source)
        raise error

    monkeypatch.setattr(batch.os, "replace", still_locked)
    monkeypatch.setattr(batch, "sleep", waits.append)
    with pytest.raises(OSError) as caught:
        batch._atomic_write_json(path, {"state": "COMPLETED"})
    assert caught.value is error
    assert 1 < len(attempts) <= 6 and len(waits) == len(attempts) - 1
    assert sum(waits) <= 2
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_unrelated_io_error_is_not_retried_or_hidden(tmp_path, monkeypatch):
    path = tmp_path / "batch_status.json"
    path.write_text('{"state":"RUNNING"}')
    error = OSError(errno.ENOSPC, "no space left")
    attempts = []
    waits = []

    def disk_full(source, destination):
        attempts.append(source)
        raise error

    monkeypatch.setattr(batch.os, "replace", disk_full)
    monkeypatch.setattr(batch, "sleep", waits.append)
    with pytest.raises(OSError) as caught:
        batch._atomic_write_json(path, {"state": "COMPLETED"})
    assert caught.value is error
    assert len(attempts) == 1 and waits == []
    assert json.loads(path.read_text()) == {"state": "RUNNING"}
    assert list(tmp_path.iterdir()) == [path]


def test_failed_cleanup_does_not_mask_original_publication_error(tmp_path, monkeypatch):
    path = tmp_path / "batch_status.json"
    path.write_text('{"state":"RUNNING"}')
    original_error = OSError(errno.ENOSPC, "no space left")
    unlink = Path.unlink

    def publication_failed(*args):
        raise original_error

    def cleanup_failed(self, *args, **kwargs):
        if self.parent == tmp_path and self.suffix == ".tmp":
            raise PermissionError("temporary file still locked")
        return unlink(self, *args, **kwargs)

    monkeypatch.setattr(batch.os, "replace", publication_failed)
    monkeypatch.setattr(Path, "unlink", cleanup_failed)
    with pytest.raises(OSError) as caught:
        batch._atomic_write_json(path, {"state": "COMPLETED"})
    assert caught.value is original_error
    assert json.loads(path.read_text()) == {"state": "RUNNING"}


def test_batch_completes_after_state_lock_without_repeating_engine_calls(tmp_path, monkeypatch):
    from test_batch_runner import _jobs, _make_report, _RecordingEngine

    source = tmp_path / "input"
    source.mkdir()
    report = _make_report(source, 1)
    engine = _RecordingEngine()
    jobs = _jobs(2)
    output = tmp_path / "output"
    replace = batch.os.replace
    waits = []
    blocked = 0

    def temporary_state_lock(source_path, destination):
        nonlocal blocked
        if Path(destination).name == "batch_status.json" and blocked < 2:
            blocked += 1
            raise _locked(32)
        return replace(source_path, destination)

    monkeypatch.setattr(batch.os, "replace", temporary_state_lock)
    monkeypatch.setattr(batch, "sleep", waits.append)
    summary = batch.run_batch(source, output, jobs=jobs, engine=engine, report=report)
    assert summary.state == "COMPLETED" and summary.failed == 0
    assert summary.succeeded == len(jobs) == len(engine.calls)
    assert blocked == 2 and len(waits) == 2
    assert json.loads((output / "batch_status.json").read_text())["state"] == "COMPLETED"

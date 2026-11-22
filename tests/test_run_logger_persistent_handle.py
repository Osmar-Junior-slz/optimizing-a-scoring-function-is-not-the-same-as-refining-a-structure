"""Live metrics keep their visibility and bytes with one persistent handle."""
import builtins
import gc
import json

import pytest

from dockingpp.pipeline import logging


def test_live_rows_visible_immediately_and_final_bytes_identical(tmp_path, monkeypatch):
    opened = []
    def track_open(*args, **kwargs):
        handle = builtins.open(*args, **kwargs)
        opened.append((args[1], handle))
        return handle
    monkeypatch.setattr(logging, "open", track_open, raising=False)
    logger = logging.RunLogger(out_dir=str(tmp_path), live_write=True)
    for step in range(12):
        logger.log_metric("n_eval", step + .5, step, {"generation": step, "note": "interação"})
        lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in lines] == logger.records
    assert len(opened) == 1
    live_bytes = (tmp_path / "metrics.jsonl").read_bytes()
    logger.flush(str(tmp_path))
    assert opened[0][1].closed
    assert [mode for mode, handle in opened] == ["a", "w"]
    assert (tmp_path / "metrics.jsonl").read_bytes() == live_bytes
    assert all(handle.closed for mode, handle in opened)
    logger.close()


def test_live_and_memory_only_final_metrics_and_timeseries_match(tmp_path):
    live_dir, memory_dir = tmp_path / "live", tmp_path / "memory"
    live_dir.mkdir()
    memory_dir.mkdir()
    live = logging.RunLogger(out_dir=str(live_dir), live_write=True)
    memory = logging.RunLogger(out_dir=str(memory_dir), live_write=False)
    for logger in (live, memory):
        logger.log_global_metrics(5, 2)
        for step in range(3):
            logger.log_metric("n_eval", 10., step, {"generation": step})
            logger.log_metric("best_score", -step / 3., step)
            logger.log_metric("expensive_ran", 1., step)
    assert not (memory_dir / "metrics.jsonl").exists()
    for logger, directory in ((live, live_dir), (memory, memory_dir)):
        logger.flush(str(directory))
        logger.flush_timeseries(str(directory), mode="reduced")
    for name in ("metrics.jsonl", "metrics.timeseries.jsonl"):
        assert (live_dir / name).read_bytes() == (memory_dir / name).read_bytes()


def test_logging_after_final_flush_appends_without_duplicates(tmp_path):
    logger = logging.RunLogger(out_dir=str(tmp_path), live_write=True)
    logger.log_metric("first", 1., 0)
    logger.flush(str(tmp_path))
    logger.log_metric("second", 2., 1)
    assert [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()] == logger.records
    logger.close()


def test_changing_directory_closes_old_live_file(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    logger = logging.RunLogger(out_dir=str(first), live_write=True)
    logger.log_metric("first", 1., 0)
    old_handle = logger._live_handle
    logger.out_dir = str(second)
    logger.log_metric("second", 2., 1)
    assert old_handle.closed
    assert len((first / "metrics.jsonl").read_text().splitlines()) == 1
    assert len((second / "metrics.jsonl").read_text().splitlines()) == 1
    logger.close()


@pytest.mark.parametrize("operation", ["write", "flush"])
def test_io_failure_closes_live_handle_and_preserves_exception(tmp_path, monkeypatch, operation):
    class FailedHandle:
        closed = False
        def write(self, value):
            if operation == "write":
                raise OSError("original write failure")
        def flush(self):
            if operation == "flush":
                raise OSError("original flush failure")
        def close(self):
            self.closed = True
    handle = FailedHandle()
    monkeypatch.setattr(logging, "open", lambda *args, **kwargs: handle, raising=False)
    logger = logging.RunLogger(out_dir=str(tmp_path), live_write=True)
    with pytest.raises(OSError, match=f"original {operation} failure"):
        logger.log_metric("failed", 1., 0)
    assert handle.closed
    assert logger._live_handle is None


def test_serialization_failure_and_destructor_release_live_file(tmp_path):
    logger = logging.RunLogger(out_dir=str(tmp_path), live_write=True)
    logger.log_metric("first", 1., 0)
    first_handle = logger._live_handle
    with pytest.raises(TypeError):
        logger.log_metric("invalid_extra", 1., 0, {"invalid": object()})
    assert first_handle.closed
    logger.log_metric("last", 1., 1)
    last_handle = logger._live_handle
    del logger
    gc.collect()
    assert last_handle.closed

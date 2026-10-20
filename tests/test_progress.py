import logging
import sys
import time
from datetime import datetime, timedelta, timezone

from mini2p_preprocessing import batch as batch_module
from mini2p_preprocessing.progress import ProgressLog, format_duration, progress_line
from tests.test_batch import _batch_with_two_sources

NOW = datetime(2026, 10, 20, 23, 40, tzinfo=timezone(timedelta(hours=1)))


def test_durations_read_naturally():
    assert format_duration(42.4) == "42s"
    assert format_duration(12 * 60 + 4) == "12m04s"
    assert format_duration(3 * 3600 + 5 * 60 + 59) == "3h05m"


def test_estimate_uses_the_mean_time_per_run_and_gives_a_clock_time():
    line = progress_line(done=2, total=6, durations_s=[600.0, 1200.0], now=NOW)
    assert line.startswith("[2/6]")
    assert "last 20m00s" in line
    assert "about 1h00m left" in line  # 4 runs x 15 min
    assert "done around 00:40 Wed" in line


def test_last_run_and_no_history():
    assert "all done" in progress_line(3, 3, [60.0, 60.0, 60.0], NOW)
    assert progress_line(0, 3, [], NOW) == "[0/3]"


def test_tracker_logs_each_step_or_at_most_every_interval():
    ticks = iter([0.0, 10.0, 20.0, 30.0, 95.0, 100.0])
    lines = []
    tracker = ProgressLog(
        "Phase correction",
        5,
        lines.append,
        min_interval_s=60,
        clock=lambda: next(ticks),
        wall=lambda: NOW,
    )
    for _ in range(5):
        tracker.step_done()
    # the first step, then once a minute has passed, then the last step
    assert [line.split(" ", 3)[2] for line in lines] == ["[1/5]", "[4/5]", "[5/5]"]
    assert lines[0].startswith("Phase correction: [1/5] last 10s")
    assert "all done" in lines[-1]


def test_batch_reports_time_left_after_each_run(tmp_path, monkeypatch, caplog):
    batch, config = _batch_with_two_sources(tmp_path)
    monkeypatch.setattr(batch_module, "worker_command", lambda *a: [sys.executable, "-c", "pass"])
    with caplog.at_level(logging.INFO, logger="mini2p_preprocessing.batch"):
        batch_module.run_batch(batch, config)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Batch: ")]
    assert "left, done around" in lines[0] and "all done" in lines[1]


def test_finish_time_follows_the_clock_change(monkeypatch):
    monkeypatch.setenv("TZ", "Europe/London")
    time.tzset()
    try:
        before_change = datetime(2026, 10, 24, 23, 0, tzinfo=timezone(timedelta(hours=1)))
        line = progress_line(1, 4, [3600.0], before_change)  # 3 h left, across 02:00 BST
        assert "done around 01:00 Sun" in line
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


def test_failed_runs_count_but_do_not_shorten_the_estimate():
    ticks = iter([0.0, 600.0, 605.0])
    lines = []
    tracker = ProgressLog("Batch", 4, lines.append, clock=lambda: next(ticks), wall=lambda: NOW)
    tracker.step_done()
    tracker.step_done(succeeded=False)  # failed after 5 s
    assert lines[-1].startswith("Batch: [2/4]") and "about 20m00s left" in lines[-1]


def test_a_batch_that_stops_on_error_gives_no_estimate(tmp_path, monkeypatch, caplog):
    batch, config = _batch_with_two_sources(tmp_path)
    fail = [sys.executable, "-c", "raise SystemExit(3)"]
    monkeypatch.setattr(batch_module, "worker_command", lambda *a: fail)
    with caplog.at_level(logging.INFO, logger="mini2p_preprocessing.batch"):
        try:
            batch_module.run_batch(batch, config)
        except Exception:
            pass
    assert not [r for r in caplog.records if r.getMessage().startswith("Batch: ")]

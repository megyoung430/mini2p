"""The batch limited to the TIFFs that align to behavior, and run as a SLURM array."""

import csv
import json
from pathlib import Path

import pytest

from mini2p_preprocessing import batch as batch_module
from mini2p_preprocessing.batch import (
    AlignedRun,
    BatchSettings,
    alignment_decision,
    load_aligned_runs,
    plan_batch,
    prepare_array,
    run_array_task,
)
from mini2p_preprocessing.cli import batch_main


def make_raw_tree(root: Path, sessions: dict[int, list[str]]) -> Path:
    for number, files in sessions.items():
        funcimg = root / f"ses-{number:03d}_date-x" / "funcimg"
        funcimg.mkdir(parents=True)
        for name in files:
            (funcimg / name).write_bytes(b"tif")
    return root


def write_listing(path: Path, rows: list[tuple]) -> Path:
    """rows: (session, run, tiff, aligned, reason), as list_aligned_runs.py writes them."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["session", "run", "tiff", "h5", "tsv", "aligned", "reason"])
        for session, run, tiff, aligned, reason in rows:
            writer.writerow([session, run, str(tiff), "", "", aligned, reason])
    return path


@pytest.fixture
def two_sessions(tmp_path):
    """Sessions 5 and 7 (6 missing), two TIFFs each; the listing aligns three of four."""
    raw = make_raw_tree(
        tmp_path / "raw",
        {
            5: ["71rtask_00001.tif", "71rtask_00002.tif"],
            7: ["71rtask_00003.tif", "71rtask_00004.tif", "._71rtask_00003.tif"],
        },
    )
    s5, s7 = raw / "ses-005_date-x" / "funcimg", raw / "ses-007_date-x" / "funcimg"
    listing = write_listing(
        tmp_path / "aligned_runs.csv",
        [
            ("ses-005_date-x", "run-01", s5 / "71rtask_00001.tif", True, ""),
            ("ses-005_date-x", "run-02", s5 / "71rtask_00002.tif", False, "no behavior file"),
            ("ses-007_date-x", "run-01", s7 / "71rtask_00003.tif", True, ""),
            ("ses-007_date-x", "run-02", s7 / "71rtask_00004.tif", True, ""),
        ],
    )
    config = tmp_path / "pipeline.toml"
    config.write_text('[distortion]\nmode = "cluster_full"\n')
    batch = BatchSettings(
        raw_subject_dir=raw,
        derivatives_subject_dir=tmp_path / "der",
        session_first=5,
        session_last=7,
        source_substring="71rtask",
        aligned_runs_csv=listing,
        destripe=True,
    )
    return batch, config


def test_only_aligned_tiffs_run_and_the_others_say_why(two_sessions):
    batch, _ = two_sessions
    plan = plan_batch(batch, corrected=True)
    assert [(p.session, p.run_name, p.action) for p in plan] == [
        (5, "run-01", "run"),
        (5, "run-02", "skip"),
        (7, "run-01", "run"),
        (7, "run-02", "run"),
    ]
    assert plan[1].reason == "not aligned to behavior: no behavior file"
    assert plan[0].run_dir.parts[-2:] == ("with_distortion_correction", "with_destriping")


def test_without_a_listing_a_missing_session_still_stops_the_plan(two_sessions):
    batch, _ = two_sessions
    with pytest.raises(FileNotFoundError, match=r"\[6\]"):
        plan_batch(batch.model_copy(update={"aligned_runs_csv": None}), corrected=True)


def test_a_tiff_the_listing_does_not_have_is_skipped(two_sessions):
    batch, _ = two_sessions
    extra = batch.raw_subject_dir / "ses-007_date-x" / "funcimg" / "71rtask_00005.tif"
    extra.write_bytes(b"tif")
    plan = plan_batch(batch, corrected=True)
    assert (plan[-1].action, plan[-1].reason) == ("skip", "not in aligned_runs.csv")


def test_a_listing_that_numbers_a_tiff_differently_conflicts():
    listed = AlignedRun("ses-005_date-x", "run-02", True, "")
    action, reason = alignment_decision(listed, "run-01", "aligned_runs.csv")
    assert action == "conflict" and "run-02" in reason and "run-01" in reason


def test_a_listing_from_other_raw_data_is_refused(two_sessions, tmp_path):
    batch, _ = two_sessions
    other = tmp_path / "elsewhere" / "ses-005_date-x" / "funcimg" / "71rtask_00009.tif"
    write_listing(
        batch.aligned_runs_csv,
        [("ses-005_date-x", "run-01", other, True, "")],
    )
    with pytest.raises(ValueError, match="did not find"):
        plan_batch(batch, corrected=True)


def test_listings_with_a_missing_column_or_a_bad_flag_are_refused(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("session,run,tiff,aligned\nses-005_x,run-01,/a.tif,True\n")
    with pytest.raises(ValueError, match="reason"):
        load_aligned_runs(bad)
    bad.write_text("session,run,tiff,aligned,reason\nses-005_x,run-01,/a.tif,yes,\n")
    with pytest.raises(ValueError, match="True or False"):
        load_aligned_runs(bad)


def stub_worker(monkeypatch, code: str = "pass"):
    import sys

    monkeypatch.setattr(batch_module, "worker_command", lambda *a: [sys.executable, "-c", code])


def test_prepare_writes_one_task_per_pending_run_with_the_frozen_config(two_sessions):
    batch, config = two_sessions
    tasks_path = prepare_array(batch, config)
    tasks = json.loads(tasks_path.read_text())
    assert tasks_path.parent == batch.derivatives_subject_dir
    assert Path(tasks["config"]).read_text() == config.read_text()
    assert Path(tasks["config"]) != config and tasks["destripe"] is True
    assert [(r["session"], r["run_name"]) for r in tasks["runs"]] == [
        (5, "run-01"),
        (7, "run-01"),
        (7, "run-02"),
    ]


def test_each_task_runs_its_row_and_a_resubmitted_task_is_skipped(two_sessions, monkeypatch):
    batch, config = two_sessions
    stub_worker(monkeypatch)
    tasks_path = prepare_array(batch, config)
    assert run_array_task(tasks_path, 2) == "complete"
    run_dir = Path(json.loads(tasks_path.read_text())["runs"][1]["run_dir"])
    manifest = json.loads((run_dir / "source_tiff.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["pipeline_config"] == json.loads(tasks_path.read_text())["config"]
    assert run_array_task(tasks_path, 2) == "skipped"
    assert not Path(json.loads(tasks_path.read_text())["runs"][0]["run_dir"]).exists()


def test_a_failed_task_is_recorded_and_runs_again_when_resubmitted(two_sessions, monkeypatch):
    import subprocess

    batch, config = two_sessions
    tasks_path = prepare_array(batch, config)
    stub_worker(monkeypatch, "raise SystemExit(3)")
    with pytest.raises(subprocess.CalledProcessError):
        run_array_task(tasks_path, 1)
    stub_worker(monkeypatch)
    assert run_array_task(tasks_path, 1) == "complete"


def test_a_task_outside_the_list_or_on_a_changed_tiff_is_refused(two_sessions, monkeypatch):
    batch, config = two_sessions
    stub_worker(monkeypatch)
    tasks_path = prepare_array(batch, config)
    with pytest.raises(IndexError, match="1-3"):
        run_array_task(tasks_path, 4)
    source = Path(json.loads(tasks_path.read_text())["runs"][0]["source"])
    source.write_bytes(b"a different tif")
    with pytest.raises(ValueError, match="changed since the launch"):
        run_array_task(tasks_path, 1)


def test_a_new_launch_skips_what_the_array_finished(two_sessions, monkeypatch):
    batch, config = two_sessions
    stub_worker(monkeypatch)
    tasks_path = prepare_array(batch, config)
    for index in (1, 2, 3):
        run_array_task(tasks_path, index)
    assert prepare_array(batch, config) is None


def test_the_command_prints_the_sbatch_line_and_reads_the_slurm_index(
    two_sessions, monkeypatch, capsys, tmp_path
):
    batch, config = two_sessions
    config.write_text(
        config.read_text()
        + "[batch]\n"
        + "".join(
            f"{key} = {json.dumps(str(value) if isinstance(value, Path) else value)}\n"
            for key, value in batch.model_dump().items()
            if value is not None and key not in ("rerun_completed_before",)
        )
    )
    batch_main(["plan", "--config", str(config)])
    assert "not aligned to behavior" in capsys.readouterr().out
    batch_main(["prepare", "--config", str(config)])
    out = capsys.readouterr().out
    assert "sbatch --array=1-3 scripts/slurm/batch_array.sh" in out
    tasks = next(batch.derivatives_subject_dir.glob("batch_tasks_*.json"))
    stub_worker(monkeypatch)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    batch_main(["task", "--tasks", str(tasks)])
    assert capsys.readouterr().out.strip() == "Task 3: complete"

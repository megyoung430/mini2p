"""The batch limited to the TIFFs that align to behavior, and run as a SLURM array."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mini2p_preprocessing import batch as batch_module
from mini2p_preprocessing.batch import (
    MANIFEST_NAME,
    AlignedRun,
    alignment_decision,
    load_aligned_runs,
    load_batch_settings,
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


def write_config(path: Path, raw: Path, der: Path, listing: Path | None, extra: str = "") -> Path:
    batch = (
        "[batch]\n"
        f'raw_subject_dir = "{raw}"\n'
        f'derivatives_subject_dir = "{der}"\n'
        "session_first = 5\nsession_last = 7\n"
        'source_substring = "71rtask"\ndestripe = true\n'
    )
    if listing is not None:
        batch += f'aligned_runs_csv = "{listing}"\n'
    path.write_text('[distortion]\nmode = "cluster_full"\n' + extra + batch)
    return path


@pytest.fixture(autouse=True)
def no_slurm(monkeypatch):
    """No SLURM in the tests: no queued jobs, and job states unknown unless a test says."""
    monkeypatch.setattr(batch_module, "_squeue", lambda *args: None)
    for name in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def config(tmp_path):
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
            ("ses-005_date-x", "run-01", s5 / "71rtask_00001.tif", False, "no behavior file"),
            ("ses-005_date-x", "run-02", s5 / "71rtask_00002.tif", True, ""),
            ("ses-007_date-x", "run-01", s7 / "71rtask_00003.tif", True, ""),
            ("ses-007_date-x", "run-02", s7 / "71rtask_00004.tif", True, ""),
        ],
    )
    return write_config(tmp_path / "pipeline.toml", raw, tmp_path / "der", listing)


def squeue(stdout: str = "", returncode: int = 0, stderr: str = ""):
    """A stand-in for ``_squeue`` returning what squeue would print."""
    result = subprocess.CompletedProcess([], returncode, stdout, stderr)
    return lambda *args: result


INVALID_JOB = squeue(returncode=1, stderr="slurm_load_jobs error: Invalid job id specified")


def plan_of(config: Path):
    return plan_batch(load_batch_settings(config), corrected=True)


def tasks_of(path: Path) -> dict:
    return json.loads(path.read_text())


def stub_worker(monkeypatch, code: str = "pass"):
    monkeypatch.setattr(batch_module, "worker_command", lambda *a: [sys.executable, "-c", code])


# ---- planning with the listing ------------------------------------------------------


def test_only_aligned_tiffs_run_and_a_skipped_one_keeps_its_number(config):
    plan = plan_of(config)
    assert [(p.session, p.run_name, p.source.name, p.action) for p in plan] == [
        (5, "run-01", "71rtask_00001.tif", "skip"),
        (5, "run-02", "71rtask_00002.tif", "run"),
        (7, "run-01", "71rtask_00003.tif", "run"),
        (7, "run-02", "71rtask_00004.tif", "run"),
    ]
    assert plan[0].reason == "not aligned to behavior: no behavior file"
    assert plan[1].run_dir.parts[-3:] == ("run-02", "with_distortion_correction", "with_destriping")


def test_without_a_listing_a_missing_session_still_stops_the_plan(config):
    batch = load_batch_settings(config).model_copy(update={"aligned_runs_csv": None})
    with pytest.raises(FileNotFoundError, match=r"\[6\]"):
        plan_batch(batch, corrected=True)


def test_a_new_tiff_of_a_listed_session_is_skipped_as_not_listed(config):
    raw = load_batch_settings(config).raw_subject_dir
    (raw / "ses-007_date-x" / "funcimg" / "71rtask_00005.tif").write_bytes(b"tif")
    plan = plan_of(config)
    assert (plan[-1].action, plan[-1].reason) == ("skip", "not in aligned_runs.csv")


def test_a_session_the_listing_does_not_cover_is_refused(config):
    raw = load_batch_settings(config).raw_subject_dir
    make_raw_tree(raw, {6: ["71rtask_00009.tif"]})  # recorded after the listing
    with pytest.raises(ValueError, match=r"lists no TIFF of session\(s\) \[6\]"):
        plan_of(config)


def test_a_listing_that_numbers_a_tiff_differently_stops_the_launch(config, tmp_path):
    batch = load_batch_settings(config)
    rows = list(csv.DictReader(batch.aligned_runs_csv.open()))
    rows[1]["run"] = "run-01"
    write_listing(
        batch.aligned_runs_csv,
        [(r["session"], r["run"], r["tiff"], r["aligned"], r["reason"]) for r in rows],
    )
    assert [p.action for p in plan_of(config)][:2] == ["skip", "conflict"]
    with pytest.raises(RuntimeError, match="numbers this TIFF run-01"):
        prepare_array(config)


def test_the_decision_names_both_numbers():
    listed = AlignedRun("ses-005_date-x", "run-02", True, "")
    action, reason = alignment_decision(listed, "run-01", "aligned_runs.csv")
    assert action == "conflict" and "run-02" in reason and "run-01" in reason


def test_output_already_under_another_run_number_conflicts(config, monkeypatch):
    stub_worker(monkeypatch)
    tasks, _ = prepare_array(config)
    run_array_task(tasks, 1)  # 71rtask_00002 in run-02
    raw = load_batch_settings(config).raw_subject_dir
    (raw / "ses-005_date-x" / "funcimg" / "71rtask_00001.tif").unlink()  # 00002 becomes run-01
    batch = load_batch_settings(config).model_copy(update={"aligned_runs_csv": None})
    batch = batch.model_copy(update={"session_last": 5})
    plan = plan_batch(batch, corrected=True)
    assert plan[0].action == "conflict" and "already in run-02" in plan[0].reason


def test_a_listing_from_other_raw_data_is_refused(config, tmp_path):
    other = tmp_path / "elsewhere" / "ses-005_date-x" / "funcimg" / "71rtask_00009.tif"
    write_listing(
        load_batch_settings(config).aligned_runs_csv,
        [("ses-005_date-x", "run-01", other, True, "")],
    )
    with pytest.raises(ValueError, match="did not find"):
        plan_of(config)


def test_listings_with_a_missing_column_a_bad_flag_or_session_are_refused(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("session,run,tiff,aligned\nses-005_x,run-01,/a.tif,True\n")
    with pytest.raises(ValueError, match="reason"):
        load_aligned_runs(bad)
    bad.write_text("session,run,tiff,aligned,reason\nses-005_x,run-01,/a.tif,yes,\n")
    with pytest.raises(ValueError, match="True or False"):
        load_aligned_runs(bad)
    bad.write_text("session,run,tiff,aligned,reason\nses-test_x,run-01,/a.tif,True,\n")
    with pytest.raises(ValueError, match="ses-<number>"):
        load_aligned_runs(bad)


def test_apple_double_files_are_not_counted_as_tiffs_without_a_listing(config):
    batch = load_batch_settings(config).model_copy(
        update={"aligned_runs_csv": None, "session_first": 7}
    )
    names = [p.source.name for p in plan_batch(batch, corrected=True)]
    assert names == ["71rtask_00003.tif", "71rtask_00004.tif"]


# ---- the array ------------------------------------------------------------------------


def test_prepare_writes_one_task_per_pending_run_with_the_frozen_config(config):
    tasks_path, launch = prepare_array(config)
    tasks = tasks_of(tasks_path)
    assert tasks_path.parent == load_batch_settings(config).derivatives_subject_dir
    assert Path(tasks["config"]).read_text() == config.read_text() and tasks["destripe"] is True
    assert [(r["session"], r["run_name"]) for r in tasks["runs"]] == [
        (5, "run-02"),
        (7, "run-01"),
        (7, "run-02"),
    ]
    assert len(launch.pending) == 3


def test_prepare_refuses_while_this_mouse_s_array_jobs_are_queued(config, monkeypatch):
    monkeypatch.setenv("USER", "me")
    der = load_batch_settings(config).derivatives_subject_dir
    monkeypatch.setattr(batch_module, "_squeue", squeue(f"4711_3|{der}\n4711_4|{der}\n"))
    with pytest.raises(RuntimeError, match="still queued or running"):
        prepare_array(config)
    assert prepare_array(config, allow_queued=True)[0] is not None


def test_another_mouse_s_queued_jobs_do_not_block_prepare(config, monkeypatch):
    monkeypatch.setenv("USER", "me")
    monkeypatch.setattr(batch_module, "_squeue", squeue("4711_3|/der/other_mouse\n"))
    assert prepare_array(config)[0] is not None


def test_prepare_refuses_a_node_local_fast_disk(config, tmp_path):
    batch_text = config.read_text().split("[batch]")[1]
    config.write_text(
        '[distortion]\nmode = "cluster_full"\n[registration]\nfast_disk = "/tmp/s2p"\n'
        "[batch]" + batch_text
    )
    with pytest.raises(ValueError, match="private to each compute node"):
        prepare_array(config)


def test_each_task_runs_its_row_and_a_resubmitted_task_is_skipped(config, monkeypatch):
    stub_worker(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "4712")
    tasks_path, _ = prepare_array(config)
    assert run_array_task(tasks_path, 2) == "complete"
    run_dir = Path(tasks_of(tasks_path)["runs"][1]["run_dir"])
    manifest = json.loads((run_dir / MANIFEST_NAME).read_text())
    assert manifest["status"] == "complete" and manifest["slurm_job_id"] == "4712"
    assert manifest["pipeline_config"] == tasks_of(tasks_path)["config"]
    assert run_array_task(tasks_path, 2).startswith("skipped: already complete")
    assert not Path(tasks_of(tasks_path)["runs"][0]["run_dir"]).exists()


def test_a_row_another_live_job_is_running_is_skipped(config, monkeypatch):
    stub_worker(monkeypatch)
    tasks_path, _ = prepare_array(config)
    run_dir = Path(tasks_of(tasks_path)["runs"][0]["run_dir"])
    run_dir.mkdir(parents=True)
    (run_dir / MANIFEST_NAME).write_text(json.dumps({"status": "running", "slurm_job_id": "99"}))
    monkeypatch.setattr(batch_module, "_squeue", squeue("RUNNING\n"))
    assert run_array_task(tasks_path, 1) == "skipped: SLURM job 99 is running it"
    monkeypatch.setattr(batch_module, "_squeue", INVALID_JOB)  # SLURM has forgotten job 99
    assert run_array_task(tasks_path, 1) == "complete"


@pytest.mark.parametrize(
    "answer, alive",
    [
        (squeue("PENDING\n"), True),
        (squeue("COMPLETING\n"), True),
        (squeue("TIMEOUT\n"), False),  # listed for a while after it ended
        (squeue("OUT_OF_MEMORY\n"), False),
        (INVALID_JOB, False),
        (squeue(returncode=1, stderr="slurm_load_jobs error: Socket timed out"), None),
        (lambda *args: None, None),  # no squeue on this machine
    ],
)
def test_job_liveness_follows_the_state_squeue_reports(monkeypatch, answer, alive):
    monkeypatch.setattr(batch_module, "_squeue", answer)
    assert batch_module.slurm_job_alive("99") is alive


def test_a_run_whose_job_ended_while_running_is_planned_again(config, monkeypatch):
    tasks_path, _ = prepare_array(config)
    row = tasks_of(tasks_path)["runs"][0]
    run_dir = Path(row["run_dir"])
    run_dir.mkdir(parents=True)
    manifest = {"source_tiff": row["source"], **row["identity"], "status": "running"}
    (run_dir / MANIFEST_NAME).write_text(json.dumps({**manifest, "slurm_job_id": "99"}))
    monkeypatch.setattr(batch_module, "_squeue", squeue("RUNNING\n"))
    assert plan_of(config)[1].action == "skip"
    monkeypatch.setattr(batch_module, "_squeue", INVALID_JOB)  # killed, e.g. out of memory
    assert plan_of(config)[1].action == "run"


def test_a_later_launch_s_result_is_not_replaced_by_an_older_task(config, monkeypatch):
    stub_worker(monkeypatch)
    first, _ = prepare_array(config)
    second, _ = prepare_array(config)
    run_array_task(second, 1)
    assert run_array_task(first, 1).startswith("skipped: a later launch")
    run_dir = Path(tasks_of(first)["runs"][0]["run_dir"])
    manifest = json.loads((run_dir / MANIFEST_NAME).read_text())
    assert manifest["pipeline_config"] == tasks_of(second)["config"]


def test_a_tiff_recorded_in_another_variant_under_another_number_conflicts(config, monkeypatch):
    stub_worker(monkeypatch)
    tasks_path, _ = prepare_array(config)  # with_destriping
    run_array_task(tasks_path, 1)  # 71rtask_00002 -> run-02/.../with_destriping
    raw = load_batch_settings(config).raw_subject_dir
    (raw / "ses-005_date-x" / "funcimg" / "71rtask_00001.tif").unlink()  # 00002 becomes run-01
    batch = load_batch_settings(config).model_copy(
        update={"aligned_runs_csv": None, "session_last": 5, "destripe": False}
    )
    plan = plan_batch(batch, corrected=True)  # without_destriping: nothing recorded there yet
    assert plan[0].action == "conflict" and "already in run-02" in plan[0].reason


def test_manifests_are_replaced_whole(config, monkeypatch, tmp_path):
    stub_worker(monkeypatch)
    tasks_path, _ = prepare_array(config)
    run_array_task(tasks_path, 1)
    run_dir = Path(tasks_of(tasks_path)["runs"][0]["run_dir"])
    assert [p.name for p in run_dir.iterdir() if p.name.endswith(".partial")] == []


def test_a_failed_task_is_recorded_and_runs_again_when_resubmitted(config, monkeypatch):
    tasks_path, _ = prepare_array(config)
    stub_worker(monkeypatch, "raise SystemExit(3)")
    with pytest.raises(subprocess.CalledProcessError):
        run_array_task(tasks_path, 1)
    stub_worker(monkeypatch)
    assert run_array_task(tasks_path, 1) == "complete"


def test_sigterm_marks_the_run_interrupted(config, monkeypatch):
    tasks_path, _ = prepare_array(config)
    stub_worker(
        monkeypatch,
        "import os, signal, time; os.kill(os.getppid(), signal.SIGTERM); time.sleep(30)",
    )
    with pytest.raises(batch_module._Terminated):
        run_array_task(tasks_path, 1)
    run_dir = Path(tasks_of(tasks_path)["runs"][0]["run_dir"])
    assert json.loads((run_dir / MANIFEST_NAME).read_text())["status"] == "interrupted"
    assert plan_of(config)[1].action == "run"  # retried by the next launch


def test_a_task_outside_the_list_or_on_a_changed_tiff_is_refused(config, monkeypatch):
    stub_worker(monkeypatch)
    tasks_path, _ = prepare_array(config)
    with pytest.raises(IndexError, match="1-3"):
        run_array_task(tasks_path, 4)
    Path(tasks_of(tasks_path)["runs"][0]["source"]).write_bytes(b"a different tif")
    with pytest.raises(ValueError, match="changed since the launch"):
        run_array_task(tasks_path, 1)


def test_a_new_launch_skips_what_the_array_finished(config, monkeypatch):
    stub_worker(monkeypatch)
    tasks_path, _ = prepare_array(config)
    for index in (1, 2, 3):
        run_array_task(tasks_path, index)
    assert prepare_array(config)[0] is None


def test_a_config_without_a_batch_table_is_refused(tmp_path):
    config = tmp_path / "pipeline.toml"
    config.write_text('[distortion]\nmode = "off"\n')
    with pytest.raises(ValueError, match="no \\[batch\\] table"):
        prepare_array(config)


# ---- the command ------------------------------------------------------------------------


def test_the_command_prints_skips_the_sbatch_line_and_reads_the_slurm_index(
    config, monkeypatch, capsys
):
    batch_main(["plan", "--config", str(config)])
    assert "not aligned to behavior" in capsys.readouterr().out
    batch_main(["prepare", "--config", str(config)])
    out = capsys.readouterr().out
    assert "skip    1: not aligned to behavior: no behavior file" in out
    der = load_batch_settings(config).derivatives_subject_dir
    assert (
        f'MINI2P_ENV={sys.prefix} sbatch --array=1-3 --comment="{der}" scripts/slurm/batch_array.sh'
    ) in out
    tasks = next(load_batch_settings(config).derivatives_subject_dir.glob("batch_tasks_*.json"))
    stub_worker(monkeypatch)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    batch_main(["task", "--tasks", str(tasks)])
    assert capsys.readouterr().out.strip() == "Task 3: complete"


def test_the_command_reports_a_bad_config_as_a_usage_error(tmp_path, capsys):
    config = tmp_path / "pipeline.toml"
    config.write_text('[distortion]\nmode = "off"\n')
    with pytest.raises(SystemExit):
        batch_main(["prepare", "--config", str(config)])
    assert "no [batch] table" in capsys.readouterr().err

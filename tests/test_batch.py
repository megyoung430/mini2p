from datetime import UTC, datetime
from pathlib import Path

import pytest

from mini2p_preprocessing.batch import (
    BatchSettings,
    ExistingRun,
    acquisition_key,
    decide_action,
    find_sessions,
    plan_batch,
    run_folder_conflict,
)

IDENTITY = {"source_size_bytes": 10, "source_mtime_ns": 5}
SRC = Path("/raw/71rtask_00002.tif")
RUN = Path("/der/run-02/without_destriping")


def manifest_run(status, source=SRC, identity=IDENTITY, run_dir=RUN):
    return ExistingRun(run_dir, source, {**identity, "status": status}, "manifest")


def decide(**kwargs):
    args = {
        "source": SRC,
        "identity": IDENTITY,
        "run_dir": RUN,
        "run_dir_exists": True,
        "prior_elsewhere": None,
        "prior_here": None,
        "retry_incomplete": True,
        "rerun_before": None,
    }
    return decide_action(**{**args, **kwargs})[0]


def test_acquisition_counter_sorts_numerically():
    names = ["a_00010.tif", "a_00002.tif", "notes.tif", "a_00001.tif"]
    ordered = [p.name for p in sorted(map(Path, names), key=acquisition_key)]
    assert ordered == ["a_00001.tif", "a_00002.tif", "a_00010.tif", "notes.tif"]


def test_decisions_cover_every_branch():
    assert decide(run_dir_exists=False) == "run"
    assert decide(prior_elsewhere=manifest_run("complete", run_dir=Path("/der/run-01"))) == "skip"
    assert decide(prior_here=None) == "conflict"
    assert decide(prior_here=manifest_run("complete", source=Path("/raw/other.tif"))) == "conflict"
    changed = {"source_size_bytes": 11, "source_mtime_ns": 5}
    assert decide(prior_here=manifest_run("complete", identity=changed)) == "conflict"
    assert decide(prior_here=manifest_run("complete")) == "skip"
    assert decide(prior_here=manifest_run("failed")) == "run"
    assert decide(prior_here=manifest_run("failed"), retry_incomplete=False) == "skip"
    assert decide(prior_here=manifest_run("running")) == "skip"
    sidecar = ExistingRun(RUN, SRC, {}, "phase sidecar")
    assert decide(prior_here=sidecar) == "skip"
    cutoff = datetime(2026, 10, 21, 9, tzinfo=UTC)
    assert decide(prior_here=sidecar, rerun_before=cutoff) == "run"  # no finish time: older
    finished = lambda iso: ExistingRun(  # noqa: E731
        RUN, SRC, {**IDENTITY, "status": "complete", "finished_utc": iso}, "manifest"
    )
    assert decide(prior_here=finished("2026-10-21T08:59:00+00:00"), rerun_before=cutoff) == "run"
    assert decide(prior_here=finished("2026-10-21T10:01:00+01:00"), rerun_before=cutoff) == "skip"
    assert decide(prior_here=manifest_run("running"), rerun_before=cutoff) == "skip"
    elsewhere = manifest_run("complete", run_dir=Path("/der/run-01"))
    assert decide(prior_elsewhere=elsewhere, rerun_before=cutoff) == "skip"


def make_raw_tree(root: Path, sessions: dict[int, list[str]]) -> Path:
    for number, files in sessions.items():
        funcimg = root / f"ses-{number:03d}_date-x" / "funcimg"
        funcimg.mkdir(parents=True)
        for name in files:
            (funcimg / name).write_bytes(b"tif")
    return root


def test_find_sessions_reports_gaps(tmp_path):
    make_raw_tree(tmp_path, {1: [], 3: []})
    with pytest.raises(FileNotFoundError, match=r"\[2\]"):
        find_sessions(tmp_path, 1, 3)


def test_plan_assigns_runs_in_acquisition_order(tmp_path):
    raw = make_raw_tree(
        tmp_path / "raw",
        {5: ["71rtask_00012.tif", "71rtask_00003.tif", "other_00001.tif", "71rtask.txt"]},
    )
    batch = BatchSettings(
        raw_subject_dir=raw,
        derivatives_subject_dir=tmp_path / "der",
        session_first=5,
        session_last=5,
        source_substring="71rtask",
    )
    plan = plan_batch(batch, corrected=True)
    assert [(p.run_name, p.source.name, p.action) for p in plan] == [
        ("run-01", "71rtask_00003.tif", "run"),
        ("run-02", "71rtask_00012.tif", "run"),
    ]
    assert plan[0].run_dir.name == "without_destriping"


def _batch_with_two_sources(tmp_path):
    raw = make_raw_tree(tmp_path / "raw", {5: ["71rtask_00001.tif", "71rtask_00002.tif"]})
    config = tmp_path / "pipeline.toml"
    config.write_text('[distortion]\nmode = "cluster_full"\n')  # corrected: run-XX/<destriping>
    batch = BatchSettings(
        raw_subject_dir=raw,
        derivatives_subject_dir=tmp_path / "der",
        session_first=5,
        session_last=5,
        source_substring="71rtask",
    )
    return batch, config


def _stub_worker(monkeypatch, code: str):
    import sys

    from mini2p_preprocessing import batch as batch_module

    monkeypatch.setattr(batch_module, "worker_command", lambda *a: [sys.executable, "-c", code])


def test_run_one_marks_failure_with_log_tail(tmp_path, monkeypatch):
    import json
    import subprocess

    from mini2p_preprocessing.batch import MANIFEST_NAME, run_one

    batch, config = _batch_with_two_sources(tmp_path)
    row = plan_batch(batch, corrected=True)[0]
    _stub_worker(monkeypatch, "print('boom'); raise SystemExit(3)")
    with pytest.raises(subprocess.CalledProcessError):
        run_one(row, config, destripe=False)
    manifest = json.loads((row.run_dir / MANIFEST_NAME).read_text())
    assert manifest["status"] == "failed" and "boom" in manifest["error_tail"]
    assert decide(prior_here=manifest_run("failed")) == "run"


def test_relaunch_replans_and_never_redoes_completed_runs(tmp_path, monkeypatch):
    from mini2p_preprocessing import batch as batch_module

    batch, config = _batch_with_two_sources(tmp_path)
    _stub_worker(monkeypatch, "pass")
    launched = []
    real_run_one = batch_module.run_one
    monkeypatch.setattr(
        batch_module,
        "run_one",
        lambda row, cfg, destripe: (
            launched.append((row.run_name, cfg)),
            real_run_one(row, cfg, destripe),
        ),
    )
    first = batch_module.run_batch(batch, config)
    assert [p.action for p in first] == ["run", "run"]
    assert len(launched) == 2
    snapshot = launched[0][1]
    assert snapshot != config and snapshot.read_text() == config.read_text()

    second = batch_module.run_batch(batch, config)
    assert [p.action for p in second] == ["skip", "skip"]
    assert len(launched) == 2


def test_corrupt_manifest_stops_planning(tmp_path):
    from mini2p_preprocessing.batch import MANIFEST_NAME

    batch, _ = _batch_with_two_sources(tmp_path)
    run_dir = plan_batch(batch, corrected=True)[0].run_dir
    run_dir.mkdir(parents=True)
    (run_dir / MANIFEST_NAME).write_text("{not json")
    with pytest.raises(ValueError, match="Corrupt JSON"):
        plan_batch(batch, corrected=True)


@pytest.mark.parametrize("first", [False, True])
def test_both_destriping_variants_coexist_in_either_order(tmp_path, monkeypatch, first):
    from mini2p_preprocessing import batch as batch_module

    batch, config = _batch_with_two_sources(tmp_path)
    _stub_worker(monkeypatch, "pass")
    for destripe in (first, not first):
        plan = batch_module.run_batch(batch.model_copy(update={"destripe": destripe}), config)
        assert [p.action for p in plan] == ["run", "run"]
        expected = "with_destriping" if destripe else "without_destriping"
        assert [p.run_dir.name for p in plan] == [expected, expected]
        assert [p.run_dir.parent.name for p in plan] == ["with_distortion_correction"] * 2
        assert [p.run_dir.parents[1].name for p in plan] == ["run-01", "run-02"]
    for destripe in (False, True):
        again = plan_batch(batch.model_copy(update={"destripe": destripe}), corrected=True)
        assert [p.action for p in again] == ["skip", "skip"]


def test_segmentation_registration_in_run_folder_is_left_alone(tmp_path):
    import json

    from mini2p_preprocessing.batch import MANIFEST_NAME

    batch, _ = _batch_with_two_sources(tmp_path)
    first_source = plan_batch(batch, corrected=True)[0].source
    runs = tmp_path / "der" / "ses-005_date-x" / "funcimg"
    for run in ("run-01", "run-02"):
        (runs / run).mkdir(parents=True)
        (runs / run / MANIFEST_NAME).write_text(json.dumps({"source_tiff": str(first_source)}))
    for destripe in (False, True):
        plan = plan_batch(batch.model_copy(update={"destripe": destripe}), corrected=True)
        assert plan[0].action == "run"  # run-01/ registered the same TIFF
        assert plan[1].action == "conflict"  # run-02/ belongs to run-01's TIFF
        assert "71rtask_00001.tif" in plan[1].reason


def test_rerun_cutoff_replaces_older_runs_once(tmp_path, monkeypatch):
    from mini2p_preprocessing import batch as batch_module

    batch, config = _batch_with_two_sources(tmp_path)
    _stub_worker(monkeypatch, "pass")
    batch_module.run_batch(batch, config)
    rerun = batch.model_copy(update={"rerun_completed_before": datetime.now(UTC)})
    first = batch_module.run_batch(rerun, config)
    assert [p.action for p in first] == ["run", "run"]
    assert all("rerun_completed_before" in p.reason for p in first)
    # Relaunching with the same cutoff keeps what the rerun already replaced.
    assert [p.action for p in plan_batch(rerun, corrected=True)] == ["skip", "skip"]


def test_rerun_cutoff_needs_a_time_zone():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="rerun_completed_before"):
        BatchSettings(
            raw_subject_dir="/r",
            derivatives_subject_dir="/d",
            session_first=1,
            session_last=1,
            source_substring="x",
            rerun_completed_before=datetime(2026, 10, 21, 9),
        )


def test_rerun_refuses_a_config_that_will_not_overwrite(tmp_path, monkeypatch):
    from mini2p_preprocessing import batch as batch_module

    batch, config = _batch_with_two_sources(tmp_path)
    _stub_worker(monkeypatch, "pass")
    batch_module.run_batch(batch, config)
    config.write_text('overwrite = false\n[distortion]\nmode = "cluster_full"\n')
    with pytest.raises(ValueError, match="overwrite"):
        rerun = batch.model_copy(update={"rerun_completed_before": datetime.now(UTC)})
        batch_module.run_batch(rerun, config)


def test_only_a_different_tiff_in_the_run_folder_conflicts():
    here = Path("/der/run-02")
    assert run_folder_conflict(None, SRC) is None
    assert run_folder_conflict(ExistingRun(here, SRC, {}, "manifest"), SRC) is None
    other = ExistingRun(here, Path("/raw/other.tif"), {}, "phase sidecar")
    assert "other.tif" in run_folder_conflict(other, SRC)


def test_config_snapshots_never_overwrite_each_other(tmp_path):
    from mini2p_preprocessing.batch import snapshot_config

    config = tmp_path / "pipeline.toml"
    config.write_text("x = 1\n")
    first, second = (snapshot_config(config, tmp_path / "der") for _ in range(2))
    assert first != second and first.read_text() == second.read_text() == "x = 1\n"


def test_finished_by_batch_needs_a_complete_manifest_since_the_cutoff(tmp_path):
    import json

    from mini2p_preprocessing.batch import MANIFEST_NAME, finished_by_batch

    variant = tmp_path / "run-01" / "without_destriping"
    variant.mkdir(parents=True)
    cutoff = datetime(2026, 10, 21, 6, tzinfo=UTC)
    assert not finished_by_batch(variant, cutoff)  # no manifest yet

    def record(**fields):
        (variant / MANIFEST_NAME).write_text(json.dumps({"source_tiff": "/raw/a.tif", **fields}))

    record(status="running", started_utc="2026-10-21T06:30:00+00:00")
    assert not finished_by_batch(variant, cutoff)
    record(status="complete", finished_utc="2026-10-20T23:00:00+00:00")
    assert not finished_by_batch(variant, cutoff)  # finished before the rerun: old output
    assert finished_by_batch(variant, None)  # no rerun cutoff: any completed run
    record(status="complete", finished_utc="2026-10-21T07:10:00+01:00")
    assert finished_by_batch(variant, cutoff)
    (variant / MANIFEST_NAME).unlink()
    (variant / "a.phase.json").write_text(json.dumps({"input_tiff": "/raw/a.tif"}))
    assert not finished_by_batch(variant, None)  # older output without a batch record


def test_each_run_folder_says_why_it_is_or_is_not_ready(tmp_path):
    import json

    from mini2p_preprocessing.batch import MANIFEST_NAME, run_state

    cutoff = datetime(2026, 10, 21, 6, tzinfo=UTC)
    variant = tmp_path / "run-01" / "without_destriping"
    assert run_state(variant, cutoff) == "not started"
    variant.mkdir(parents=True)
    assert run_state(variant, cutoff) == "no batch record"

    def state(**fields):
        (variant / MANIFEST_NAME).write_text(json.dumps({"source_tiff": "/raw/a.tif", **fields}))
        return run_state(variant, cutoff)

    assert state(status="running") == "running"
    assert state(status="failed") == "failed"
    assert state(status="interrupted") == "interrupted"
    assert state(status="complete") == "awaiting rerun"  # no finish time recorded
    assert state(status="complete", finished_utc="2026-10-21T05:59:59+00:00") == "awaiting rerun"
    assert state(status="complete", finished_utc="2026-10-21T06:00:00+00:00") == "ready"
    assert run_state(variant, None) == "ready"


def test_the_cutoff_comes_from_the_running_batch_not_the_live_config(tmp_path):
    from mini2p_preprocessing.batch import launch_cutoff

    with pytest.raises(FileNotFoundError, match="batch_config"):
        launch_cutoff(tmp_path)
    body = '[batch]\nraw_subject_dir = "/r"\nderivatives_subject_dir = "/d"\nsession_first = 1\n'
    body += 'session_last = 2\nsource_substring = "x"\n'
    (tmp_path / "batch_config_20261020T224733321160Z.toml").write_text(
        body + "rerun_completed_before = 2026-10-20T23:00:00+01:00\n"
    )
    (tmp_path / "batch_config_20261021T055000000000Z.toml").write_text(
        body + "rerun_completed_before = 2026-10-21T06:00:00+01:00\n"
    )
    assert launch_cutoff(tmp_path) == datetime(2026, 10, 21, 5, tzinfo=UTC)


def test_runs_without_distortion_correction_get_their_own_folders(tmp_path, monkeypatch):
    from mini2p_preprocessing import batch as batch_module

    batch, config = _batch_with_two_sources(tmp_path)
    _stub_worker(monkeypatch, "pass")
    batch_module.run_batch(batch, config)  # distortion corrected
    config.write_text('[distortion]\nmode = "off"\n')
    for destripe in (False, True):
        plan = batch_module.run_batch(batch.model_copy(update={"destripe": destripe}), config)
        assert [p.action for p in plan] == ["run", "run"]  # not the corrected runs' folders
        expected = "with_destriping" if destripe else "without_destriping"
        assert [p.run_dir.relative_to(p.run_dir.parents[2]).as_posix() for p in plan] == [
            f"run-0{i}/without_distortion_correction/{expected}" for i in (1, 2)
        ]
    for corrected in (True, False):
        again = plan_batch(batch, corrected=corrected)
        assert [p.action for p in again] == ["skip", "skip"]

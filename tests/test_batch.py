from pathlib import Path

import pytest

from mini2p_preprocessing.batch import (
    BatchSettings,
    ExistingRun,
    acquisition_key,
    decide_action,
    find_sessions,
    older_layout_conflict,
    plan_batch,
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
    plan = plan_batch(batch)
    assert [(p.run_name, p.source.name, p.action) for p in plan] == [
        ("run-01", "71rtask_00003.tif", "run"),
        ("run-02", "71rtask_00012.tif", "run"),
    ]
    assert plan[0].run_dir.name == "without_destriping"


def _batch_with_two_sources(tmp_path):
    raw = make_raw_tree(tmp_path / "raw", {5: ["71rtask_00001.tif", "71rtask_00002.tif"]})
    config = tmp_path / "pipeline.toml"
    config.write_text('[distortion]\nmode = "off"\n')
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
    row = plan_batch(batch)[0]
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
    run_dir = plan_batch(batch)[0].run_dir
    run_dir.mkdir(parents=True)
    (run_dir / MANIFEST_NAME).write_text("{not json")
    with pytest.raises(ValueError, match="Corrupt JSON"):
        plan_batch(batch)


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
        assert [p.run_dir.parent.name for p in plan] == ["run-01", "run-02"]
    for destripe in (False, True):
        again = plan_batch(batch.model_copy(update={"destripe": destripe}))
        assert [p.action for p in again] == ["skip", "skip"]


def test_older_destriped_output_directly_in_run_folder_is_flagged(tmp_path):
    import json

    from mini2p_preprocessing.batch import MANIFEST_NAME

    batch, _ = _batch_with_two_sources(tmp_path)
    source = plan_batch(batch)[0].source
    old_layout = tmp_path / "der" / "ses-005_date-x" / "funcimg" / "run-01"
    old_layout.mkdir(parents=True)
    (old_layout / MANIFEST_NAME).write_text(
        json.dumps({"source_tiff": str(source), "status": "complete", "destriping_enabled": True})
    )
    destriped = plan_batch(batch.model_copy(update={"destripe": True}))
    assert destriped[0].action == "conflict"
    assert "with_destriping" in destriped[0].reason
    assert destriped[1].action == "run"
    assert [p.action for p in plan_batch(batch)] == ["run", "run"]  # other variant unaffected


OLD = Path("/der/run-02")


def older(enabled, source=SRC, evidence="manifest"):
    metadata = {} if enabled is None else {"destriping_enabled": enabled}
    return ExistingRun(OLD, source, metadata, evidence)


def test_older_layout_names_where_its_files_belong():
    assert older_layout_conflict(None, SRC, destripe=True) is None
    assert "move it into with_destriping/" in older_layout_conflict(older(True), SRC, True)
    assert older_layout_conflict(older(True), SRC, destripe=False) is None  # other variant
    assert older_layout_conflict(older(False), SRC, destripe=True) is None
    assert "move it into without_destriping/" in older_layout_conflict(older(False), SRC, False)
    sidecar = older_layout_conflict(older(None, evidence="phase sidecar"), SRC, True)
    assert "unknown" in sidecar and "phase sidecar" in sidecar and "move it into" not in sidecar
    for destripe in (False, True):
        other = older_layout_conflict(older(True, Path("/raw/other.tif")), SRC, destripe)
        assert "other.tif" in other


def test_config_snapshots_never_overwrite_each_other(tmp_path):
    from mini2p_preprocessing.batch import snapshot_config

    config = tmp_path / "pipeline.toml"
    config.write_text("x = 1\n")
    first, second = (snapshot_config(config, tmp_path / "der") for _ in range(2))
    assert first != second and first.read_text() == second.read_text() == "x = 1\n"

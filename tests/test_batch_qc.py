"""QC after each batch run: recorded in the manifest, never changing the run's status."""

import json
import sys

import pytest

from mini2p_preprocessing import batch as batch_module
from mini2p_preprocessing.batch import MANIFEST_NAME, QcRequest, plan_batch, run_one
from mini2p_preprocessing.batch import qc_command as real_qc_command  # before conftest stubs it
from tests.test_batch import _batch_with_two_sources, _stub_worker


def stub_qc(monkeypatch, code: str):
    commands = []

    def command(row, qc):
        commands.append((row.run_dir, qc))
        return [sys.executable, "-c", code]

    monkeypatch.setattr(batch_module, "qc_command", command)
    return commands


@pytest.fixture
def row(tmp_path, monkeypatch):
    batch, config = _batch_with_two_sources(tmp_path)
    _stub_worker(monkeypatch, "pass")
    return plan_batch(batch, corrected=True)[0], config


def manifest_of(row) -> dict:
    return json.loads((row.run_dir / MANIFEST_NAME).read_text())


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("pass", "saved"),
        ("raise SystemExit(2)", "some plots failed"),
        ("raise ValueError('bug')", "failed (exit code 1)"),
    ],
)
def test_qc_runs_after_the_run_is_complete_and_never_fails_it(row, monkeypatch, code, status):
    row, config = row
    stub_qc(monkeypatch, code)
    run_one(row, config, destripe=False, qc=QcRequest())
    manifest = manifest_of(row)
    assert manifest["status"] == "complete"
    assert manifest["qc"]["status"] == status
    assert manifest["qc"]["log"] == str(row.run_dir / "qc.log")


def test_a_run_whose_pipeline_fails_gets_no_qc(row, monkeypatch):
    import subprocess

    row, config = row
    _stub_worker(monkeypatch, "raise SystemExit(3)")
    commands = stub_qc(monkeypatch, "pass")
    with pytest.raises(subprocess.CalledProcessError):
        run_one(row, config, destripe=False, qc=QcRequest())
    assert commands == [] and "qc" not in manifest_of(row)


def test_sigterm_during_qc_is_recorded_and_the_run_stays_complete(row, monkeypatch):
    row, config = row

    def terminated(*args, **kwargs):
        raise batch_module._Terminated("signal 15")

    stub_qc(monkeypatch, "pass")
    real_run = batch_module.subprocess.run
    calls = []

    def run_then_terminate(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:  # the QC process, after the pipeline
            terminated()
        return real_run(command, **kwargs)

    monkeypatch.setattr(batch_module.subprocess, "run", run_then_terminate)
    run_one(row, config, destripe=False, qc=QcRequest())
    manifest = manifest_of(row)
    assert manifest["status"] == "complete" and manifest["qc"]["status"] == "interrupted"


def test_without_qc_requested_none_runs(row, monkeypatch):
    row, config = row
    commands = stub_qc(monkeypatch, "pass")
    run_one(row, config, destripe=False)
    assert commands == [] and "qc" not in manifest_of(row)


def test_the_batch_settings_decide_the_request():
    from mini2p_preprocessing.batch import BatchSettings

    base = {"raw_subject_dir": "/r", "derivatives_subject_dir": "/d", "session_first": 1,
            "session_last": 2, "source_substring": "x"}  # fmt: skip
    assert QcRequest.from_batch(BatchSettings(**base)) == QcRequest(frame_order=False)
    assert QcRequest.from_batch(BatchSettings(**base, qc_frame_order=True)) == QcRequest(True)
    assert QcRequest.from_batch(BatchSettings(**base, save_qc=False)) is None


def test_the_qc_process_is_run_qc_on_the_variant_folder(row):
    row, _ = row
    command = real_qc_command(row, QcRequest(frame_order=True))
    assert command[1:3] == ["-m", "mini2p_preprocessing.run_qc"]
    assert command[3:] == ["--variant-dir", str(row.run_dir.resolve()), "--frame-order"]

import json
import sys

import pytest

from mini2p_preproc import cli
from mini2p_preproc import pipeline as pipeline_module
from mini2p_preproc.distortion import load_calibration_module


def test_cli_resolves_relative_paths_and_applies_destripe_override(
    tmp_path, recording, monkeypatch
):
    raw_tiff, _ = recording
    (tmp_path / "pipeline.toml").write_text(
        '[phase]\nchunk_frames = 20\nmax_shift_px = 8\n[distortion]\nmode = "off"\n'
    )
    monkeypatch.setattr(pipeline_module, "run_registration", lambda tiff, out, s: out)
    monkeypatch.chdir(tmp_path)
    out = "der/run-03/without_destriping"
    cli.main(
        [
            "--config",
            "pipeline.toml",
            "--input",
            raw_tiff.name,
            "--output-dir",
            out,
            "--destripe",
            "1",
        ]
    )

    run_dir = tmp_path / out
    sidecar = json.loads(next(run_dir.glob("*.phase.json")).read_text())
    assert sidecar["input_tiff"] == str(raw_tiff.resolve())
    assert sidecar["run_name"] == "run-03"
    assert sidecar["destriping_enabled"] is True
    used = json.loads((run_dir / pipeline_module.SETTINGS_RECORD).read_text())
    assert used["destripe"]["enabled"] is True


CALIBRATION_SOURCE = """
from __future__ import annotations
from dataclasses import dataclass
from calibration_helpers import SCALE

@dataclass
class Transform:
    k: float

def load_transform(path, z_um):
    return Transform(SCALE), 0.0, "test", 3

def make_maps(transform, shape):
    import numpy as np
    return np.mgrid[0:shape[0], 0:shape[1]].astype(float)

def crop_bounds(map_y, map_x, shape, padding):
    return (padding, shape[0] - padding, padding, shape[1] - padding)
"""


def test_calibration_script_with_dataclass_and_sibling_import_loads(tmp_path):
    folder = tmp_path / "scripts"
    folder.mkdir()
    (folder / "distortion_correct_full_recording.py").write_text(CALIBRATION_SOURCE)
    (folder / "calibration_helpers.py").write_text("SCALE = 0.5\n")
    module = load_calibration_module(folder)
    assert module.load_transform("x.mat", 0)[0].k == 0.5
    sys.modules.pop("distortion_correct_full_recording", None)
    sys.modules.pop("calibration_helpers", None)


def test_missing_calibration_script_says_which_setting_to_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibration_script_dir"):
        load_calibration_module(tmp_path)

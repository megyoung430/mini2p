import json

from mini2p_preprocessing import cli
from mini2p_preprocessing import pipeline as pipeline_module


def test_cli_resolves_relative_paths_and_applies_destripe_override(
    tmp_path, recording, monkeypatch
):
    raw_tiff, _ = recording
    (tmp_path / "pipeline.toml").write_text(
        '[phase]\nchunk_frames = 20\nmax_shift_px = 8\n[distortion]\nmode = "off"\n'
    )
    monkeypatch.setattr(pipeline_module, "run_registration", lambda tiff, out, s: out)
    monkeypatch.chdir(tmp_path)
    out = "der/run-03/without_distortion_correction/with_destriping"  # [distortion] is off
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

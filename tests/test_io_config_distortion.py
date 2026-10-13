from pathlib import Path

import numpy as np
import pytest
import tifffile
from pydantic import ValidationError

from mini2p_preproc.config import (
    PipelineSettings,
    corrected_tiff_path,
    load_settings,
    run_name_from_output_dir,
    variant_output_dir,
    with_destriping,
)
from mini2p_preproc.distortion import DistortionMaps, apply_distortion_to_chunk
from mini2p_preproc.tiff_io import (
    chunk_bounds,
    count_timepoints,
    read_all_channels,
    write_interleaved,
)
from tests.conftest import write_interleaved_tiff


def test_chunk_bounds_cover_recording_with_short_tail():
    assert chunk_bounds(25, 10) == [(0, 10), (10, 20), (20, 25)]
    assert chunk_bounds(0, 10) == []


def test_interleaved_round_trip(tmp_path, rng):
    stacks = [rng.integers(-50, 50, size=(4, 6, 8)).astype(np.int16) for _ in range(2)]
    path = write_interleaved_tiff(tmp_path / "in.tif", stacks)
    with tifffile.TiffFile(path) as tif:
        assert count_timepoints(tif, 2) == 4
        assert count_timepoints(tif, 2, max_timepoints=3) == 3
        back = read_all_channels(tif, 1, 3, 2)
    assert all(np.array_equal(b, s[1:3]) for b, s in zip(back, stacks, strict=True))

    out = tmp_path / "out.tif"
    with tifffile.TiffWriter(out) as writer:
        write_interleaved(writer, stacks, np.dtype("float32"))
    with tifffile.TiffFile(out) as tif:
        assert np.array_equal(read_all_channels(tif, 0, 4, 2)[1], stacks[1])


def test_count_timepoints_rejects_partial_channel(tmp_path, rng):
    path = tmp_path / "odd.tif"
    tifffile.imwrite(
        path, rng.integers(0, 9, size=(5, 4, 4)).astype(np.int16), photometric="minisblack"
    )
    with tifffile.TiffFile(path) as tif, pytest.raises(ValueError, match="not divisible"):
        count_timepoints(tif, 2)


def test_settings_reject_unknown_source_and_typos():
    with pytest.raises(ValidationError):
        PipelineSettings.model_validate({"phase": {"alignment_source": "functinal"}})
    with pytest.raises(ValidationError):
        PipelineSettings.model_validate({"phase": {"chunk_frame": 10}})
    with pytest.raises(ValidationError, match="transform_mat"):
        PipelineSettings.model_validate({"distortion": {"mode": "local_batched"}})


def test_load_settings_reads_toml(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text(
        '[phase]\nchunk_frames = 25\n[phase.manual_first_phase_by_start]\n"200" = -17\n'
        '[distortion]\nmode = "off"\n'
    )
    settings = load_settings(path)
    assert settings.phase.chunk_frames == 25
    assert settings.phase.manual_first_phase_by_start == {200: -17}


def test_distortion_mode_must_be_stated(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text("[phase]\nchunk_frames = 25\n")
    with pytest.raises(ValidationError, match="distortion"):
        load_settings(path)
    path.write_text("[distortion]\nz_um = 0\n")
    with pytest.raises(ValidationError, match="mode"):
        load_settings(path)


def test_output_naming_follows_destriping_variant():
    base = PipelineSettings.model_validate({"distortion": {"mode": "off"}})
    assert variant_output_dir(Path("/d/run-02"), False) == Path("/d/run-02/without_destriping")
    assert variant_output_dir(Path("/d/run-02"), True) == Path("/d/run-02")
    assert run_name_from_output_dir(Path("/d/run-02/without_destriping")) == "run-02"
    assert run_name_from_output_dir(Path("/d/run-02")) == "run-02"
    assert corrected_tiff_path(Path("/o"), Path("/r/a_001.tif"), base).name == (
        "a_001_dynamic_bidiphase.tif"
    )
    striped = with_destriping(base, True)
    assert corrected_tiff_path(Path("/o"), Path("a.tif"), striped).name == (
        "a_dynamic_bidiphase_destriped.tif"
    )


def test_identity_distortion_only_crops(rng):
    chunk = rng.normal(size=(3, 10, 12)).astype(np.float32)
    yy, xx = np.mgrid[0:10, 0:12].astype(float)
    maps = DistortionMaps(yy, xx, (1, 9, 2, 10), (10, 12), 0.0, "identity", 0)
    out = apply_distortion_to_chunk(chunk, maps, order=1)
    assert np.allclose(out, chunk[:, 1:9, 2:10], atol=1e-5)


def test_example_config_is_valid():
    from mini2p_preproc.batch import load_batch_settings
    from mini2p_preproc.config import load_qc_run
    from mini2p_preproc.exports import load_export_settings

    example = Path(__file__).parents[1] / "config" / "pipeline.example.toml"
    settings = load_settings(example)
    assert settings.distortion.mode == "local_batched"
    assert load_batch_settings(example).session_first == 216
    assert load_export_settings(example).expected_runs == 68
    assert load_qc_run(example).qc_registration_frames == 500

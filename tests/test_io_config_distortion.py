from pathlib import Path

import numpy as np
import pytest
import tifffile
from pydantic import ValidationError

from mini2p_preprocessing.config import (
    PipelineSettings,
    corrected_tiff_path,
    load_settings,
    run_name_from_output_dir,
    variant_output_dir,
    with_destriping,
)
from mini2p_preprocessing.distortion import DistortionMaps, apply_distortion_to_chunk
from mini2p_preprocessing.tiff_io import (
    chunk_bounds,
    count_timepoints,
    frame_number,
    read_all_channels,
    unfinished_frame,
    write_interleaved,
)
from tests.conftest import scanimage_tiff, write_interleaved_tiff


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


def test_count_timepoints_drops_an_unfinished_last_frame(tmp_path, caplog):
    # stopped during frame 3: channel 1 was saved, channel 2 never was
    path = scanimage_tiff(tmp_path / "stopped.tif", [1, 1, 2, 2, 3])
    with tifffile.TiffFile(path) as tif, caplog.at_level("WARNING"):
        assert count_timepoints(tif, 2) == 2
        assert count_timepoints(tif, 2, max_timepoints=1) == 1
    assert "unfinished last frame" in caplog.text


def test_count_timepoints_drops_a_partial_frame_of_three_channels(tmp_path):
    path = scanimage_tiff(tmp_path / "stopped3.tif", [1, 1, 1, 2, 2])
    with tifffile.TiffFile(path) as tif:
        assert count_timepoints(tif, 3) == 1


@pytest.mark.parametrize(
    ("frame_numbers", "n_channels"),
    [
        ([1, 1, 2, 3, 3], 2),  # a page of frame 2 missing: every later page has shifted channel
        ([1, 1, 1, 2, 3], 3),  # the leftover pages are from two different frames
        ([1, 1, 2, 2, 2], 2),  # the leftover page repeats the last complete frame
        ([1, 1, 2, 2, 9], 2),  # a gap in the numbering: frames were lost
        ([1, 1, 3, 3, 4], 2),  # a whole frame lost earlier, pages still in step
        ([1, 2, 3, 3, 4, 4, 5], 2),  # pages missing from frames 1 and 2, then a stop
        ([1], 2),  # not even one whole timepoint
    ],
)
def test_count_timepoints_refuses_leftover_pages_that_are_not_one_new_frame(
    tmp_path, frame_numbers, n_channels
):
    path = scanimage_tiff(tmp_path / "bad.tif", frame_numbers)
    with tifffile.TiffFile(path) as tif, pytest.raises(ValueError, match="not divisible"):
        count_timepoints(tif, n_channels)


def test_count_timepoints_counts_frames_of_a_later_file_from_its_first_page(tmp_path, caplog):
    # the second file of a split acquisition starts at frame 1001
    path = scanimage_tiff(tmp_path / "part2.tif", [1001, 1001, 1002, 1002, 1003])
    with tifffile.TiffFile(path) as tif, caplog.at_level("WARNING"):
        assert count_timepoints(tif, 2) == 2
    assert "frame 1003" in caplog.text


def test_frame_number_reads_the_header_of_a_page_served_as_a_tiffframe(tmp_path):
    # classic (non-BigTIFF) ScanImage files: tifffile serves later pages as TiffFrames,
    # which carry no description of their own
    path = scanimage_tiff(tmp_path / "classic.tif", [1, 1, 2, 2, 3, 3], software="SI.LINE")
    with tifffile.TiffFile(path) as tif:
        assert isinstance(tif.pages[3], tifffile.TiffFrame)
        assert frame_number(tif, 3) == 2


def test_unfinished_frame_needs_numbers_in_step_with_page_positions():
    assert unfinished_frame(1, {3}, {4}, 3) == 4
    for first, last_whole, leftover in [(None, {3}, {4}), (1, {3}, {None}), (1, {3}, {5})]:
        with pytest.raises(ValueError, match="frame numbers"):
            unfinished_frame(first, last_whole, leftover, 3)


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
        '[phase]\nmethod = "chunked"\nchunk_frames = 25\n'
        '[phase.manual_first_phase_by_start]\n"200" = -17\n'
        '[distortion]\nmode = "off"\n[registration]\nmotion_alignment_source = "functional"\n'
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
    assert variant_output_dir(Path("/d/run-02"), False, distortion_corrected=True) == Path(
        "/d/run-02/with_distortion_correction/without_destriping"
    )
    assert variant_output_dir(Path("/d/run-02"), True, distortion_corrected=True) == Path(
        "/d/run-02/with_distortion_correction/with_destriping"
    )
    assert run_name_from_output_dir(Path("/d/run-02/without_destriping")) == "run-02"
    assert run_name_from_output_dir(Path("/d/run-02")) == "run-02"
    assert run_name_from_output_dir(Path("/d/run-02/with_destriping")) == "run-02"
    nested = Path("/d/run-02/with_distortion_correction/with_destriping")
    assert run_name_from_output_dir(nested) == "run-02"
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
    from mini2p_preprocessing.batch import load_batch_settings
    from mini2p_preprocessing.config import load_qc_run
    from mini2p_preprocessing.exports import load_export_settings

    example = Path(__file__).parents[1] / "config" / "pipeline.example.toml"
    settings = load_settings(example)
    assert settings.distortion.mode == "local_batched"
    assert load_batch_settings(example).session_first == 216
    assert load_export_settings(example).expected_runs == 68
    assert load_qc_run(example).qc_registration_frames == 500


def test_export_run_subdir_must_be_stated(tmp_path):
    from mini2p_preprocessing.exports import load_export_settings

    path = tmp_path / "p.toml"
    path.write_text(
        "[exports]\nsubject_dir = '/d'\nsession_first = 1\nsession_last = 2\nexpected_runs = 3\n"
    )
    with pytest.raises(ValidationError, match="run_subdir"):
        load_export_settings(path)


def test_export_run_subdir_is_a_variant_folder():
    from mini2p_preprocessing.exports import ExportSettings

    common = {"subject_dir": "/d", "session_first": 1, "session_last": 2, "expected_runs": 3}
    with pytest.raises(ValidationError, match="run_subdir"):
        ExportSettings(**common, run_subdir="with_destripping")

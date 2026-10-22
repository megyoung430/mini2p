"""Movies of each preprocessing stage of a run, on small synthetic run folders."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest
import tifffile

from mini2p_preprocessing import movies
from mini2p_preprocessing.cli import movie_main
from mini2p_preprocessing.config import PipelineSettings, load_settings
from mini2p_preprocessing.contrast import scale_contrast
from mini2p_preprocessing.movies import (
    PANEL_GAP_PX,
    MovieSettings,
    average_windows,
    block_size,
    compose_panels,
    display_limits,
    find_plane_dir,
    frame_label,
    frame_span,
    make_movies,
    open_stage,
    plan_movie,
    rendered_frames,
    stage_title,
    to_display,
    upscale_and_pad,
    window_starts,
)

N_FRAMES, LY, LX, FS = 60, 30, 41, 7.5  # odd width: the MP4 must be padded
CROP = (2, 28, 3, 38)  # the undistorted stage is cropped: 26 x 35
SESSION = "ses-224_date-20260809"
NAMES = ("sub-03_id-MY_71_R", SESSION, "run-01")
# Each stage has its own levels, so a reader can be told apart by its values.
GREEN_OFFSET = {"raw": 1000, "bidiphase": 2000, "undistorted": 3000, "registered": 0}
RED_LEVEL = {"raw": 110, "bidiphase": 120, "undistorted": 130, "registered": 100}


def run_settings(variant: str = "without_destriping", **changes) -> PipelineSettings:
    """What the pipeline records in pipeline_settings.json; channel 1 red, channel 2 green."""
    raw = {
        "distortion": {"mode": "local_batched", "transform_mat": "transform.mat"},
        "destripe": {"enabled": variant == "with_destriping"},
        **changes,
    }
    return PipelineSettings.model_validate(raw)


def ramp(offset: int, shape=(LY, LX), n_frames: int = N_FRAMES) -> np.ndarray:
    """Green: offset + 10 x frame index, the same over the whole frame."""
    values = offset + 10 * np.arange(n_frames, dtype=np.int16)
    return np.broadcast_to(values[:, None, None], (n_frames, *shape)).astype(np.int16)


def flat(level: int, shape=(LY, LX), n_frames: int = N_FRAMES) -> np.ndarray:
    return np.full((n_frames, *shape), level, np.int16)


def write_interleaved_tiff(path: Path, green: np.ndarray, red: np.ndarray, green_page: int = 1):
    """Pages in channel order: by default red t0, green t0, red t1, ..."""
    channels = [red, green] if green_page == 1 else [green, red]
    tifffile.imwrite(path, np.stack(channels, axis=1).reshape(-1, *green.shape[1:]))


def make_run(
    root: Path,
    variant: str = "without_destriping",
    reg_file: Path | None = None,
    tiffs: bool = True,
    manifest: bool = True,
    run: PipelineSettings | None = None,
    raw_frames: int = N_FRAMES,
):
    """A run folder as the batch leaves it: the acquired TIFF (*raw_frames* long), the
    bidiphase and undistorted TIFFs, the registered binaries and pipeline_settings.json."""
    run = run or run_settings(variant)
    green_page = run.channels.functional_channel - 1
    run_dir = root / "cohort-01" / "sub-03_id-MY_71_R" / SESSION / "funcimg" / "run-01"
    variant_dir = run_dir / variant
    plane = variant_dir / "suite2p" / "plane0"
    plane.mkdir(parents=True)
    ramp(GREEN_OFFSET["registered"]).tofile(reg_file or plane / "data.bin")
    flat(RED_LEVEL["registered"]).tofile(plane / "data_chan2.bin")
    ops = {"nframes": N_FRAMES, "Ly": LY, "Lx": LX, "fs": FS}
    if reg_file is not None:
        ops["reg_file"] = str(reg_file)
    np.save(plane / "ops.npy", ops, allow_pickle=True)
    (variant_dir / "pipeline_settings.json").write_text(run.model_dump_json())
    if tiffs:
        raw_dir = root / "rawdata"
        raw_dir.mkdir(exist_ok=True)
        raw = raw_dir / "71rtask_00039.tif"
        write_interleaved_tiff(
            raw,
            ramp(GREEN_OFFSET["raw"], n_frames=raw_frames),
            flat(RED_LEVEL["raw"], n_frames=raw_frames),
            green_page,
        )
        suffix = "dynamic_bidiphase" + ("_destriped" if run.destripe.enabled else "")
        write_interleaved_tiff(
            variant_dir / f"71rtask_00039_{suffix}.tif",
            ramp(GREEN_OFFSET["bidiphase"]),
            flat(RED_LEVEL["bidiphase"]),
            green_page,
        )
        if run.distortion.mode != "off":
            top, bottom, left, right = CROP
            shape = (bottom - top, right - left)
            write_interleaved_tiff(
                variant_dir / f"71rtask_00039_{suffix}_undistorted.tif",
                ramp(GREEN_OFFSET["undistorted"], shape),
                flat(RED_LEVEL["undistorted"], shape),
                green_page,
            )
        if manifest:
            (variant_dir / "source_tiff.json").write_text(json.dumps({"source_tiff": str(raw)}))
    return plane


ONLY_REGISTERED = {"stage_raw": False, "stage_bidiphase": False, "stage_undistorted": False}
ALL_STAGES = {"stage_raw": True, "stage_bidiphase": True, "stage_undistorted": True}


def settings_for(root: Path, **overrides) -> MovieSettings:
    """Run 1 of session 224 of sub-03; the registered stage only unless *overrides* add others."""
    request = {
        "cohort": "cohort-01",
        "subject": "sub-03",
        "session": 224,
        "run": 1,
        "whole_recording": True,
    }
    return MovieSettings(
        derivatives_root=root, **{**request, "label": False, **ONLY_REGISTERED, **overrides}
    )


def decoded(path: Path) -> tuple[dict, np.ndarray]:
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    w, h = meta["size"]
    frames = np.stack([np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in reader])
    return meta, frames


def movie_dir(root: Path, variant: str = "without_destriping") -> Path:
    run_dir = root / "cohort-01" / "sub-03_id-MY_71_R" / SESSION / "funcimg" / "run-01"
    return run_dir / variant / "movies"


# ---------------------------------------------------------------- pure functions


def test_frame_span_includes_the_end_frame():
    assert frame_span(100, True, None, None) == (0, 100)
    assert frame_span(100, False, 10, 90) == (10, 91)
    assert frame_span(100, False, 99, 99) == (99, 100)


@pytest.mark.parametrize(
    ("whole", "start", "end"),
    [(True, 10, None), (False, None, 5), (False, 10, 5), (False, -1, 5), (False, 0, 100)],
)
def test_frame_span_refuses_inconsistent_requests(whole, start, end):
    with pytest.raises(ValueError):
        frame_span(100, whole, start, end)


def test_window_starts_bin_rolling_and_every_frame():
    assert window_starts(10, 22, 5, "bin").tolist() == [10, 15]  # partial last bin dropped
    assert window_starts(10, 16, 5, "rolling").tolist() == [10, 11]
    assert window_starts(10, 13, 1, "bin").tolist() == [10, 11, 12]
    with pytest.raises(ValueError, match="cannot be averaged"):
        window_starts(0, 3, 5, "bin")


@pytest.mark.parametrize("mode", ["bin", "rolling"])
def test_average_windows_is_the_mean_of_each_window(mode):
    frames = np.random.default_rng(0).normal(size=(20, 3, 4)).astype(np.float32)
    first = window_starts(0, 20, 4, mode)
    expected = np.stack([frames[f : f + 4].mean(0) for f in first])
    np.testing.assert_allclose(average_windows(frames, first, 4), expected, rtol=1e-5, atol=1e-6)


def test_display_matches_the_qc_notebook_contrast():
    lo, hi = display_limits(np.arange(101.0), 1, 99)
    assert (lo, hi) == (1.0, 99.0)
    values = np.array([0.0, 1, 50, 99, 200])
    assert to_display(values, 0, 100, 1.0).tolist() == [0, 3, 128, 252, 255]
    # gamma as in contrast.scale_contrast: ((x - black) / (white - black)) ** gamma
    assert to_display(np.array([25.0]), 0, 100, 0.5).tolist() == [128]
    assert display_limits(np.full(10, 3.0), 1, 99.8) == (3.0, 4.0)  # flat image: not empty


@pytest.mark.parametrize("gamma", [0.6, 1.0, 1.7])
def test_display_equals_scale_contrast_on_the_same_pixels(gamma):
    image = np.random.default_rng(1).gamma(2.0, 50.0, size=(40, 50)).astype(np.float32)
    expected, black, white = scale_contrast(image, 1.0, 99.8, gamma)
    assert display_limits(image, 1.0, 99.8) == pytest.approx((black, white))
    shown = to_display(image, black, white, gamma)
    np.testing.assert_array_equal(shown, (expected * 255).round().astype(np.uint8))


def test_black_must_be_below_white():
    with pytest.raises(ValueError, match="below white_percentile"):
        settings_for(Path("."), black_percentile=99.9, white_percentile=99.8)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"stage_registered": False}, "at least one stage"),
        ({"unaveraged_movie": False, "averaged_movie": False}, "unaveraged_movie"),
        ({"whole_recording": False, "start_frame": 3}, "set start_frame and end_frame"),
        ({"whole_recording": False, "start_frame": 9, "end_frame": 3}, "is after end_frame"),
        ({"session": -1}, "greater than or equal to 0"),
        ({"subject": ""}, "at least 1 character"),
    ],
)
def test_settings_that_ask_for_nothing_or_contradict_are_refused(changes, message):
    """Each case would otherwise make no movie, or not the movie asked for."""
    with pytest.raises(ValueError, match=message):
        settings_for(Path("."), **changes)


def test_settings_list_the_stages_versions_and_frames_asked_for(caplog):
    settings = settings_for(
        Path("."), stage_raw=True, averaged_movie=False, start_frame=3, end_frame=9
    )
    assert settings.stages == ("raw", "registered")
    assert settings.versions == ("unaveraged",)
    assert settings.frames_requested == (True, None, None)  # whole recording: frames ignored
    assert "start_frame 3 and end_frame 9 are ignored" in caplog.text
    ranged = settings.model_copy(update={"whole_recording": False})
    assert ranged.frames_requested == (False, 3, 9)


def test_panels_are_grey_functional_left_structural_right():
    green = np.full((2, 3, 4), 200, np.uint8)
    red = np.full((2, 3, 4), 50, np.uint8)
    rgb = compose_panels({"green": green, "red": red})
    assert rgb.shape == (2, 3, 4 + PANEL_GAP_PX + 4, 3)
    assert rgb[0, 0, 0].tolist() == [200, 200, 200]
    assert rgb[0, 0, 4].tolist() == [0, 0, 0]  # the gap
    assert rgb[0, 0, -1].tolist() == [50, 50, 50]
    single = compose_panels({"red": red})
    assert single.shape == (2, 3, 4, 3) and single[0, 0, 0].tolist() == [50, 50, 50]


def test_upscale_then_pad_to_even():
    rgb = np.zeros((4, 3, 5, 3), np.uint8)
    assert upscale_and_pad(rgb, 1).shape == (4, 4, 6, 3)
    assert upscale_and_pad(rgb, 3).shape == (4, 10, 16, 3)


def test_stage_titles_say_what_was_done():
    assert stage_title("raw", run_settings()) == "raw"
    assert stage_title("undistorted", run_settings()) == "bidiphase + distortion"
    assert stage_title("registered", run_settings("with_destriping")) == (
        "bidiphase + destriping + distortion + registration"
    )
    no_distortion = run_settings(distortion={"mode": "off"})
    assert stage_title("registered", no_distortion) == "bidiphase + registration"
    assert stage_title("registered", None) == "registered"


def test_plan_names_the_stage_and_the_frames_actually_shown():
    settings = settings_for(Path("."), average_frames=5, speed=4.0, variant="with_destriping")
    plan = plan_movie("averaged", "undistorted", (10, 151), FS, settings, NAMES)
    assert plan.name == (
        f"sub-03_id-MY_71_R_{SESSION}_run-01_with_destriping_undistorted_side_by_side_"
        "frames-10-149_bin5_4x.mp4"
    )
    assert (plan.first_frame, plan.last_frame, len(plan.starts)) == (10, 149, 28)
    assert plan.fps == pytest.approx(FS / 5 * 4)


def test_plan_unaveraged_is_never_smoothed_and_smoothing_is_named():
    settings = settings_for(Path("."), spatial_sigma_px=1.5, average_mode="rolling")
    every = plan_movie("unaveraged", "raw", (0, 20), FS, settings, NAMES)
    rolling = plan_movie("averaged", "raw", (0, 20), FS, settings, NAMES)
    assert every.sigma_px == 0 and "sigma" not in every.name
    assert every.fps == pytest.approx(FS * 4)
    assert rolling.sigma_px == 1.5 and "_rolling5_sigma1.5_4x" in rolling.name
    assert (rolling.last_frame, rolling.fps) == (19, pytest.approx(FS * 4))


def test_plan_refuses_a_playback_rate_too_slow_to_encode():
    settings = settings_for(Path("."), average_frames=50, speed=1)
    with pytest.raises(ValueError, match="raise speed"):
        plan_movie("averaged", "registered", (0, 60), FS, settings, NAMES)


def test_frame_label_gives_the_window_centre():
    assert frame_label(15, 1, 7.5, 4) == "    2.0 s  frame 15  4x"
    assert frame_label(15, 5, 7.5, 0.5) == "    2.3 s  frames 15-19  0.5x"


def test_blocks_hold_a_bounded_number_of_acquired_frames():
    assert block_size(1, "bin") == 128
    assert block_size(50, "bin") == 2
    assert block_size(256, "bin") == 1
    assert block_size(50, "rolling") == 79
    for n in (100, 200, 256):  # each acquired frame read at most twice
        assert (block_size(n, "rolling") + n - 1) / block_size(n, "rolling") < 2


# ---------------------------------------------------------------- finding and reading


@pytest.mark.parametrize("subject", ["sub-03", "MY_71_R", "sub-03_id-MY_71_R"])
def test_find_plane_dir_by_subject_name_or_id(tmp_path, subject):
    plane = make_run(tmp_path, tiffs=False)
    assert find_plane_dir(tmp_path, "cohort-01", subject, 224, 1, "without_destriping") == plane


def test_find_plane_dir_lists_what_is_there(tmp_path):
    make_run(tmp_path, tiffs=False)
    with pytest.raises(FileNotFoundError, match="ses-224_date-20260809"):
        find_plane_dir(tmp_path, "cohort-01", "sub-03", 225, 1, "without_destriping")
    with pytest.raises(FileNotFoundError, match="run-02"):
        find_plane_dir(tmp_path, "cohort-01", "sub-03", 224, 2, "without_destriping")
    with pytest.raises(FileNotFoundError, match="with_destriping"):
        find_plane_dir(tmp_path, "cohort-01", "sub-03", 224, 1, "with_destriping")
    (tmp_path / "cohort-01" / "sub-03_id-other").mkdir()
    with pytest.raises(FileNotFoundError, match="2 folders match"):
        find_plane_dir(tmp_path, "cohort-01", "sub-03", 224, 1, "without_destriping")


@pytest.mark.parametrize("variant", ["without_destriping", "with_destriping"])
@pytest.mark.parametrize("stage", ["raw", "bidiphase", "undistorted", "registered"])
def test_each_stage_reads_its_own_file_and_channel(tmp_path, stage, variant):
    plane = make_run(tmp_path, variant=variant)
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    with ExitStack() as stack:
        sources = open_stage(stage, plane, ops, "side_by_side", run_settings(variant), None, stack)
        assert list(sources) == ["green", "red"]
        green, red = sources["green"].read(4, 7), sources["red"].read(4, 7)
    assert green.dtype == np.float32 and green.shape[0] == 3
    np.testing.assert_array_equal(green[:, 0, 0], GREEN_OFFSET[stage] + 10 * np.arange(4, 7))
    assert np.all(red == RED_LEVEL[stage])
    expected_shape = (CROP[1] - CROP[0], CROP[3] - CROP[2]) if stage == "undistorted" else (LY, LX)
    assert sources["green"].frame_shape == expected_shape


@pytest.mark.parametrize("stage", ["raw", "bidiphase", "undistorted"])
def test_channel_order_comes_from_the_run_settings(tmp_path, stage):
    swapped = run_settings(channels={"structural_channel": 2, "functional_channel": 1})
    plane = make_run(tmp_path, run=swapped)
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    with ExitStack() as stack:
        sources = open_stage(stage, plane, ops, "side_by_side", swapped, None, stack)
        assert sources["green"].read(2, 3)[0, 0, 0] == GREEN_OFFSET[stage] + 20
        assert sources["red"].read(2, 3)[0, 0, 0] == RED_LEVEL[stage]


def test_raw_tiff_longer_than_the_processed_run_is_cut_to_it(tmp_path):
    """With max_timepoints the pipeline processed only the first frames of the TIFF."""
    plane = make_run(tmp_path, run=run_settings(max_timepoints=N_FRAMES), raw_frames=N_FRAMES + 25)
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    with ExitStack() as stack:
        raw = open_stage("raw", plane, ops, "green", run_settings(), None, stack)["green"]
        assert raw.n_frames == N_FRAMES
        assert raw.read(N_FRAMES - 1, N_FRAMES)[0, 0, 0] == GREEN_OFFSET["raw"] + 10 * 59


def test_a_tiff_shorter_than_the_run_fails_when_read(tmp_path):
    plane = make_run(tmp_path, raw_frames=N_FRAMES - 10)
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    with ExitStack() as stack:
        raw = open_stage("raw", plane, ops, "green", run_settings(), None, stack)["green"]
        with pytest.raises(ValueError, match="fewer than 60 frames"):
            raw.read(40, 60)


def test_raw_tiff_needs_the_manifest_or_a_path(tmp_path):
    plane = make_run(tmp_path, manifest=False)
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    with ExitStack() as stack, pytest.raises(FileNotFoundError, match="raw_tiff"):
        open_stage("raw", plane, ops, "green", run_settings(), None, stack)
    raw = tmp_path / "rawdata" / "71rtask_00039.tif"
    with ExitStack() as stack:
        sources = open_stage("bidiphase", plane, ops, "green", run_settings(), raw, stack)
        assert sources["green"].read(0, 1)[0, 0, 0] == GREEN_OFFSET["bidiphase"]


def test_missing_stage_file_is_named(tmp_path):
    plane = make_run(tmp_path)
    next(plane.parents[1].glob("*_undistorted.tif")).unlink()
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    with ExitStack() as stack, pytest.raises(FileNotFoundError, match="No undistorted TIFF"):
        open_stage("undistorted", plane, ops, "green", run_settings(), None, stack)


def test_binary_on_fast_disk_is_found_through_ops(tmp_path):
    fast = tmp_path / "fast"
    fast.mkdir()
    plane = make_run(tmp_path / "share", reg_file=fast / "data.bin", tiffs=False)
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    with ExitStack() as stack:
        sources = open_stage("registered", plane, ops, "green", None, None, stack)
    assert sources["green"].path == fast / "data.bin"


def test_incomplete_binary_is_refused(tmp_path):
    plane = make_run(tmp_path, tiffs=False)
    with (plane / "data.bin").open("r+b") as handle:
        handle.truncate(100)
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    with ExitStack() as stack, pytest.raises(ValueError, match="registration complete"):
        open_stage("registered", plane, ops, "side_by_side", None, None, stack)


@pytest.mark.parametrize("mode", ["bin", "rolling"])
def test_reading_in_blocks_gives_the_same_frames(tmp_path, mode):
    plane = make_run(tmp_path)
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    settings = settings_for(tmp_path, average_frames=4, average_mode=mode, spatial_sigma_px=1.0)
    plan = plan_movie("averaged", "raw", (3, 50), FS, settings, NAMES)
    limits = {"green": (1000.0, 1600.0), "red": (0.0, 220.0)}
    with ExitStack() as stack:
        sources = open_stage("raw", plane, ops, "side_by_side", run_settings(), None, stack)
        small = np.stack(list(rendered_frames(sources, plan, limits, settings, FS, block=3)))
        large = np.stack(list(rendered_frames(sources, plan, limits, settings, FS, block=500)))
        labelled = settings.model_copy(update={"label": True})
        drawn = next(rendered_frames(sources, plan, limits, labelled, FS, "raw", block=500))
    np.testing.assert_array_equal(small, large)
    expected_green = to_display(np.array([s * 10 + 15.0 for s in plan.starts]), 0, 600, 1.0)
    np.testing.assert_array_equal(small[:, 5, 5, 1], expected_green)  # left panel, green
    right = small[:, 5, LX + PANEL_GAP_PX + 5]
    np.testing.assert_array_equal(right, [[128, 128, 128]] * len(small))  # structural, grey
    assert not np.array_equal(drawn, small[0])  # the labels are drawn


# ---------------------------------------------------------------- end to end


def test_unaveraged_and_binned_movies_of_a_frame_range(tmp_path):
    make_run(tmp_path, tiffs=False)
    settings = settings_for(
        tmp_path, average_frames=5, speed=2.0, whole_recording=False, start_frame=10, end_frame=49
    )
    written = make_movies(settings)
    out = movie_dir(tmp_path)
    assert [p.parent for p in written] == [out, out]
    assert [p.name.split("_")[-2] for p in written] == ["unaveraged", "bin5"]
    assert written[0].name.startswith(f"sub-03_id-MY_71_R_{SESSION}_run-01_")
    assert "_registered_side_by_side_" in written[0].name
    assert not list(out.glob("*.partial.mp4"))

    meta, frames = decoded(written[0])
    assert frames.shape == (40, LY, 2 * LX + PANEL_GAP_PX, 3)  # frames 10..49
    assert meta["fps"] == pytest.approx(FS * 2, rel=1e-3)
    functional = frames[:, :, :LX].astype(int)
    assert np.abs(functional[..., 0] - functional[..., 1]).max() < 8  # grey: R = G = B
    assert np.abs(functional[..., 2] - functional[..., 1]).max() < 8
    level = functional[..., 1].mean(axis=(1, 2))
    assert np.all(np.diff(level) > -2) and level[-1] - level[0] > 150  # the ramp survives

    meta, binned = decoded(written[1])
    assert len(binned) == 8 and meta["fps"] == pytest.approx(FS / 5 * 2, rel=1e-3)


def test_every_stage_gives_its_movies(tmp_path):
    make_run(tmp_path, variant="with_destriping")
    settings = settings_for(
        tmp_path,
        subject="MY_71_R",
        variant="with_destriping",
        average_mode="rolling",
        average_frames=3,
        label=True,
        upscale=2,
        **ALL_STAGES,
    )
    written = make_movies(settings)
    assert [p.name.split("_with_destriping_")[1].split("_")[0] for p in written[::2]] == list(
        movies.STAGES
    )
    for path in written:
        frames = decoded(path)[1]
        width = CROP[3] - CROP[2] if "_undistorted_" in path.name else LX
        height = CROP[1] - CROP[0] if "_undistorted_" in path.name else LY
        assert frames.shape[1:3] == (2 * height, 2 * (2 * width + PANEL_GAP_PX))
        version = path.name.split("_")[-2]
        assert len(frames) == (N_FRAMES if version == "unaveraged" else N_FRAMES - 2)


def test_a_missing_stage_stops_before_anything_is_written(tmp_path):
    plane = make_run(tmp_path)
    next(plane.parents[1].glob("*_undistorted.tif")).unlink()
    with pytest.raises(FileNotFoundError, match="undistorted"):
        make_movies(settings_for(tmp_path, **ALL_STAGES))
    assert not movie_dir(tmp_path).exists()


def test_tiff_stages_need_the_runs_settings_record(tmp_path):
    plane = make_run(tmp_path)
    (plane.parents[1] / "pipeline_settings.json").unlink()
    with pytest.raises(FileNotFoundError, match="pipeline_settings.json"):
        make_movies(settings_for(tmp_path, stage_raw=True))
    make_movies(settings_for(tmp_path, averaged_movie=False))


def test_a_run_without_distortion_refuses_the_undistorted_stage(tmp_path):
    make_run(tmp_path, run=run_settings(distortion={"mode": "off"}))
    with pytest.raises(ValueError, match="set stage_undistorted = false"):
        make_movies(settings_for(tmp_path, **ALL_STAGES))
    written = make_movies(
        settings_for(tmp_path, **{**ALL_STAGES, "stage_undistorted": False}, averaged_movie=False)
    )
    assert [p.name.split("_without_destriping_")[1].split("_")[0] for p in written] == [
        "raw",
        "bidiphase",
        "registered",
    ]


def test_existing_movies_are_kept_unless_overwrite(tmp_path):
    make_run(tmp_path, tiffs=False)
    settings = settings_for(tmp_path, averaged_movie=False)
    (path,) = make_movies(settings)
    stamp = path.stat().st_mtime_ns
    with pytest.raises(FileExistsError, match="overwrite = true"):
        make_movies(settings_for(tmp_path))
    assert path.stat().st_mtime_ns == stamp
    make_movies(settings.model_copy(update={"overwrite": True}))


def test_failed_write_leaves_no_movie(tmp_path, monkeypatch):
    make_run(tmp_path, tiffs=False)

    def broken(*args, **kwargs):
        yield np.zeros((LY, 2 * LX + PANEL_GAP_PX, 3), np.uint8)
        raise RuntimeError("disk full")

    monkeypatch.setattr(movies, "rendered_frames", broken)
    with pytest.raises(RuntimeError):
        make_movies(settings_for(tmp_path, averaged_movie=False))
    assert not list(movie_dir(tmp_path).glob("*.mp4"))  # neither the movie nor its partial


def test_incomplete_encode_is_not_kept(tmp_path, monkeypatch):
    make_run(tmp_path, tiffs=False)
    monkeypatch.setattr(movies, "_encoded_frames", lambda path: 3)
    with pytest.raises(OSError, match="the encode failed"):
        make_movies(settings_for(tmp_path, averaged_movie=False))
    assert not list(movie_dir(tmp_path).glob("*.mp4"))


def write_config(tmp_path: Path, extra: str = "") -> Path:
    """A config whose [movies] section says everything: run, frames, stages, versions."""
    config = tmp_path / "pipeline.toml"
    config.write_text(
        '[distortion]\nmode = "off"\n\n'
        "[movies]\n"
        'cohort = "cohort-01"\nsubject = "sub-03"\nsession = 224\nrun = 1\n'
        "whole_recording = false\nstart_frame = 0\nend_frame = 19\n"
        "stage_raw = false\nstage_bidiphase = false\nstage_undistorted = false\n"
        "stage_registered = true\nunaveraged_movie = true\naveraged_movie = true\n"
        f'derivatives_root = "{tmp_path}"\nlabel = false\nspeed = 3.0\n{extra}'
    )
    return config


def test_the_config_alone_says_which_movies_to_make(tmp_path):
    make_run(tmp_path, tiffs=False)
    config = write_config(tmp_path)
    load_settings(config)  # the pipeline loader skips [movies]
    result = movie_main(["--config", str(config)])
    assert result is None  # the console script passes this to sys.exit: None is exit code 0
    names = sorted(p.name for p in movie_dir(tmp_path).glob("*.mp4"))
    assert [n.split("_registered_")[1] for n in names] == [
        "side_by_side_frames-0-19_bin5_3x.mp4",
        "side_by_side_frames-0-19_unaveraged_3x.mp4",
    ]


def test_command_line_options_override_the_config(tmp_path):
    make_run(tmp_path)
    config = write_config(tmp_path)
    movie_main(
        ["--config", str(config), "--no-unaveraged-movie", "--average-frames", "4",
         "--channel", "red", "--stage-raw", "--end-frame", "11", "--gamma", "0.8"]
    )  # fmt: skip
    names = sorted(p.name for p in movie_dir(tmp_path).glob("*.mp4"))
    assert [n.split("_without_destriping_")[1] for n in names] == [
        "raw_red_frames-0-11_bin4_3x.mp4",
        "registered_red_frames-0-11_bin4_3x.mp4",
    ]
    with pytest.raises(SystemExit):  # a bad setting is a usage error, not a traceback
        movie_main(["--config", str(config), "--average-frames", "1"])


def test_cli_reports_a_frame_past_the_end_as_a_usage_error(tmp_path, capsys):
    make_run(tmp_path, tiffs=False)
    config = write_config(tmp_path)
    with pytest.raises(SystemExit):
        movie_main(["--config", str(config), "--end-frame", "999"])
    assert "end_frame" in capsys.readouterr().err


def test_registered_movies_survive_an_outdated_settings_record(tmp_path):
    """A record from an older version of the settings only loses the registered stage's title."""
    plane = make_run(tmp_path, tiffs=False)
    record = plane.parents[1] / "pipeline_settings.json"
    old = json.loads(record.read_text())
    old["phase"]["a_removed_setting"] = 1
    record.write_text(json.dumps(old))
    (path,) = make_movies(settings_for(tmp_path, averaged_movie=False))
    assert path.is_file()
    with pytest.raises(ValueError, match="a_removed_setting"):
        make_movies(settings_for(tmp_path, stage_raw=True, overwrite=True))


def test_whole_recording_is_never_chosen_by_omission():
    with pytest.raises(ValueError, match="whole_recording"):
        MovieSettings(derivatives_root=Path("."), cohort="c", subject="s", session=1, run=1)
    settings = MovieSettings(
        derivatives_root=Path("."), cohort="c", subject="s", session=1, run=1, whole_recording=True
    )
    assert settings.stages == ("registered",)  # the cheap stage unless others are asked for


def test_plan_refuses_an_unknown_version():
    with pytest.raises(ValueError, match="'unaveraged' or 'averaged'"):
        plan_movie("raw", "raw", (0, 20), FS, settings_for(Path(".")), NAMES)


def test_every_setting_has_a_command_line_option():
    from mini2p_preprocessing.cli import MOVIE_HELP

    assert set(MOVIE_HELP) == set(MovieSettings.model_fields)


def test_frames_on_the_command_line_override_a_whole_recording_config(tmp_path):
    make_run(tmp_path, tiffs=False)
    config = write_config(tmp_path, "")
    text = config.read_text().replace("whole_recording = false", "whole_recording = true")
    config.write_text(text.replace("\naveraged_movie = true", "\naveraged_movie = false"))
    movie_main(["--config", str(config), "--start-frame", "2", "--end-frame", "6"])
    names = [p.name for p in movie_dir(tmp_path).glob("*.mp4")]
    assert len(names) == 1 and "_frames-2-6_unaveraged_" in names[0]
    with pytest.raises(SystemExit):
        movie_main(["--config", str(config), "--whole-recording", "--end-frame", "6"])


def test_a_switch_set_in_the_config_stays_on_without_its_option(tmp_path):
    make_run(tmp_path, tiffs=False)
    config = write_config(tmp_path, "overwrite = true\n")
    movie_main(["--config", str(config)])
    movie_main(["--config", str(config)])  # overwrite = true from the config: no FileExistsError
    assert len(list(movie_dir(tmp_path).glob("*.mp4"))) == 2

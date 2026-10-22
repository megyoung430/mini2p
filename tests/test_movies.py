"""Movies of a registered run, on a small synthetic plane folder."""

from __future__ import annotations

from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest

from mini2p_preprocessing import movies
from mini2p_preprocessing.cli import movie_main
from mini2p_preprocessing.config import load_settings
from mini2p_preprocessing.movies import (
    MovieSettings,
    average_windows,
    block_size,
    compose_rgb,
    contrast_limits,
    find_plane_dir,
    frame_label,
    frame_span,
    make_movies,
    open_channels,
    plan_movie,
    rendered_frames,
    to_uint8,
    upscale_and_pad,
    window_starts,
)

N_FRAMES, LY, LX, FS = 60, 30, 41, 7.5  # odd width: the MP4 must be padded
SESSION = "ses-224_date-20260809"


def make_plane(root: Path, variant: str = "without_destriping", reg_file: Path | None = None):
    """A registered plane: green brightens frame by frame, red is constant at 100."""
    plane = root / "cohort-01" / "sub-03_id-MY_71_R" / SESSION / "funcimg" / "run-01"
    plane = plane / variant / "suite2p" / "plane0"
    plane.mkdir(parents=True)
    ramp = np.arange(N_FRAMES, dtype=np.int16)[:, None, None] * 10
    green = np.broadcast_to(ramp, (N_FRAMES, LY, LX)).astype(np.int16)
    red = np.full((N_FRAMES, LY, LX), 100, np.int16)
    green_path = reg_file or plane / "data.bin"
    green.tofile(green_path)
    red.tofile(plane / "data_chan2.bin")
    ops = {"nframes": N_FRAMES, "Ly": LY, "Lx": LX, "fs": FS}
    if reg_file is not None:
        ops["reg_file"] = str(reg_file)
    np.save(plane / "ops.npy", ops, allow_pickle=True)
    return plane, green, red


def settings_for(root: Path, **overrides) -> MovieSettings:
    return MovieSettings(derivatives_root=root, label=False, **overrides)


def decoded(path: Path) -> tuple[dict, np.ndarray]:
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    w, h = meta["size"]
    frames = np.stack([np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in reader])
    return meta, frames


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


def test_window_starts_bin_rolling_and_raw():
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


def test_contrast_and_uint8_scaling():
    lo, hi = contrast_limits(np.arange(101.0), 0, 100)
    assert (lo, hi) == (0.0, 100.0)
    assert to_uint8(np.array([-5.0, 0, 50, 100, 200]), lo, hi).tolist() == [0, 0, 128, 255, 255]
    assert contrast_limits(np.full(10, 3.0), 1, 99) == (3.0, 4.0)  # flat image: not empty


def test_merge_puts_green_in_green_and_red_in_magenta():
    g, r = np.full((2, 2), 200, np.uint8), np.full((2, 2), 50, np.uint8)
    assert compose_rgb(g, r)[0, 0].tolist() == [50, 200, 50]
    assert compose_rgb(g, None)[0, 0].tolist() == [200, 200, 200]


def test_upscale_then_pad_to_even():
    rgb = np.zeros((4, 3, 5, 3), np.uint8)
    assert upscale_and_pad(rgb, 1).shape == (4, 4, 6, 3)
    assert upscale_and_pad(rgb, 3).shape == (4, 10, 16, 3)


NAMES = ("sub-03_id-MY_71_R", SESSION, "run-01")


def test_plan_names_the_frames_actually_shown():
    settings = settings_for(Path("."), average_frames=5, speed=4.0, variant="with_destriping")
    plan = plan_movie("averaged", (10, 151), FS, settings, NAMES)  # 141 frames: 28 bins of 5
    assert plan.name == (
        f"sub-03_id-MY_71_R_{SESSION}_run-01_with_destriping_green_frames-10-149_bin5_4x.mp4"
    )
    assert (plan.first_frame, plan.last_frame, len(plan.starts)) == (10, 149, 28)
    assert plan.fps == pytest.approx(FS / 5 * 4)


def test_plan_raw_is_never_smoothed_and_smoothing_is_named():
    settings = settings_for(Path("."), spatial_sigma_px=1.5, average_mode="rolling")
    raw = plan_movie("raw", (0, 20), FS, settings, NAMES)
    rolling = plan_movie("averaged", (0, 20), FS, settings, NAMES)
    assert raw.sigma_px == 0 and "sigma" not in raw.name and raw.fps == pytest.approx(FS * 4)
    assert rolling.sigma_px == 1.5 and "_rolling5_sigma1.5_4x" in rolling.name
    assert (rolling.last_frame, rolling.fps) == (19, pytest.approx(FS * 4))


def test_plan_refuses_a_playback_rate_too_slow_to_encode():
    with pytest.raises(ValueError, match="raise speed"):
        plan_movie(
            "averaged", (0, 60), FS, settings_for(Path("."), average_frames=50, speed=1), NAMES
        )


def test_frame_label_gives_the_window_centre():
    assert frame_label(15, 1, 7.5, 4) == "    2.0 s  frame 15  4x"
    assert frame_label(15, 5, 7.5, 0.5) == "    2.3 s  frames 15-19  0.5x"


def test_blocks_hold_a_bounded_number_of_acquired_frames():
    assert block_size(1, "bin") == 256
    assert block_size(50, "bin") == 5
    assert block_size(256, "bin") == 1
    assert block_size(50, "rolling") == 207
    for n in (100, 200, 256):  # each acquired frame read at most twice
        assert (block_size(n, "rolling") + n - 1) / block_size(n, "rolling") < 2


# ---------------------------------------------------------------- finding and reading


@pytest.mark.parametrize("subject", ["sub-03", "MY_71_R", "sub-03_id-MY_71_R"])
def test_find_plane_dir_by_subject_name_or_id(tmp_path, subject):
    plane, _, _ = make_plane(tmp_path)
    assert find_plane_dir(tmp_path, "cohort-01", subject, 224, 1, "without_destriping") == plane


def test_find_plane_dir_lists_what_is_there(tmp_path):
    make_plane(tmp_path)
    with pytest.raises(FileNotFoundError, match="ses-224_date-20260809"):
        find_plane_dir(tmp_path, "cohort-01", "sub-03", 225, 1, "without_destriping")
    with pytest.raises(FileNotFoundError, match="run-02"):
        find_plane_dir(tmp_path, "cohort-01", "sub-03", 224, 2, "without_destriping")
    with pytest.raises(FileNotFoundError, match="with_destriping"):
        find_plane_dir(tmp_path, "cohort-01", "sub-03", 224, 1, "with_destriping")
    (tmp_path / "cohort-01" / "sub-03_id-other").mkdir()
    with pytest.raises(FileNotFoundError, match="2 folders match"):
        find_plane_dir(tmp_path, "cohort-01", "sub-03", 224, 1, "without_destriping")


def test_binary_on_fast_disk_is_found_through_ops(tmp_path):
    fast = tmp_path / "fast"
    fast.mkdir()
    plane, green, _ = make_plane(tmp_path / "share", reg_file=fast / "data.bin")
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    np.testing.assert_array_equal(open_channels(plane, ops, "green")["green"], green)


def test_incomplete_binary_is_refused(tmp_path):
    plane, _, _ = make_plane(tmp_path)
    with (plane / "data.bin").open("r+b") as handle:
        handle.truncate(100)
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    with pytest.raises(ValueError, match="registration complete"):
        open_channels(plane, ops, "merge")


@pytest.mark.parametrize("mode", ["bin", "rolling"])
def test_reading_in_blocks_gives_the_same_frames(tmp_path, mode):
    plane, _, _ = make_plane(tmp_path)
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    sources = open_channels(plane, ops, "merge")
    settings = settings_for(tmp_path, spatial_sigma_px=1.0)
    plan = plan_movie(
        "averaged",
        (3, 50),
        FS,
        settings_for(tmp_path, average_frames=4, average_mode=mode, spatial_sigma_px=1.0),
        NAMES,
    )
    starts = plan.starts
    limits = {"green": (0.0, 600.0), "red": (0.0, 200.0)}
    small = np.stack(list(rendered_frames(sources, plan, limits, settings, FS, block=3)))
    large = np.stack(list(rendered_frames(sources, plan, limits, settings, FS, block=500)))
    np.testing.assert_array_equal(small, large)
    expected_green = to_uint8(np.array([s * 10 + 15.0 for s in starts]), 0, 600)  # mean of 4
    np.testing.assert_array_equal(small[:, 5, 5, 1], expected_green)


# ---------------------------------------------------------------- end to end


def test_raw_and_binned_movies_of_a_frame_range(tmp_path):
    make_plane(tmp_path)
    written = make_movies(
        settings_for(tmp_path, average_frames=5, speed=2.0),
        "cohort-01",
        "sub-03",
        224,
        1,
        whole_recording=False,
        start_frame=10,
        end_frame=49,
    )
    out = tmp_path / "cohort-01" / "sub-03_id-MY_71_R" / SESSION / "funcimg" / "run-01"
    out = out / "without_destriping" / "movies"
    assert [p.parent for p in written] == [out, out]
    assert [p.name.split("_")[-2] for p in written] == ["raw", "bin5"]
    assert written[0].name.startswith(f"sub-03_id-MY_71_R_{SESSION}_run-01_")
    assert not list(out.glob("*.partial.mp4"))

    meta, raw = decoded(written[0])
    assert raw.shape == (40, LY, LX + 1, 3)  # frames 10..49, width padded to even
    assert meta["fps"] == pytest.approx(FS * 2, rel=1e-3)
    brightness = raw[:, :LY, :LX].mean(axis=(1, 2, 3))
    assert np.all(np.diff(brightness) > -2)  # the ramp survives encoding
    assert brightness[-1] - brightness[0] > 150

    meta, binned = decoded(written[1])
    assert len(binned) == 8 and meta["fps"] == pytest.approx(FS / 5 * 2, rel=1e-3)


def test_whole_recording_rolling_merge_with_label(tmp_path):
    make_plane(tmp_path, variant="with_destriping")
    settings = MovieSettings(
        derivatives_root=tmp_path,
        variant="with_destriping",
        channel="merge",
        average_mode="rolling",
        average_frames=3,
        label=True,
        upscale=2,
    )
    raw, rolling = make_movies(settings, "cohort-01", "MY_71_R", 224, 1)
    assert decoded(raw)[1].shape == (N_FRAMES, 2 * LY, 2 * LX, 3)
    assert len(decoded(rolling)[1]) == N_FRAMES - 2
    assert "merge" in raw.name and "rolling3" in rolling.name
    assert decoded(rolling)[0]["fps"] == pytest.approx(FS * 4, rel=1e-3)


def test_merge_colours_survive_encoding(tmp_path):
    make_plane(tmp_path)
    settings = settings_for(tmp_path, channel="merge", low_percentile=0, high_percentile=100)
    (raw,) = make_movies(settings, "cohort-01", "sub-03", 224, 1, versions=("raw",))
    r, g, b = decoded(raw)[1][-1, 10, 10].astype(int)  # brightest green; red is flat
    assert g > 200 and abs(r - b) < 10


def test_existing_movies_are_kept_unless_overwrite(tmp_path):
    make_plane(tmp_path)
    settings = settings_for(tmp_path)
    (path,) = make_movies(settings, "cohort-01", "sub-03", 224, 1, versions=("raw",))
    stamp = path.stat().st_mtime_ns
    with pytest.raises(FileExistsError):
        make_movies(settings, "cohort-01", "sub-03", 224, 1)
    assert path.stat().st_mtime_ns == stamp
    make_movies(settings, "cohort-01", "sub-03", 224, 1, versions=("raw",), overwrite=True)


def test_failed_write_leaves_no_movie(tmp_path, monkeypatch):
    make_plane(tmp_path)

    def broken(*args, **kwargs):
        yield np.zeros((LY, LX + 1, 3), np.uint8)
        raise RuntimeError("disk full")

    monkeypatch.setattr(movies, "rendered_frames", broken)
    with pytest.raises(RuntimeError):
        make_movies(settings_for(tmp_path), "cohort-01", "sub-03", 224, 1, versions=("raw",))
    out = next(tmp_path.rglob("movies"))
    assert not list(out.glob("*.mp4"))  # neither the movie nor its partial file


def test_incomplete_encode_is_not_kept(tmp_path, monkeypatch):
    make_plane(tmp_path)
    monkeypatch.setattr(movies, "_encoded_frames", lambda path: 3)
    with pytest.raises(OSError, match="the encode failed"):
        make_movies(settings_for(tmp_path), "cohort-01", "sub-03", 224, 1, versions=("raw",))
    assert not list(next(tmp_path.rglob("movies")).glob("*.mp4"))


def test_cli_reads_movies_section_and_pipeline_ignores_it(tmp_path):
    make_plane(tmp_path)
    config = tmp_path / "pipeline.toml"
    config.write_text(
        '[distortion]\nmode = "off"\n\n'
        f'[movies]\nderivatives_root = "{tmp_path}"\nlabel = false\nspeed = 3.0\n'
    )
    load_settings(config)  # the pipeline loader skips [movies]
    result = movie_main(
        [
            "--config",
            str(config),
            "--cohort",
            "cohort-01",
            "--subject",
            "sub-03",
            "--session",
            "224",
            "--run",
            "1",
            "--start-frame",
            "0",
            "--end-frame",
            "19",
            "--only",
            "averaged",
            "--average-frames",
            "4",
            "--channel",
            "red",
        ]
    )
    assert result is None  # the console script passes this to sys.exit: None is exit code 0
    (written,) = next(tmp_path.rglob("movies")).glob("*.mp4")
    assert written.name.endswith("_red_frames-0-19_bin4_3x.mp4")
    assert len(decoded(written)[1]) == 5
    with pytest.raises(SystemExit):
        movie_main(
            [
                "--config",
                str(config),
                "--cohort",
                "c",
                "--subject",
                "s",
                "--session",
                "1",
                "--run",
                "1",
                "--start-frame",
                "3",
            ]
        )


def test_cli_reports_a_frame_past_the_end_as_a_usage_error(tmp_path, capsys):
    make_plane(tmp_path)
    config = tmp_path / "pipeline.toml"
    config.write_text(f'[movies]\nderivatives_root = "{tmp_path}"\n')
    with pytest.raises(SystemExit):
        movie_main(
            [
                "--config",
                str(config),
                "--cohort",
                "cohort-01",
                "--subject",
                "sub-03",
                "--session",
                "224",
                "--run",
                "1",
                "--start-frame",
                "0",
                "--end-frame",
                "999",
            ]
        )
    assert "end_frame" in capsys.readouterr().err

"""Movies with the behavior video beside the imaging, paired by pyControl time."""

from __future__ import annotations

from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest

from mini2p_preprocessing.behavior_video import (
    FRAME_TIMES_FILE,
    VIDEO_FRAME_TIMES_FILE,
    VideoFrames,
    behavior_panel,
    behavior_width,
    load_behavior_sync,
    movie_frame_times,
    video_frame_for,
)
from mini2p_preprocessing.movies import PANEL_GAP_PX, make_movies
from tests.test_movies import FS, LX, LY, N_FRAMES, decoded, make_run, movie_dir, settings_for

VIDEO_HZ, VIDEO_W, VIDEO_H, VIDEO_FRAMES = 10.0, 64, 48, 100
IMAGING_START_S, VIDEO_START_S = 100.0, 99.5  # pyControl time of the first frames


def video_level(index: int) -> int:
    """Each video frame is one grey level, so the frame shown can be read off a movie."""
    return 2 * index


def write_video(path: Path, n_frames: int = VIDEO_FRAMES) -> None:
    writer = imageio_ffmpeg.write_frames(
        str(path), (VIDEO_W, VIDEO_H), fps=VIDEO_HZ, pix_fmt_in="gray", quality=10,
        macro_block_size=1,
    )  # fmt: skip
    writer.send(None)
    for index in range(n_frames):
        writer.send(np.full((VIDEO_H, VIDEO_W), video_level(index), np.uint8))
    writer.close()


def add_behavior(plane: Path, video_name: str = "behavior.mp4", n_video: int = VIDEO_FRAMES):
    """frame_times.npz and video_frame_times.npz beside the run, as magnitude-bandit-analysis
    writes them, and the video they name. The video's first and last frames have no time
    (outside the sync pulses)."""
    run_dir = plane.parents[4]  # run-01/<distortion>/<destriping>/aligned_*/suite2p/plane0
    video = run_dir.parent / video_name
    write_video(video, n_video)
    np.savez(run_dir / FRAME_TIMES_FILE, time_s=IMAGING_START_S + np.arange(N_FRAMES + 5) / FS)
    video_time_s = VIDEO_START_S + np.arange(VIDEO_FRAMES) / VIDEO_HZ
    video_time_s[[0, -1]] = np.nan
    np.savez(run_dir / VIDEO_FRAME_TIMES_FILE, time_s=video_time_s, video=str(video))
    return run_dir, video


# ---------------------------------------------------------------- pure functions


def test_movie_frame_time_is_the_middle_of_its_frames():
    times = np.arange(10.0)  # one frame per second
    np.testing.assert_allclose(movie_frame_times(times, np.array([0, 2]), 1), [0.5, 2.5])
    np.testing.assert_allclose(movie_frame_times(times, np.array([0, 4]), 3), [1.5, 5.5])


def test_movie_frame_without_a_time_has_none():
    times = np.arange(10.0)
    times[3] = np.nan
    result = movie_frame_times(times, np.array([0, 2, 5]), 2)
    assert np.isfinite(result[0]) and np.isnan(result[1]) and np.isfinite(result[2])


def test_video_frame_shown_is_the_last_one_started():
    video = np.array([np.nan, 0.0, 0.5, 1.0, 1.5, np.nan])
    times = np.array([0.2, 1.4, 1.6, 2.1, -0.1, np.nan, 1.0])
    np.testing.assert_array_equal(video_frame_for(times, video), [1, 3, 4, -1, -1, -1, 3])


def test_video_times_must_increase():
    with pytest.raises(ValueError, match="increase"):
        video_frame_for(np.array([0.1]), np.array([0.0, 0.5, 0.4]))


def test_behavior_panel_keeps_the_aspect_and_an_even_width():
    assert behavior_width((1280, 1024), 478) == 598
    frame = np.full((48, 64), 90, np.uint8)
    panel = behavior_panel(frame, 30, behavior_width((64, 48), 30))
    assert panel.shape == (30, 40, 3) and np.all(panel == 90)
    assert np.all(behavior_panel(None, 30, 40) == 0)


def test_video_frames_are_read_forward_only(tmp_path):
    write_video(tmp_path / "v.mp4", 20)
    with VideoFrames(tmp_path / "v.mp4") as video:
        assert video.size == (VIDEO_W, VIDEO_H)
        assert abs(int(video.frame(3).mean()) - video_level(3)) <= 2
        assert abs(int(video.frame(3).mean()) - video_level(3)) <= 2  # the same frame again
        assert abs(int(video.frame(10).mean()) - video_level(10)) <= 2
        with pytest.raises(ValueError, match="forward"):
            video.frame(5)
        with pytest.raises(ValueError, match="20 frames"):
            video.frame(25)


# ---------------------------------------------------------------- the run's sync files


def test_sync_is_read_from_beside_the_run(tmp_path):
    plane = make_run(tmp_path, tiffs=False)
    run_dir, video = add_behavior(plane)
    sync = load_behavior_sync(run_dir, N_FRAMES)
    assert len(sync.frame_time_s) == N_FRAMES  # the run processed the first N_FRAMES frames
    assert sync.video == video and len(sync.video_time_s) == VIDEO_FRAMES


def test_sync_files_are_required(tmp_path):
    plane = make_run(tmp_path, tiffs=False)
    with pytest.raises(FileNotFoundError, match="frame_times.npz.*magnitude-bandit-analysis"):
        load_behavior_sync(plane.parents[4], N_FRAMES)


def test_too_few_frame_times_are_refused(tmp_path):
    plane = make_run(tmp_path, tiffs=False)
    run_dir, _ = add_behavior(plane)
    with pytest.raises(ValueError, match=f"{N_FRAMES + 5} frame times"):
        load_behavior_sync(run_dir, N_FRAMES + 10)


def test_a_moved_video_is_named_or_given(tmp_path):
    plane = make_run(tmp_path, tiffs=False)
    run_dir, video = add_behavior(plane)
    moved = video.rename(tmp_path / "elsewhere.mp4")
    with pytest.raises(FileNotFoundError, match="behavior_video"):
        load_behavior_sync(run_dir, N_FRAMES)
    assert load_behavior_sync(run_dir, N_FRAMES, video=moved).video == moved


# ---------------------------------------------------------------- whole movies


def test_behavior_makes_a_second_movie_with_the_video_left_of_the_imaging(tmp_path):
    plane = make_run(tmp_path, tiffs=False)
    add_behavior(plane)
    settings = settings_for(tmp_path, averaged_movie=False, behavior=True)
    written = make_movies(settings)
    assert len(written) == 2
    plain, with_video = sorted(written, key=lambda p: "behavior" in p.name)
    assert "behavior" not in plain.name and "_registered_side_by_side_behavior_" in with_video.name
    _, plain_frames = decoded(plain)
    _, frames = decoded(with_video)
    panel = behavior_width((VIDEO_W, VIDEO_H), LY)
    assert frames.shape[1:3] == (LY + LY % 2, panel + PANEL_GAP_PX + plain_frames.shape[2])
    np.testing.assert_allclose(  # the imaging is unchanged beside the video
        frames[:, :, panel + PANEL_GAP_PX :].astype(float), plain_frames.astype(float), atol=6
    )
    for i in (0, 20, N_FRAMES - 1):
        middle_s = IMAGING_START_S + (i + 0.5) / FS
        shown = int(np.floor((middle_s - VIDEO_START_S) * VIDEO_HZ + 1e-9))
        level = frames[i, 5 : LY - 5, 5 : panel - 5].mean()
        assert abs(level - video_level(shown)) <= 3, (i, level, shown)


def test_imaging_after_the_video_shows_a_black_panel(tmp_path):
    plane = make_run(tmp_path, tiffs=False)
    add_behavior(plane, n_video=VIDEO_FRAMES)
    run_dir = plane.parents[4]
    with np.load(run_dir / VIDEO_FRAME_TIMES_FILE) as saved:
        video_time_s, video = saved["time_s"], str(saved["video"])
    video_time_s[40:] = np.nan  # no time after 103.4 s: imaging frames after it have no video
    np.savez(run_dir / VIDEO_FRAME_TIMES_FILE, time_s=video_time_s, video=video)
    written = make_movies(settings_for(tmp_path, averaged_movie=False, behavior=True))
    _, frames = decoded(next(p for p in written if "behavior" in p.name))
    panel = behavior_width((VIDEO_W, VIDEO_H), LY)
    assert frames[0, 5 : LY - 5, 5 : panel - 5].mean() > 5
    assert frames[-1, 5 : LY - 5, 5 : panel - 5].max() <= 3


def test_behavior_without_sync_stops_before_anything_is_written(tmp_path):
    plane = make_run(tmp_path, tiffs=False)
    with pytest.raises(FileNotFoundError, match="frame_times.npz"):
        make_movies(settings_for(tmp_path, behavior=True))
    assert not movie_dir(tmp_path).exists()
    assert plane.is_dir()


def test_behavior_movies_for_every_stage_version_and_crop(tmp_path):
    plane = make_run(tmp_path, tiffs=False)
    add_behavior(plane)
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    ops.update(xoff=np.zeros(N_FRAMES), yoff=np.zeros(N_FRAMES))
    np.save(plane / "ops.npy", ops, allow_pickle=True)
    written = make_movies(settings_for(tmp_path, behavior=True, suite2p_crop=True))
    names = sorted(p.name for p in written)
    assert len(names) == 8  # (unaveraged, averaged) x (whole, cropped) x (plain, behavior)
    assert sum("behavior" in n for n in names) == 4
    assert LX % 2  # the fixture's odd width exercises the padding


# ---------------------------------------------------------------- uneven and broken timing


def test_uneven_video_timing_leaves_no_gaps_inside_the_video():
    rng = np.random.default_rng(0)
    gaps = np.full(6000, 1 / 60) + rng.normal(0, 2e-4, 6000)
    gaps[rng.choice(6000, 12, replace=False)] *= 2  # dropped camera frames
    video = np.concatenate([[0.0], np.cumsum(gaps)])
    times = np.linspace(0.5, video[-1] - 0.5, 700)
    assert np.all(video_frame_for(times, video) >= 0)


@pytest.mark.parametrize("n_timed", [0, 1])
def test_too_few_timed_video_frames_are_refused(tmp_path, n_timed):
    plane = make_run(tmp_path, tiffs=False)
    run_dir, video = add_behavior(plane)
    video_time_s = np.full(VIDEO_FRAMES, np.nan)
    video_time_s[:n_timed] = VIDEO_START_S
    np.savez(run_dir / VIDEO_FRAME_TIMES_FILE, time_s=video_time_s, video=str(video))
    with pytest.raises(ValueError, match=f"has {n_timed} frames with a time"):
        load_behavior_sync(run_dir, N_FRAMES)


def test_a_timing_file_without_its_arrays_is_named(tmp_path):
    plane = make_run(tmp_path, tiffs=False)
    run_dir, _ = add_behavior(plane)
    np.savez(run_dir / VIDEO_FRAME_TIMES_FILE, time_s=np.arange(VIDEO_FRAMES, dtype=float))
    with pytest.raises(ValueError, match="video_frame_times.npz lacks \\['video'\\]"):
        load_behavior_sync(run_dir, N_FRAMES)


def test_imaging_times_that_go_back_are_refused(tmp_path):
    plane = make_run(tmp_path, tiffs=False)
    run_dir, _ = add_behavior(plane)
    times = IMAGING_START_S + np.arange(N_FRAMES) / FS
    times[10] = times[5]
    np.savez(run_dir / FRAME_TIMES_FILE, time_s=times)
    with pytest.raises(ValueError, match="frame times must increase"):
        load_behavior_sync(run_dir, N_FRAMES)


def test_a_video_shorter_than_its_frame_times_is_refused(tmp_path):
    plane = make_run(tmp_path, tiffs=False)
    run_dir, _ = add_behavior(plane, n_video=VIDEO_FRAMES - 20)
    with pytest.raises(ValueError, match=f"about {VIDEO_FRAMES - 20} frames"):
        load_behavior_sync(run_dir, N_FRAMES)


def test_a_video_that_misses_the_imaging_stops_before_anything_is_written(tmp_path):
    plane = make_run(tmp_path, tiffs=False)
    run_dir, video = add_behavior(plane)
    late = VIDEO_START_S + 1000 + np.arange(VIDEO_FRAMES) / VIDEO_HZ  # another session's clock
    np.savez(run_dir / VIDEO_FRAME_TIMES_FILE, time_s=late, video=str(video))
    with pytest.raises(ValueError, match="do not overlap"):
        make_movies(settings_for(tmp_path, behavior=True))
    assert not movie_dir(tmp_path).exists()


def test_imaging_frames_without_a_time_show_a_black_panel(tmp_path):
    plane = make_run(tmp_path, tiffs=False)
    run_dir, _ = add_behavior(plane)
    times = IMAGING_START_S + np.arange(N_FRAMES) / FS
    times[:5] = np.nan  # before the first sync pulse
    np.savez(run_dir / FRAME_TIMES_FILE, time_s=times)
    written = make_movies(settings_for(tmp_path, averaged_movie=False, behavior=True))
    _, frames = decoded(next(p for p in written if "behavior" in p.name))
    panel = behavior_width((VIDEO_W, VIDEO_H), LY)
    assert frames[0, 5 : LY - 5, 5 : panel - 5].max() <= 3
    assert frames[10, 5 : LY - 5, 5 : panel - 5].mean() > 5


@pytest.mark.parametrize("mode", ["bin", "rolling"])
def test_averaged_movies_show_the_video_at_the_middle_of_each_window(tmp_path, mode):
    plane = make_run(tmp_path, tiffs=False)
    add_behavior(plane)
    settings = settings_for(
        tmp_path, unaveraged_movie=False, behavior=True, average_frames=4, average_mode=mode
    )
    written = make_movies(settings)
    _, frames = decoded(next(p for p in written if "behavior" in p.name))
    step = 4 if mode == "bin" else 1
    panel = behavior_width((VIDEO_W, VIDEO_H), LY)
    for k in (0, len(frames) // 2, len(frames) - 1):
        first = k * step
        middle_s = IMAGING_START_S + (first + 2) / FS  # 4 frames from first: middle after 2
        shown = int(np.floor((middle_s - VIDEO_START_S) * VIDEO_HZ + 1e-9))
        level = frames[k, 5 : LY - 5, 5 : panel - 5].mean()
        assert abs(level - video_level(shown)) <= 3, (mode, k, level, shown)

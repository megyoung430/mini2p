"""Frame order: every stage's frames in the acquired order, checked by content."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from mini2p_preprocessing.config import PipelineSettings
from mini2p_preprocessing.correction import preprocess_tiff
from mini2p_preprocessing.distortion import write_distortion_corrected
from mini2p_preprocessing.frame_order import (
    SIGNATURE_CACHE,
    check_frame_order,
    fast_part,
    match_windows,
    order_report,
)
from mini2p_preprocessing.movies import load_run_settings
from tests.conftest import fake_calibration, write_interleaved_tiff
from tests.test_movies import LX, LY, N_FRAMES, make_run
from tests.test_movies import write_interleaved_tiff as write_run_tiff

N = 1200


def flicker(seed: int, n: int = N) -> np.ndarray:
    """Frame-to-frame brightness changes, as laser, detector and motion give."""
    return np.random.default_rng(seed).normal(0, 1, n)


# ---------------------------------------------------------------- signatures


def test_fast_part_removes_a_slow_trend():
    trend = np.linspace(0, 50, 500)
    assert np.abs(fast_part(trend)[20:-20]).max() < 1e-9


def test_frames_in_place_match_their_own_position():
    signal = flicker(0)
    match = match_windows(signal, signal + 0.1 * flicker(1))
    assert set(match.status) == {"in place"} and np.all(match.best == match.start)


def test_a_moved_block_is_found_with_where_it_came_from():
    acquired = flicker(2)
    stage = acquired.copy()
    stage[200:400], stage[600:800] = acquired[600:800], acquired[200:400]  # two blocks swapped
    match = match_windows(acquired, stage)
    moved = match.status == "moved"
    np.testing.assert_array_equal(match.start[moved], [200, 600])
    np.testing.assert_array_equal(match.best[moved], [600, 200])
    assert set(match.status[~moved]) == {"in place"}


def test_an_off_by_one_shift_after_a_seam_is_found():
    acquired = flicker(3)
    stage = np.concatenate([acquired[:600], acquired[601:], acquired[-1:]])  # a frame dropped
    match = match_windows(acquired, stage)
    assert set(match.status[:3]) == {"in place"}
    assert set(match.status[3:5]) == {"moved"} and np.all(match.best[3:5] == match.start[3:5] + 1)


def test_a_signature_too_weak_to_tell_is_unclear_not_wrong():
    acquired = flicker(4)
    stage = flicker(5)  # nothing in common
    assert set(match_windows(acquired, stage).status) == {"unclear"}


def test_matching_refuses_a_stage_longer_than_the_acquisition_or_shorter_than_a_window():
    with pytest.raises(ValueError, match="checked against only"):
        match_windows(flicker(0, 100), flicker(0, 300))
    with pytest.raises(ValueError, match="fewer than one window"):
        match_windows(flicker(0, 300), flicker(0, 100))


def test_swapped_channels_are_reported():
    green, red = 1000 + flicker(6), 500 + flicker(7)
    same = order_report("registered", "green", green, {"green": green, "red": red})
    swapped = order_report("registered", "green", red, {"green": green, "red": red})
    assert same.passed and not same.channels_swapped
    assert swapped.channels_swapped and not swapped.passed
    assert "channels swapped" in swapped.summary()


# ---------------------------------------------------------------- the pipeline's chunked steps


def recording_with_flicker(tmp_path, n_frames=47):
    """Two channels of textured frames, each frame with its own brightness."""
    rng = np.random.default_rng(8)
    texture = rng.normal(0, 1, (48, 96)).cumsum(axis=1)
    levels = [300 + 20 * flicker(9, n_frames), 600 + 30 * flicker(10, n_frames)]
    stacks = [
        (level[:, None, None] + 40 * texture + rng.normal(0, 1, (n_frames, 48, 96))).astype(
            np.float32
        )
        for level in levels
    ]
    return write_interleaved_tiff(tmp_path / "raw_00002.tif", stacks), levels


def page_means(path, channel, n_channels=2):
    with tifffile.TiffFile(path) as tif:
        return np.array([p.asarray().mean() for p in tif.pages[channel::n_channels]])


@pytest.mark.parametrize("method", ["per_frame", "chunked"])
def test_bidiphase_correction_keeps_every_frame_in_place(tmp_path, method):
    raw, levels = recording_with_flicker(tmp_path)
    phase = {"method": method, "chunk_frames": 10, "frames_per_read": 7, "max_shift_px": 8}
    if method == "per_frame":
        phase |= {"trust_window_frames": 11, "fallback_window_frames": 21}
    settings = PipelineSettings.model_validate(
        {"phase": phase, "distortion": {"mode": "off"},
         "registration": {"motion_alignment_source": "functional"}}
    )  # fmt: skip
    out = tmp_path / "corrected.tif"
    preprocess_tiff(raw, out, settings, run_name="run-01")
    for channel, level in enumerate(levels):
        np.testing.assert_allclose(page_means(out, channel), page_means(raw, channel), rtol=0.01)
        assert np.corrcoef(page_means(out, channel), level)[0, 1] > 0.99


def test_distortion_correction_keeps_every_frame_in_place(tmp_path):
    raw, levels = recording_with_flicker(tmp_path)
    settings = PipelineSettings.model_validate(
        {"distortion": {"mode": "local_batched", "transform_mat": "calib.mat", "batch_frames": 9},
         "output_dtype": "float32"}
    )  # fmt: skip
    out = tmp_path / "undistorted.tif"
    write_distortion_corrected(raw, out, settings, calibration=fake_calibration())
    for channel, level in enumerate(levels):
        assert np.corrcoef(page_means(out, channel), level)[0, 1] > 0.99


# ---------------------------------------------------------------- a whole run


def run_with_flicker(tmp_path, move_registered_block: bool = False):
    """A run (see test_movies.make_run) whose stages share each frame's brightness."""
    plane = make_run(tmp_path)
    variant = plane.parents[2]
    run = load_run_settings(variant)
    green_page = run.channels.functional_channel - 1
    rng = np.random.default_rng(11)
    texture = rng.normal(0, 1, (LY, LX))
    green_level = 1000 + 50 * flicker(12, N_FRAMES)
    red_level = 400 + 30 * flicker(13, N_FRAMES)

    def stack(level, shape=(LY, LX)):
        return (level[:, None, None] + 20 * texture[: shape[0], : shape[1]]).astype(np.int16)

    green, red = stack(green_level), stack(red_level)
    raw = tmp_path / "rawdata" / "71rtask_00039.tif"
    write_run_tiff(raw, green, red, green_page)
    for tif in variant.glob("71rtask_00039_*.tif"):
        shape = tifffile.TiffFile(tif).pages[0].shape
        write_run_tiff(tif, green[:, : shape[0], : shape[1]], red[:, : shape[0], : shape[1]],
                       green_page)  # fmt: skip
    registered = green.copy()
    if move_registered_block:
        registered[10:20], registered[30:40] = green[30:40], green[10:20]
    registered.tofile(plane / "data.bin")
    red.tofile(plane / "data_chan2.bin")
    return plane, run


def test_a_run_processed_in_order_passes_every_stage(tmp_path):
    plane, run = run_with_flicker(tmp_path)
    reports = check_frame_order(plane, ("bidiphase", "registered"), run, window=10)
    assert [(r.stage, r.channel) for r in reports] == [
        ("bidiphase", "green"), ("bidiphase", "red"), ("registered", "green"),
        ("registered", "red"),
    ]  # fmt: skip
    assert all(r.passed for r in reports), [r.summary() for r in reports]
    assert (plane / SIGNATURE_CACHE).is_file()


def test_a_block_moved_in_the_registered_binary_is_reported(tmp_path):
    plane, run = run_with_flicker(tmp_path, move_registered_block=True)
    green = check_frame_order(plane, ("registered",), run, window=10)[0]
    assert not green.passed and green.moved == 2
    assert "frames 10-19 match acquired frame 30" in green.summary()


def test_cached_signatures_are_read_again_when_a_file_changes(tmp_path):
    plane, run = run_with_flicker(tmp_path)
    assert check_frame_order(plane, ("registered",), run, window=10)[0].passed
    data = np.fromfile(plane / "data.bin", np.int16).reshape(N_FRAMES, LY, LX)
    data[[*range(10, 20), *range(30, 40)]] = data[[*range(30, 40), *range(10, 20)]]
    data.tofile(plane / "data.bin")  # replaced: new modification time
    assert not check_frame_order(plane, ("registered",), run, window=10)[0].passed


def test_stages_this_check_cannot_cover_are_refused(tmp_path):
    plane, run = run_with_flicker(tmp_path)
    with pytest.raises(ValueError, match="background_removed"):
        check_frame_order(plane, ("background_removed",), run)


def test_frame_order_figure_draws_into_given_axes(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mini2p_preprocessing import plots

    plane, run = run_with_flicker(tmp_path, move_registered_block=True)
    reports = check_frame_order(plane, ("bidiphase", "registered"), run, window=10)
    _, axes = plt.subplots(2, 1)
    assert len(plots.plot_frame_order(reports, axes=axes)) == 2
    plt.close("all")

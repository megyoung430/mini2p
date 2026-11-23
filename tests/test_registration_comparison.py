"""Comparing registrations of one recording, on synthetic movies."""

from __future__ import annotations

import csv

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift

from mini2p_preprocessing.cli import compare_registrations_main
from mini2p_preprocessing.registration_comparison import (
    block_slices,
    common_crop,
    compare_registrations,
    crispness,
    frame_alignment,
    local_correlation_image,
    noise_level,
    output_stems,
    patch_traces,
    peak_snr,
    registration_settings,
    robust_noise,
    sample_frames,
    signal_measures,
    trace_correlations,
)

FS = 7.5
LY = LX = 40
N_FRAMES = 300


def texture(rng, shape=(LY, LX), sigma=1.5):
    image = gaussian_filter(rng.normal(size=shape), sigma)
    return (image - image.min()) / np.ptp(image)


def transients(rng, n_frames, rate=0.03, decay_frames=6.0):
    events = (rng.random(n_frames) < rate).astype(float)
    return np.convolve(events, np.exp(-np.arange(40) / decay_frames))[:n_frames]


# ---- alignment --------------------------------------------------------------------


def test_common_crop_is_the_overlap_and_refuses_disjoint_regions():
    crop = common_crop([(slice(2, 30), slice(0, 20)), (slice(5, 40), slice(3, 25))])
    assert crop == (slice(5, 30), slice(3, 20))
    with pytest.raises(ValueError, match="do not overlap"):
        common_crop([(slice(0, 5), slice(0, 5)), (slice(6, 9), slice(0, 5))])


def test_sampled_frames_are_evenly_spread_and_never_more_than_the_recording():
    assert list(sample_frames(10, 4)) == [0, 3, 6, 9]
    assert list(sample_frames(3, 10)) == [0, 1, 2]


def test_blocks_are_contiguous_spread_and_shortened_to_fit():
    blocks = block_slices(1000, 3, 200)
    assert [(b.start, b.stop) for b in blocks] == [(0, 200), (400, 600), (800, 1000)]
    assert [b.stop - b.start for b in block_slices(300, 3, 200)] == [100, 100, 100]
    with pytest.raises(ValueError, match="fewer blocks"):
        block_slices(100, 5, 200)


def test_blur_lowers_crispness(rng):
    image = 100 + 50 * texture(rng)
    assert crispness(gaussian_filter(image, 2)) < crispness(image)


def test_residual_shift_recovers_motion_left_in_the_frames(rng):
    image = 100 + 100 * texture(rng, (64, 64))
    still = np.stack([image + rng.normal(0, 2, image.shape) for _ in range(30)])
    offsets = rng.uniform(-3, 3, (30, 2))
    moving = np.stack([shift(f, o, mode="nearest") for f, o in zip(still, offsets, strict=True)])
    r_still, s_still = frame_alignment(still, still.mean(axis=0))
    r_moving, s_moving = frame_alignment(moving, moving.mean(axis=0))
    assert np.median(s_still) < 0.3
    assert np.median(s_moving) > 1.0
    assert np.median(r_still) > np.median(r_moving)


def test_registration_settings_come_from_suite2p_1x_or_older_ops():
    nested = {"registration": {"nonrigid": False, "smooth_sigma": 3.0, "norm_frames": True}}
    assert registration_settings(nested) == {
        "nonrigid": False, "smooth_sigma": 3.0, "norm_frames": True
    }  # fmt: skip
    assert registration_settings({"nonrigid": True})["nonrigid"] is True
    assert registration_settings({})["smooth_sigma"] is None


# ---- signal -----------------------------------------------------------------------


def test_patches_average_their_pixels_and_leave_out_the_ragged_edge():
    block = np.arange(2 * 5 * 5, dtype=float).reshape(2, 5, 5)
    traces = patch_traces(block, 2)
    assert traces.shape == (4, 2)
    assert traces[0, 0] == pytest.approx(block[0, :2, :2].mean())


def test_frame_to_frame_noise_ignores_slow_signal(rng):
    noise = rng.normal(0, 2.0, (50, 4000))
    slow = 20 * np.sin(np.linspace(0, 6, 4000))
    assert np.median(robust_noise(noise + slow)) == pytest.approx(2.0, rel=0.05)


def test_peak_snr_is_higher_with_transients_than_noise_alone(rng):
    noise = rng.normal(0, 1, (20, 2000))
    active = noise + 8 * np.stack([transients(rng, 2000) for _ in range(20)])
    assert np.median(peak_snr(active)) > 2 * np.median(peak_snr(noise))


def test_noise_level_matches_its_definition_for_white_noise(rng):
    f0, sd = 100.0, 2.0
    traces = f0 + rng.normal(0, sd, (30, 20000))
    level, baseline = noise_level(traces, FS)
    # |difference| of normal noise: median 0.6745 x sqrt(2) x sd; F0 (20th pct) is near f0
    expected = 100 * 0.6745 * np.sqrt(2) * sd / np.median(baseline) / np.sqrt(FS)
    assert np.median(level) == pytest.approx(expected, rel=0.03)
    level, _ = noise_level(np.zeros((2, 100)), FS)
    assert np.isnan(level).all()


def test_local_correlation_finds_a_cell_and_not_independent_noise(rng):
    movie = rng.normal(0, 1, (600, 20, 20)).astype(np.float32)
    movie[:, 8:12, 8:12] += 3 * transients(rng, 600)[:, None, None]
    image = local_correlation_image(movie, window=75)
    assert image[9:11, 9:11].min() > 0.5
    assert abs(np.median(image)) < 0.1


def test_trace_correlations_pair_rows():
    a = np.array([[1.0, 2, 3, 4], [1, 2, 3, 4], [1, 1, 1, 1]])
    b = np.array([[2.0, 4, 6, 8], [4, 3, 2, 1], [1, 2, 3, 4]])
    assert np.allclose(trace_correlations(a, b)[:2], [1, -1])
    assert np.isnan(trace_correlations(a, b)[2])


def test_shared_motion_shows_in_red_fluctuation_and_green_red_correlation(rng):
    def block(wobble):
        red_image, green_image = 400 + 300 * texture(rng), 200 + 100 * texture(rng)
        common = wobble * np.sin(np.arange(600) / 4)[:, None, None]  # field-wide brightness
        green = green_image * (1 + common) + rng.normal(0, 3, (600, LY, LX))
        red = red_image * (1 + common) + rng.normal(0, 3, (600, LY, LX))
        return green, red

    still, _ = signal_measures([block(0.0)], FS)
    wobbly, _ = signal_measures([block(0.05)], FS)
    assert wobbly["red_fluctuation_pct"] > 2 * still["red_fluctuation_pct"]
    assert wobbly["green_red_corr_median"] > 0.8
    assert abs(still["green_red_corr_median"]) < 0.2


# ---- the whole comparison ---------------------------------------------------------


def write_registration(aligned, name, green, red, nonrigid=False):
    plane = aligned / name / "plane0"
    plane.mkdir(parents=True)
    green.astype(np.int16).tofile(plane / "data.bin")
    red.astype(np.int16).tofile(plane / "data_chan2.bin")
    ops = {
        "nframes": len(green), "Ly": LY, "Lx": LX, "fs": FS,
        "yoff": np.zeros(len(green)), "xoff": np.zeros(len(green)),
        "badframes": np.zeros(len(green), bool),
        "registration": {"nonrigid": nonrigid, "smooth_sigma": 1.15, "norm_frames": False},
        "reg_file": str(plane / "data.bin"), "reg_file_chan2": str(plane / "data_chan2.bin"),
    }  # fmt: skip
    np.save(plane / "ops.npy", ops)  # after the binaries, as Suite2p saves it
    return plane


def sweep(tmp_path, rng, n_frames=N_FRAMES):
    """Two registrations of one recording: one aligned, one with motion left in it."""
    red_image, green_image = 300 + 600 * texture(rng), 100 + 300 * texture(rng)
    cell = np.zeros((LY, LX))
    cell[18:23, 18:23] = 1
    activity = transients(rng, n_frames)
    green = (
        green_image + 400 * activity[:, None, None] * cell + rng.normal(0, 5, (n_frames, LY, LX))
    )
    red = red_image + rng.normal(0, 5, (n_frames, LY, LX))
    offsets = rng.uniform(-2, 2, (n_frames, 2))
    moved = [np.stack([shift(f, o, mode="nearest") for f, o in zip(movie, offsets, strict=True)])
             for movie in (green, red)]  # fmt: skip
    aligned = tmp_path / "aligned_structural"
    write_registration(aligned, "suite2p_good", green, red)
    write_registration(aligned, "suite2p_wobbly", *moved, nonrigid=True)
    return aligned


def test_the_aligned_registration_wins_and_everything_is_saved(tmp_path, rng):
    aligned = sweep(tmp_path, rng)
    rows = compare_registrations(aligned, n_frames=100, n_blocks=2, block_frames=150)
    good, wobbly = rows
    assert (good["registration"], wobbly["nonrigid"]) == ("suite2p_good", True)
    assert good["red_corr_median"] > wobbly["red_corr_median"]
    assert good["red_residual_shift_px"] < wobbly["red_residual_shift_px"]
    assert good["red_crispness"] > wobbly["red_crispness"]
    assert good["red_fluctuation_pct"] < wobbly["red_fluctuation_pct"]
    assert good["green_local_corr_contrast"] > wobbly["green_local_corr_contrast"]
    table, images = output_stems(aligned)
    with open(table.with_suffix(".csv")) as handle:
        names = [r["registration"] for r in csv.DictReader(handle)]
    assert names == ["suite2p_good", "suite2p_wobbly"]
    for stem in (table, images):
        for suffix in (".png", ".pdf", ".svg", ".caption.txt"):
            assert stem.with_name(stem.name + suffix).is_file()
    with pytest.raises(FileExistsError):
        compare_registrations(aligned, n_frames=100, n_blocks=2, block_frames=150)
    compare_registrations(aligned, n_frames=100, n_blocks=2, block_frames=150, overwrite=True)


def test_registrations_of_different_recordings_are_refused(tmp_path, rng):
    aligned = sweep(tmp_path, rng)
    frames = np.zeros((N_FRAMES - 10, LY, LX))
    write_registration(aligned, "suite2p_other", frames, frames)
    with pytest.raises(ValueError, match="nframes"):
        compare_registrations(aligned, n_frames=100, n_blocks=2, block_frames=100)


def test_the_command_prints_the_summary_and_reports_bad_requests(tmp_path, rng, capsys):
    aligned = sweep(tmp_path, rng)
    args = [str(aligned), "--frames", "50", "--blocks", "2", "--block-frames", "100"]
    compare_registrations_main(args)
    out = capsys.readouterr().out
    assert "suite2p_good" in out and "suite2p_wobbly" in out
    with pytest.raises(SystemExit):
        compare_registrations_main(args)  # exists, no --overwrite
    with pytest.raises(SystemExit):
        compare_registrations_main([str(tmp_path / "nothing")])

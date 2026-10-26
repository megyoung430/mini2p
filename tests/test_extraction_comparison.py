"""Comparing neuropil and background corrections on synthetic traces."""

from __future__ import annotations

import numpy as np
import pytest

from mini2p_preprocessing.extraction_comparison import (
    background_channel_check,
    contrast_ratio,
    corrected_traces,
    fitted_neuropil_coefficients,
    high_pass,
    method_metrics,
    valid_everywhere,
)
from mini2p_preprocessing.fluorescence import DffResult

N_FRAMES, FS = 4000, 7.5


def transients(rng, n_rois, rate=0.01, amplitude=30.0):
    events = (rng.random((n_rois, N_FRAMES)) < rate).astype(float)
    kernel = np.exp(-np.arange(30) / 8)
    return amplitude * np.array([np.convolve(e, kernel)[:N_FRAMES] for e in events])


def synthetic(coefficients, seed=0, gain=None, neuropil_noise=0.3):
    """ROIs: baseline + transients + c x the true neuropil + noise, all times a field-wide
    *gain* (bleaching, laser power); the measured neuropil is the true one plus its own
    noise, which never reaches the ROI. Returns F, Fneu and the field-wide signal."""
    rng = np.random.default_rng(seed)
    n = len(coefficients)
    local = np.array([np.convolve(rng.normal(0, 3, N_FRAMES), np.ones(5) / 5, "same")
                      for _ in range(n)])  # fmt: skip
    true_neuropil = 20 + local
    gain = np.ones(N_FRAMES) if gain is None else gain
    f = gain * (10 + transients(rng, n) + np.asarray(coefficients)[:, None] * true_neuropil)
    f = f + rng.normal(0, 1, f.shape)
    fneu = gain * true_neuropil + rng.normal(0, neuropil_noise, f.shape)
    return f, fneu, 20 * gain


def result_of(f_corr, f0):
    f_corr, f0 = np.asarray(f_corr, float), np.asarray(f0, float)
    invalid = f0 <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        dff = np.where(invalid, np.nan, (f_corr - f0) / f0)
    return DffResult(f_corr, f0, f_corr - f0, dff, invalid)


# ---------------------------------------------------------------- fitted neuropil


TRUTH = np.array([0.2, 0.5, 0.7, 0.9, 0.4, 0.6])


def test_fitted_coefficients_recover_the_neuropil_in_each_roi():
    f, fneu, _ = synthetic(TRUTH)
    fitted = fitted_neuropil_coefficients(f, fneu, FS)
    assert np.all(fitted <= TRUTH + 0.03)  # a lower bound: neuropil noise biases it down
    assert np.all(fitted >= TRUTH - 0.1)


def test_a_field_wide_change_is_not_taken_as_neuropil():
    rng = np.random.default_rng(4)
    bleach = np.linspace(1, 0.85, N_FRAMES)
    flicker = 1 + 0.2 * np.convolve(rng.normal(0, 1, N_FRAMES), np.ones(8) / 8, "same")
    f, fneu, field = synthetic(TRUTH, gain=bleach * flicker)
    fitted = fitted_neuropil_coefficients(f, fneu, FS, background=field)
    np.testing.assert_allclose(fitted, TRUTH, atol=0.12)
    unaware = fitted_neuropil_coefficients(f, fneu, FS)  # without the background regressor
    assert np.median(unaware) > np.median(fitted) + 0.1


def test_leaving_out_transients_steadies_the_fit():
    errors = {True: [], False: []}
    for seed in range(8):
        rng = np.random.default_rng(seed + 50)
        f, fneu, _ = synthetic(np.full(6, 0.5), seed=seed + 50)
        f = f + transients(rng, 6, rate=0.03, amplitude=80.0)  # active cells
        for trim in errors:
            fitted = fitted_neuropil_coefficients(f, fneu, FS, transient_sd=2.0 if trim else 1e9)
            errors[trim].extend(fitted - 0.5)
    assert np.std(errors[True]) < 0.75 * np.std(errors[False])


def test_fitted_coefficients_are_kept_between_bounds():
    f, fneu, _ = synthetic([1.6, -0.3])
    np.testing.assert_array_equal(fitted_neuropil_coefficients(f, fneu, FS), [1.0, 0.0])


def test_fitted_coefficients_refuse_flat_or_bad_input():
    f, fneu, _ = synthetic([0.5, 0.5])
    flat = fneu.copy()
    flat[1] = 3.0
    with pytest.raises(ValueError, match=r"ROI\(s\) \[1\]"):
        fitted_neuropil_coefficients(f, flat, FS)
    broken = f.copy()
    broken[0, 3] = np.nan
    with pytest.raises(ValueError, match="F has values that are not finite"):
        fitted_neuropil_coefficients(broken, fneu, FS)
    with pytest.raises(ValueError, match="transient_sd"):
        fitted_neuropil_coefficients(f, fneu, FS, transient_sd=0)


# ---------------------------------------------------------------- corrections


def test_contrast_ratio_is_median_f_over_median_neuropil():
    f = np.array([[10.0, 12, 14], [5, 5, 5]])
    fneu = np.array([[5.0, 6, 7], [5, 5, 5]])
    np.testing.assert_allclose(contrast_ratio(f, fneu), [2.0, 1.0])
    assert np.isnan(contrast_ratio(np.ones((1, 3)), np.zeros((1, 3)))[0])


def test_corrected_traces_name_and_build_every_method():
    f, fneu, field = synthetic([0.5, 0.7], gain=1 + 0.1 * np.sin(np.arange(N_FRAMES) / 50))
    traces = corrected_traces(f, fneu, {"Vessel": field}, 0.7, np.array([0.5, 0.7]))
    assert list(traces) == [
        "Neuropil 0.7",
        "Neuropil Fitted",
        "No Neuropil",
        "No Neuropil, Vessel Bg Regressed",
        "Neuropil 0.7, Vessel Bg Regressed",
    ]
    np.testing.assert_allclose(traces["No Neuropil"], f, rtol=1e-6)
    np.testing.assert_allclose(traces["Neuropil 0.7"], f - 0.7 * fneu, rtol=1e-5)
    np.testing.assert_allclose(traces["Neuropil Fitted"][0], f[0] - 0.5 * fneu[0], rtol=1e-5)
    regressed = traces["No Neuropil, Vessel Bg Regressed"]
    assert all(abs(np.corrcoef(row, field)[0, 1]) < 1e-3 for row in regressed)
    np.testing.assert_allclose(regressed.mean(axis=1), f.mean(axis=1), rtol=1e-5)
    with pytest.raises(ValueError, match="2 ROIs"):
        corrected_traces(f, fneu, {}, 0.7, np.array([0.5]))


# ---------------------------------------------------------------- metrics


def test_valid_everywhere_is_the_rois_valid_under_every_correction():
    a = result_of(np.ones((3, 5)), np.array([[1.0] * 5, [1] * 5, [0] + [1] * 4]))
    b = result_of(np.ones((3, 5)), np.array([[1.0] * 5, [-1] * 5, [1] * 5]))
    np.testing.assert_array_equal(valid_everywhere({"a": a, "b": b}), [True, False, False])


def test_metrics_count_invalid_baselines_on_all_rois():
    rng = np.random.default_rng(0)
    f_corr = 1 + rng.normal(0, 0.1, (3, 100))
    f0 = np.ones((3, 100))
    f0[0, :10] = 0  # 10 % of frames
    f0[1] = -1  # all frames
    metrics = method_metrics(result_of(f_corr, f0), rng.normal(size=100), FS,
                             common=np.array([False, False, True]))  # fmt: skip
    assert metrics["rois_invalid_f0_anywhere"] == 2
    assert metrics["rois_invalid_f0_mostly"] == 1
    assert metrics["fraction_frames_invalid_f0"] == pytest.approx((10 + 100) / 300)


def test_metrics_without_a_common_roi_still_count_baselines():
    rng = np.random.default_rng(8)
    f0 = np.ones((2, 50))
    f0[0] = -1
    metrics = method_metrics(result_of(1 + rng.normal(0, 0.1, (2, 50)), f0), rng.normal(size=50),
                             FS, common=np.zeros(2, bool))  # fmt: skip
    assert metrics["rois_invalid_f0_anywhere"] == 1
    assert np.isnan(metrics["median_noise"]) and np.isnan(metrics["median_skew"])


def test_metrics_see_a_fast_shared_signal_and_its_removal():
    rng = np.random.default_rng(6)
    flicker = 1 + 0.3 * np.convolve(rng.normal(0, 1, N_FRAMES), np.ones(8) / 8, "same")
    f, fneu, field = synthetic(np.full(5, 0.9), gain=flicker)
    common = np.ones(5, bool)
    raw = method_metrics(result_of(f, np.full_like(f, 20.0)), field, FS, common)
    regressed = corrected_traces(f, fneu, {"Vessel": field}, 0.7, np.full(5, 0.9))
    clean = method_metrics(
        result_of(regressed["No Neuropil, Vessel Bg Regressed"], np.full_like(f, 20.0)),
        field, FS, common,
    )  # fmt: skip
    assert raw["median_abs_r_background"] > 0.3 and clean["median_abs_r_background"] < 0.1
    assert raw["mean_pairwise_r"] > clean["mean_pairwise_r"] + 0.1


def test_metrics_ignore_a_slow_shared_trend():
    rng = np.random.default_rng(7)
    trend = np.linspace(0, 50, N_FRAMES)
    f_corr = 100 + trend + rng.normal(0, 1, (4, N_FRAMES))
    metrics = method_metrics(result_of(f_corr, np.full_like(f_corr, 100.0)), trend, FS,
                             np.ones(4, bool))  # fmt: skip
    assert abs(metrics["mean_pairwise_r"]) < 0.05


def test_metrics_noise_is_the_frame_to_frame_sd():
    rng = np.random.default_rng(1)
    f_corr = 100 + rng.normal(0, 5, (2, 20000))
    metrics = method_metrics(result_of(f_corr, np.full_like(f_corr, 100.0)), rng.normal(size=20000),
                             FS, np.ones(2, bool))  # fmt: skip
    assert metrics["median_noise"] == pytest.approx(0.05, rel=0.05)


# ---------------------------------------------------------------- green vs red background


def fluctuation(seed, sd=0.05):
    rng = np.random.default_rng(seed)
    return sd * np.convolve(rng.normal(0, 1, N_FRAMES), np.ones(10) / 10, "same") * np.sqrt(10)


def test_shared_change_gives_slope_one_whatever_else_is_in_green():
    rng = np.random.default_rng(2)
    shared, own = fluctuation(1), fluctuation(2)
    bleach = np.linspace(1, 0.9, N_FRAMES)  # slow, green only: not counted
    green = 200 * bleach * (1 + shared + own) + rng.normal(0, 0.5, N_FRAMES)
    red = 50 * (1 + shared) + rng.normal(0, 0.1, N_FRAMES)
    check = background_channel_check(green, red, FS)
    assert check["green_on_red_slope"] == pytest.approx(1, abs=0.1)
    assert 0.3 < check["r_squared"] < 0.7  # half of green's fluctuation is its own


def test_blood_volume_moves_green_more_than_red():
    volume = fluctuation(3)
    check = background_channel_check(200 * (1 - 2 * volume), 50 * (1 - 0.5 * volume), FS)
    assert check["green_on_red_slope"] == pytest.approx(4, rel=0.05)


def test_the_slope_range_holds_the_truth_when_red_is_noisy():
    rng = np.random.default_rng(9)
    volume = fluctuation(6, sd=0.02)
    green = 200 * (1 - 4 * volume) + rng.normal(0, 0.5, N_FRAMES)
    red = 50 * (1 - volume) + rng.normal(0, 2, N_FRAMES)  # red noise drags the slope down
    check = background_channel_check(green, red, FS)
    assert check["green_on_red_slope"] < 2.5
    assert check["slope_low"] < 4 < check["slope_high"]


def test_a_dark_offset_is_taken_off_before_the_fractions():
    shared = fluctuation(4)
    red_with_offset = 50 + 50 * (1 + shared)  # 50 counts of offset
    compressed = background_channel_check(200 * (1 + shared), red_with_offset, FS)
    corrected = background_channel_check(200 * (1 + shared), red_with_offset, FS, dark_red=50)
    assert compressed["green_on_red_slope"] == pytest.approx(2, rel=0.05)
    assert corrected["green_on_red_slope"] == pytest.approx(1, rel=0.05)
    with pytest.raises(ValueError, match="red"):
        background_channel_check(np.full(10, 5.0), np.full(10, 3.0), FS, dark_red=3)


def test_high_pass_removes_a_linear_trend_away_from_the_ends():
    trend = np.linspace(0, 10, N_FRAMES)
    window = int(60 * FS)
    assert np.abs(high_pass(trend, FS)[window:-window]).max() < 1e-9


# ---------------------------------------------------------------- figures


def test_comparison_figures_draw_into_given_axes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mini2p_preprocessing import plots

    f, fneu, _ = synthetic([0.5, 0.7, 0.9])
    _, ax = plt.subplots()
    assert plots.plot_contrast_ratio(contrast_ratio(f, fneu), ax=ax) is ax
    _, ax = plt.subplots()
    assert plots.plot_fitted_coefficients(np.array([0.2, 0.5, 0.9]), 0.7, ax=ax) is ax
    _, ax = plt.subplots()
    assert plots.plot_f0_by_method({"A": np.arange(3.0), "B": -np.arange(3.0)}, 0, ax=ax) is ax
    metric = {"A": {"median_f0": 1.0, "mean_pairwise_r": 0.2},
              "B": {"median_f0": 2.0, "mean_pairwise_r": 0.1}}  # fmt: skip
    _, axes = plt.subplots(1, 2)
    shown = plots.plot_method_metrics(metric, ["median_f0", "mean_pairwise_r"], 0, axes=axes)
    assert [a.get_title() for a in shown] == ["Median F0", "Mean Pairwise r Between ROIs"]
    shared = fluctuation(5)
    green, red = 200 * (1 + shared), 50 * (1 + shared)
    check = background_channel_check(green, red, FS)
    _, axes = plt.subplots(1, 2)
    assert len(plots.plot_background_channels(green, red, FS, check, axes=axes)) == 2
    plt.close("all")

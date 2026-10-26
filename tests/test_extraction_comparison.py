"""Comparing neuropil and background corrections on synthetic traces."""

from __future__ import annotations

import numpy as np
import pytest

from mini2p_preprocessing.extraction_comparison import (
    background_channel_check,
    contrast_ratio,
    corrected_traces,
    fitted_neuropil_coefficients,
    method_metrics,
)
from mini2p_preprocessing.fluorescence import DffResult

N_ROIS, N_FRAMES = 6, 3000


def synthetic(coefficients, seed=0):
    """ROIs whose F is a sparse transient train plus *coefficients* x their neuropil plus
    noise; the neuropil carries a slow shared signal."""
    rng = np.random.default_rng(seed)
    shared = 20 + 5 * np.sin(np.arange(N_FRAMES) / 200)
    fneu = shared + rng.normal(0, 1, (len(coefficients), N_FRAMES))
    events = (rng.random((len(coefficients), N_FRAMES)) < 0.01).astype(float)
    kernel = np.exp(-np.arange(30) / 8)
    cell = 30 * np.array([np.convolve(e, kernel)[:N_FRAMES] for e in events])
    f = 10 + cell + np.asarray(coefficients)[:, None] * fneu + rng.normal(0, 1, fneu.shape)
    return f, fneu, shared


def result_of(f_corr, f0):
    f_corr, f0 = np.asarray(f_corr, float), np.asarray(f0, float)
    invalid = f0 <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        dff = np.where(invalid, np.nan, (f_corr - f0) / f0)
    return DffResult(f_corr, f0, f_corr - f0, dff, invalid)


# ---------------------------------------------------------------- fitted neuropil


def test_fitted_coefficients_recover_the_neuropil_in_each_roi():
    truth = np.array([0.2, 0.5, 0.7, 0.9, 0.4, 0.6])
    f, fneu, _ = synthetic(truth)
    np.testing.assert_allclose(fitted_neuropil_coefficients(f, fneu), truth, atol=0.08)


def test_fitted_coefficients_are_kept_between_bounds():
    f, fneu, _ = synthetic([1.6, -0.3])
    np.testing.assert_array_equal(fitted_neuropil_coefficients(f, fneu), [1.0, 0.0])


def test_fitted_coefficients_refuse_a_flat_neuropil():
    f, fneu, _ = synthetic([0.5, 0.5])
    fneu[1] = 3.0
    with pytest.raises(ValueError, match=r"ROI\(s\) \[1\]"):
        fitted_neuropil_coefficients(f, fneu)


def test_fitted_coefficients_need_a_positive_transient_threshold():
    f, fneu, _ = synthetic([0.5])
    with pytest.raises(ValueError, match="transient_sd"):
        fitted_neuropil_coefficients(f, fneu, transient_sd=0)


# ---------------------------------------------------------------- corrections


def test_contrast_ratio_is_median_f_over_median_neuropil():
    f = np.array([[10.0, 12, 14], [5, 5, 5]])
    fneu = np.array([[5.0, 6, 7], [5, 5, 5]])
    np.testing.assert_allclose(contrast_ratio(f, fneu), [2.0, 1.0])


def test_contrast_ratio_is_nan_without_a_positive_neuropil():
    ratio = contrast_ratio(np.ones((1, 3)), np.zeros((1, 3)))
    assert np.isnan(ratio[0])


def test_corrected_traces_name_and_build_every_method():
    f, fneu, shared = synthetic([0.5, 0.7])
    traces = corrected_traces(f, fneu, {"Vessel": shared}, 0.7, np.array([0.5, 0.7]))
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
    assert all(abs(np.corrcoef(row, shared)[0, 1]) < 1e-3 for row in regressed)
    np.testing.assert_allclose(regressed.mean(axis=1), f.mean(axis=1), rtol=1e-5)


def test_corrected_traces_need_one_fitted_coefficient_per_roi():
    f, fneu, _ = synthetic([0.5, 0.7])
    with pytest.raises(ValueError, match="2 ROIs"):
        corrected_traces(f, fneu, {}, 0.7, np.array([0.5]))


# ---------------------------------------------------------------- metrics


def test_metrics_count_invalid_baselines():
    f_corr = np.ones((3, 100))
    f0 = np.ones((3, 100))
    f0[0, :10] = 0  # 10 % of frames
    f0[1] = -1  # all frames
    metrics = method_metrics(result_of(f_corr, f0), np.arange(100.0))
    assert metrics["rois_f0_le_0_anywhere"] == 2
    assert metrics["rois_f0_le_0_median"] == 1
    assert metrics["fraction_frames_f0_le_0"] == pytest.approx((10 + 100) / 300)


def test_metrics_see_a_shared_signal_and_its_removal():
    f, fneu, shared = synthetic([0.9, 0.9, 0.9, 0.9])
    raw = method_metrics(result_of(f, np.full_like(f, 20.0)), shared)
    traces = corrected_traces(f, fneu, {"Vessel": shared}, 0.7, np.full(4, 0.9))
    regressed = traces["No Neuropil, Vessel Bg Regressed"]
    clean = method_metrics(result_of(regressed, np.full_like(f, 20.0)), shared)
    assert raw["median_abs_r_background"] > 0.3 and clean["median_abs_r_background"] < 0.01
    assert raw["mean_pairwise_r"] > clean["mean_pairwise_r"]


def test_metrics_noise_is_the_frame_to_frame_sd():
    rng = np.random.default_rng(1)
    f_corr = 100 + rng.normal(0, 5, (2, 20000))
    metrics = method_metrics(result_of(f_corr, np.full_like(f_corr, 100.0)), np.arange(20000.0))
    assert metrics["median_noise"] == pytest.approx(0.05, rel=0.05)


# ---------------------------------------------------------------- green vs red background


def test_background_channel_check_tells_proportional_from_green_only():
    rng = np.random.default_rng(2)
    dip = 1 + 0.1 * np.sin(np.arange(5000) / 300)
    green = 200 * dip + rng.normal(0, 1, 5000)
    same = background_channel_check(green, 50 * dip + rng.normal(0, 0.2, 5000))
    assert same["red_vs_green_slope"] == pytest.approx(1, abs=0.05) and same["r"] > 0.9
    green_only = background_channel_check(green, 50 + rng.normal(0, 0.2, 5000))
    assert abs(green_only["red_vs_green_slope"]) < 0.05 and abs(green_only["r"]) < 0.1


def test_background_channel_check_refuses_a_mean_at_or_below_zero():
    with pytest.raises(ValueError, match="red"):
        background_channel_check(np.full(10, 5.0), np.zeros(10))


# ---------------------------------------------------------------- figures


def test_comparison_figures_draw_into_given_axes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mini2p_preprocessing import plots

    f, fneu, shared = synthetic([0.5, 0.7, 0.9])
    _, ax = plt.subplots()
    assert plots.plot_contrast_ratio(contrast_ratio(f, fneu), ax=ax) is ax
    _, ax = plt.subplots()
    assert plots.plot_fitted_coefficients(np.array([0.2, 0.5, 0.9]), 0.7, ax=ax) is ax
    _, ax = plt.subplots()
    assert plots.plot_f0_by_method({"A": np.arange(3.0), "B": -np.arange(3.0)}, ax=ax) is ax
    metric = {"A": {"median_f0": 1.0, "mean_pairwise_r": 0.2},
              "B": {"median_f0": 2.0, "mean_pairwise_r": 0.1}}  # fmt: skip
    _, axes = plt.subplots(1, 2)
    shown = plots.plot_method_metrics(metric, ["median_f0", "mean_pairwise_r"], axes=axes)
    assert [a.get_title() for a in shown] == ["Median F0", "Mean Pairwise r Between ROIs"]
    check = background_channel_check(shared, 0.5 * shared)
    _, axes = plt.subplots(1, 2)
    assert len(plots.plot_background_channels(shared, 0.5 * shared, 7.5, check, axes=axes)) == 2
    plt.close("all")

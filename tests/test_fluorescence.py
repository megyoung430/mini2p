import numpy as np
import pytest
from pydantic import ValidationError

from mini2p_preprocessing.config import FluorescenceSettings
from mini2p_preprocessing.fluorescence import (
    compute_dff,
    green_red_dff,
    neuropil_correct,
    select_frames,
)

FS = 7.52
SETTINGS = FluorescenceSettings(baseline="prctile", prctile_baseline=10.0)


def traces_with_known_baseline(rng, baselines, n_frames=600, transient=2.0):
    """Flat baseline per cell plus sparse transients of known relative size."""
    baselines = np.asarray(baselines, dtype=np.float32)[:, None]
    events = np.zeros((len(baselines), n_frames), np.float32)
    events[:, 100:110] = transient  # 5x10/600 of frames: far above the 10th percentile
    return baselines * (1 + events)


def test_neuropil_correct_uses_coefficient_and_leaves_inputs():
    f = np.array([[10.0, 12.0]])
    fneu = np.array([[4.0, 4.0]])
    out = neuropil_correct(f, fneu, 0.5)
    assert np.allclose(out, [[8.0, 10.0]])
    assert np.allclose(f, [[10.0, 12.0]])


def test_dff_recovers_known_f0_and_relative_change(rng):
    f = traces_with_known_baseline(rng, [50.0, 400.0], transient=2.0)
    result = compute_dff(f, np.zeros_like(f), FS, SETTINGS)
    assert np.allclose(result.f0[:, 0], [50.0, 400.0], rtol=1e-4)
    assert np.allclose(result.delta_f, result.f_corr - result.f0)
    assert np.allclose(result.dff[:, 105], 2.0, rtol=1e-4)  # same ΔF/F0 for both cells
    assert np.allclose(result.dff[:, 0], 0.0, atol=1e-4)
    assert not result.invalid_f0.any()


def test_maximin_baseline_follows_slow_drift(rng):
    t = np.arange(4000) / FS
    drift = 100 + 20 * np.sin(2 * np.pi * t / 500)  # slow compared with the 60 s window
    f = drift[None].astype(np.float32)
    result = compute_dff(f, np.zeros_like(f), FS, FluorescenceSettings())
    assert np.abs(result.dff[0, 500:-500]).max() < 0.1


def test_inputs_are_never_modified_even_for_suite2p_inplace_baselines(rng):
    f = traces_with_known_baseline(rng, [50.0])
    fneu = np.full_like(f, 5.0)
    f_before, fneu_before = f.copy(), fneu.copy()
    compute_dff(f, fneu, FS, SETTINGS)  # Suite2p's prctile baseline subtracts in place
    assert np.array_equal(f, f_before) and np.array_equal(fneu, fneu_before)


def test_cells_with_nonpositive_f0_get_nan_dff_and_are_flagged(rng):
    f = traces_with_known_baseline(rng, [50.0, 10.0])
    fneu = np.zeros_like(f)
    fneu[1] = 30.0  # 10 - 0.7*30 < 0: neuropil-dominated cell
    result = compute_dff(f, fneu, FS, SETTINGS)
    assert result.invalid_f0[1].all() and not result.invalid_f0[0].any()
    assert np.isnan(result.dff[1]).all()
    assert np.isfinite(result.dff[0]).all()
    assert np.isfinite(result.delta_f[1]).all()  # ΔF itself stays usable


def test_min_f0_threshold_is_configurable(rng):
    f = traces_with_known_baseline(rng, [50.0, 400.0])
    strict = SETTINGS.model_copy(update={"min_f0": 100.0})
    result = compute_dff(f, np.zeros_like(f), FS, strict)
    assert result.invalid_f0[0].all() and not result.invalid_f0[1].any()


def test_constant_baseline_is_rejected_because_suite2p_pools_it_across_cells():
    with pytest.raises(ValidationError):
        FluorescenceSettings(baseline="constant")


def test_shape_mismatch_fails_loudly():
    with pytest.raises(ValueError, match="shape"):
        compute_dff(np.ones((2, 10)), np.ones((3, 10)), FS, SETTINGS)


def test_select_frames_accepts_mask_or_indices_and_checks_bounds():
    traces = np.arange(12.0).reshape(2, 6)
    mask = np.array([True, False, True, False, False, True])
    assert np.array_equal(select_frames(traces, mask), traces[:, [0, 2, 5]])
    assert np.array_equal(select_frames(traces, np.array([1, 3])), traces[:, [1, 3]])
    assert select_frames(traces, None) is traces
    with pytest.raises(ValueError, match="frames"):
        select_frames(traces, np.ones(5, bool))
    with pytest.raises(ValueError, match="frames"):
        select_frames(traces, np.array([0, 6]))


@pytest.mark.parametrize(
    "frames",
    [
        np.array([1, 0, 1, 1, 0, 1], np.uint8),  # a 0/1 mask saved as integers
        np.array([3, 1]),  # out of time order
        np.array([1, 1, 2]),
    ],  # repeated
)
def test_select_frames_rejects_ambiguous_integer_selections(frames):
    with pytest.raises(ValueError, match="increasing"):
        select_frames(np.arange(12.0).reshape(2, 6), frames)


@pytest.mark.parametrize("frames", [np.array([]), np.zeros(6, bool), np.array([0.0, 2.0])])
def test_select_frames_rejects_empty_and_float_selections(frames):
    with pytest.raises(ValueError, match="frames"):
        select_frames(np.arange(12.0).reshape(2, 6), frames)


def test_non_finite_traces_fail_loudly():
    f = np.full((1, 50), 10.0, np.float32)
    f[0, 20] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_dff(f, np.zeros_like(f), FS, SETTINGS)


def test_maximin_recovers_flat_baseline_under_sparse_transients(rng):
    f = traces_with_known_baseline(rng, [80.0], n_frames=3000, transient=1.0)
    result = compute_dff(f, np.zeros_like(f), FS, FluorescenceSettings())
    assert np.allclose(result.f0[0, 500:2500], 80.0, rtol=1e-3)


def test_green_red_dff_cancels_shared_brightness_changes(rng):
    n = 600
    shared = np.linspace(1.0, 0.6, n, dtype=np.float32)  # e.g. bleaching or z-drift
    activity = np.ones(n, np.float32)
    activity[200:210] = 1.5
    green = (300 * shared * activity)[None]
    red = (200 * shared)[None]
    zeros = np.zeros_like(green)
    gr = green_red_dff(green, zeros, red, zeros, FS, SETTINGS)
    assert np.abs(gr.dff[0, :150]).max() < 0.02  # drift removed by the ratio
    assert gr.dff[0, 205] == pytest.approx(0.5, rel=0.05)


def test_green_red_median_filter_and_red_floor_at_a_fast_frame_rate():
    fs = 30.0  # 0.25 s window = 7 frames, so the median filter is active
    n = 900
    green = np.full((1, n), 300.0, np.float32)
    green[0, 400] = 3000.0  # single-frame spike: removed by the median filter
    red = np.full((1, n), 200.0, np.float32)
    red[0, 600:640] = -50.0  # red drops to <= 0: the floor stops the ratio exploding
    zeros = np.zeros_like(green)
    gr = green_red_dff(green, zeros, red, zeros, fs, SETTINGS)
    assert abs(gr.dff[0, 400]) < 0.01
    assert np.isfinite(gr.dff).all()
    assert np.nanmax(np.abs(gr.dff)) < 1e3

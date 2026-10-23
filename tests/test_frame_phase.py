"""Per-frame bidiphase: measuring each frame's phase and choosing the channel stretch by stretch."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from mini2p_preprocessing.bidiphase import apply_bidiphase
from mini2p_preprocessing.config import PhaseSettings
from mini2p_preprocessing.frame_phase import (
    apply_frame_phases,
    best_shifts,
    choose_frame_phases,
    fallback_stretches,
    line_correlation_curves,
    local_consistency,
    phase_loss,
    running_median,
    windowed_best_shifts,
)
from tests.conftest import misalign, textured_image

MAX_SHIFT = 8
PHASE = PhaseSettings(
    method="per_frame",
    max_shift_px=MAX_SHIFT,
    consistency_window_frames=9,
    trust_window_frames=11,
    fallback_window_frames=31,
)


def frames_with_phases(rng, phases, noise_sd, anatomy=None):
    """One misaligned frame per entry of ``phases``: shared anatomy plus fresh noise."""
    anatomy = textured_image(rng) if anatomy is None else anatomy
    return np.stack(
        [
            misalign((anatomy + noise_sd * rng.standard_normal(anatomy.shape))[None], p)[0]
            for p in phases
        ]
    )


def noise_frames(rng, n):
    """Frames with no tissue: row-correlated stripes and pixel noise, no shared picture."""
    stripes = gaussian_filter(rng.standard_normal((n, 48, 1)), (0, 3, 0))
    return (stripes + rng.standard_normal((n, 48, 96))).astype(np.float32)


# ---- measuring one frame -------------------------------------------------------------


@pytest.mark.parametrize("true_phase", [-7, -2, 0, 1, 6])
def test_each_frame_gives_its_own_phase(rng, true_phase):
    frames = frames_with_phases(rng, [true_phase, 0, -true_phase], noise_sd=0.05)
    curves = line_correlation_curves(frames, MAX_SHIFT)
    assert curves.shape == (3, 2 * MAX_SHIFT + 1)
    assert list(best_shifts(curves, MAX_SHIFT)) == [true_phase, 0, -true_phase]


def test_the_measured_phase_undoes_the_error(rng):
    anatomy = textured_image(rng)
    frame = misalign(anatomy[None], 5)
    phase = best_shifts(line_correlation_curves(frame, MAX_SHIFT), MAX_SHIFT)[0]
    fixed = apply_bidiphase(frame, phase)[0]
    inner = slice(MAX_SHIFT, -MAX_SHIFT)
    np.testing.assert_allclose(fixed[:, inner], anatomy[:, inner], atol=1e-5)


def test_curves_ignore_per_row_brightness(rng):
    frame = frames_with_phases(rng, [3], noise_sd=0.05)
    offsets = rng.normal(0, 5, size=(1, frame.shape[1], 1))
    np.testing.assert_allclose(
        line_correlation_curves(frame + offsets, MAX_SHIFT),
        line_correlation_curves(frame, MAX_SHIFT),
        atol=1e-5,
    )


def test_a_flat_frame_has_no_phase():
    curves = line_correlation_curves(np.ones((1, 10, 20)), 4)
    assert np.all(np.isnan(curves))


def test_curves_reject_a_search_wider_than_the_frame():
    with pytest.raises(ValueError, match="max_shift_px"):
        line_correlation_curves(np.zeros((1, 10, 20)), 20)


# ---- consistency and windows ---------------------------------------------------------


def test_a_steady_phase_is_fully_consistent():
    assert np.all(local_consistency(np.full(50, 4), window=9, tolerance_px=2) == 1)


def test_a_scattered_phase_is_not(rng):
    scattered = rng.integers(-8, 9, size=500)
    assert local_consistency(scattered, window=9, tolerance_px=2).mean() < 0.5


def test_one_outlier_barely_lowers_consistency():
    phases = np.full(50, 4)
    phases[25] = -8
    consistency = local_consistency(phases, window=9, tolerance_px=2)
    assert consistency.min() == pytest.approx(8 / 9)


def test_a_long_window_finds_a_phase_single_frames_cannot(rng):
    frames = frames_with_phases(rng, [4] * 120, noise_sd=3.0)
    curves = line_correlation_curves(frames, MAX_SHIFT)
    assert np.mean(best_shifts(curves, MAX_SHIFT) == 4) < 0.6
    assert np.all(np.abs(windowed_best_shifts(curves, 61, MAX_SHIFT) - 4) <= 1)


# ---- choosing the phase of each frame -------------------------------------------------


def two_channels(rng, true_phases, structural_is_noise=None):
    """Structural: clean tissue, or no tissue where ``structural_is_noise``. Functional: noisy."""
    n = len(true_phases)
    structural = frames_with_phases(rng, true_phases, noise_sd=0.2)
    if structural_is_noise is not None:
        structural[structural_is_noise] = noise_frames(rng, int(np.sum(structural_is_noise)))
    functional = frames_with_phases(rng, true_phases, noise_sd=2.0)
    assert len(functional) == n
    return (
        line_correlation_curves(structural, MAX_SHIFT),
        line_correlation_curves(functional, MAX_SHIFT),
    )


def test_a_good_structural_channel_follows_a_jump_exactly(rng):
    true = np.r_[np.full(57, -2), np.full(83, 6)]  # the jump is not on a chunk boundary
    chosen = choose_frame_phases(*two_channels(rng, true), PHASE)
    assert chosen.used_structural.all()
    assert np.array_equal(chosen.phase_px, true)


def test_stretches_of_noise_use_the_functional_channel(rng):
    true = np.full(240, 5)
    noise = np.zeros(240, bool)
    noise[80:160] = True
    chosen = choose_frame_phases(*two_channels(rng, true, noise), PHASE)
    assert not chosen.used_structural[90:150].any()
    assert chosen.used_structural[:70].all() and chosen.used_structural[170:].all()
    assert np.all(np.abs(chosen.phase_px - true) <= 1)


def test_noise_that_settles_at_zero_is_rejected(rng):
    # A tissue-free structural channel that sits steadily at 0 px while the real phase,
    # seen in the functional channel, is well away from 0.
    wide = PHASE.model_copy(update={"max_shift_px": 16})
    structural = noise_frames(rng, 200)
    structural[:, 1::2] = structural[:, ::2]  # odd rows copy even rows: a steady 0 px
    functional = frames_with_phases(rng, np.full(200, 12), noise_sd=2.0)
    chosen = choose_frame_phases(
        line_correlation_curves(structural, 16), line_correlation_curves(functional, 16), wide
    )
    assert np.all(local_consistency(chosen.structural_px, 9, 2) == 1)  # it does look steady
    assert not chosen.used_structural.any()
    assert np.all(np.abs(chosen.phase_px - 12) <= 1)


def test_single_doubtful_frames_do_not_switch_channel(rng):
    true = np.full(120, 3)
    noise = np.zeros(120, bool)
    noise[[30, 70, 71]] = True
    chosen = choose_frame_phases(*two_channels(rng, true, noise), PHASE)
    assert chosen.used_structural.all()


def test_a_missing_structural_channel_uses_the_functional_one(rng):
    true = np.full(100, -3)
    _, functional = two_channels(rng, true)
    chosen = choose_frame_phases(None, functional, PHASE)
    assert not chosen.used_structural.any()
    assert np.all(chosen.phase_px == -3)


def test_the_record_holds_one_value_per_frame(rng):
    chosen = choose_frame_phases(*two_channels(rng, np.full(40, 2)), PHASE)
    for field in (
        "phase_px",
        "used_structural",
        "structural_px",
        "functional_px",
        "structural_consistency",
        "fallback_px",
        "structural_loss",
        "functional_loss",
    ):
        assert len(getattr(chosen, field)) == 40, field


# ---- applying and checking -----------------------------------------------------------


def test_each_frame_is_shifted_by_its_own_phase(rng):
    frames = rng.normal(size=(4, 6, 20)).astype(np.float32)
    phases = np.array([0, 2, -3, 2])
    shifted = apply_frame_phases(frames, phases)
    for i, p in enumerate(phases):
        np.testing.assert_array_equal(shifted[i], apply_bidiphase(frames[i : i + 1], p)[0])


def test_applying_phases_leaves_the_input_untouched(rng):
    frames = rng.integers(0, 100, size=(3, 6, 20)).astype(np.int16)
    before = frames.copy()
    apply_frame_phases(frames, np.array([1, 2, 3]))
    assert np.array_equal(frames, before)


def test_apply_needs_one_phase_per_frame(rng):
    with pytest.raises(ValueError, match="one phase per frame"):
        apply_frame_phases(np.zeros((3, 6, 20)), np.array([1, 2]))


def test_loss_is_zero_at_the_best_phase_and_grows_away_from_it(rng):
    curves = line_correlation_curves(frames_with_phases(rng, [3, 3], noise_sd=0.05), MAX_SHIFT)
    loss = phase_loss(curves, np.array([3, -5]), MAX_SHIFT)
    assert loss[0] == pytest.approx(0)
    assert loss[1] > 0.1


def test_fallback_stretches_are_half_open_ranges():
    used = np.array([1, 1, 0, 0, 0, 1, 0, 0, 1], bool)
    assert fallback_stretches(used) == [(2, 5), (6, 8)]
    assert fallback_stretches(np.ones(5, bool)) == []
    assert fallback_stretches(np.zeros(3, bool)) == [(0, 3)]


def test_blank_frames_do_not_count_as_a_zero_phase():
    values = np.array([7, 0, 7, 0, 7, 0, 7])
    valid = values != 0
    assert np.all(running_median(values, 5, valid) == 7)
    consistency = local_consistency(values, 3, 0, valid)
    assert np.all(consistency <= 2 / 3)  # blanks never count as agreeing


def test_a_window_of_blank_frames_has_no_median():
    assert np.all(np.isnan(running_median(np.zeros(5), 3, np.zeros(5, bool))))


def test_a_functional_channel_without_texture_leaves_structural_in_charge(rng):
    true = np.full(80, 7)
    structural, _ = two_channels(rng, true)
    blank = np.full_like(structural, np.nan)  # e.g. green PMT off: nothing to check against
    chosen = choose_frame_phases(structural, blank, PHASE)
    assert chosen.used_structural.all()
    assert np.all(chosen.phase_px == 7)


def test_without_a_structural_channel_the_loss_is_the_functional_one(rng):
    _, functional = two_channels(rng, np.full(60, 2))
    chosen = choose_frame_phases(None, functional, PHASE)
    assert not chosen.has_structural
    assert chosen.structural_px is None and chosen.structural_loss is None
    assert chosen.loss is chosen.functional_loss


def test_the_loss_is_zero_where_structural_fixed_the_frame(rng):
    chosen = choose_frame_phases(*two_channels(rng, np.r_[np.full(40, 1), np.full(40, -4)]), PHASE)
    assert np.allclose(chosen.structural_loss, 0, atol=1e-6)


def test_a_blank_frame_beside_a_jump_takes_one_side_of_it():
    # With a blank frame the window holds an even number of values; the median must still
    # be a value from the window, never the average of the two sides of a jump.
    phases = np.r_[np.full(50, -2), np.full(50, 6)]
    for blank in range(46, 54):
        valid = np.ones(100, bool)
        valid[blank] = False
        median = running_median(phases, 9, valid)
        assert set(np.unique(median)) <= {-2, 6}, blank


def test_the_median_of_an_even_count_is_one_of_the_values():
    median = running_median(np.array([-5, -4, -4, -5]), 3, np.array([1, 1, 0, 1], bool))
    assert set(median) <= {-5, -4}

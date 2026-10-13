import numpy as np
import pytest

from mini2p_preproc.bidiphase import (
    adjacent_line_score,
    apply_bidiphase,
    estimate_phase,
    select_alignment,
)
from mini2p_preproc.config import ChannelSettings, PhaseSettings
from tests.conftest import misalign, textured_image

PHASE = PhaseSettings(max_shift_px=8)


def test_apply_bidiphase_leaves_input_untouched(rng):
    raw = rng.integers(0, 100, size=(2, 6, 10)).astype(np.int16)
    before = raw.copy()
    shifted = apply_bidiphase(raw, 2)
    assert np.array_equal(raw, before)
    assert shifted.dtype == np.float32
    assert np.array_equal(shifted[:, ::2], raw[:, ::2])  # even rows never move


@pytest.mark.parametrize("true_phase", [-6, -1, 0, 3, 7])
def test_wide_estimator_recovers_known_phase(rng, true_phase):
    frames = misalign(np.repeat(textured_image(rng)[None], 5, axis=0), true_phase)
    estimate = estimate_phase(frames, PHASE)
    assert estimate.phase_px == true_phase
    assert estimate.peak_score > 0.9


def test_score_ignores_per_row_brightness_offsets(rng):
    image = textured_image(rng)
    offsets = rng.normal(0, 5, size=(image.shape[0], 1))
    assert adjacent_line_score(image + offsets, 4) == pytest.approx(adjacent_line_score(image, 4))


def test_score_is_minus_inf_for_flat_image():
    assert adjacent_line_score(np.ones((10, 20)), 2) == -np.inf


def test_score_rejects_crop_wider_than_image():
    with pytest.raises(ValueError, match="too large"):
        adjacent_line_score(np.zeros((10, 20)), 9)


def test_boundary_flag_when_phase_near_search_limit(rng):
    frames = misalign(np.repeat(textured_image(rng)[None], 3, axis=0), 7)
    assert estimate_phase(frames, PHASE).at_boundary  # 8 - 7 <= 2
    frames = misalign(np.repeat(textured_image(rng)[None], 3, axis=0), 2)
    assert not estimate_phase(frames, PHASE).at_boundary


def test_estimate_rejects_2d_input():
    with pytest.raises(ValueError, match="frames x height x width"):
        estimate_phase(np.zeros((10, 20)), PHASE)


def _two_channel_chunks(rng, red_noise, green_noise, true_phase=3):
    anatomy = textured_image(rng)
    chunks = []
    for noise in (red_noise, green_noise):
        frames = anatomy + noise * rng.standard_normal((4, *anatomy.shape))
        chunks.append(misalign(frames, true_phase))
    return chunks  # 0-based order: channel 1 red, channel 2 green


def test_best_score_prefers_structural_unless_functional_clearly_wins(rng):
    channels = ChannelSettings()
    best = PhaseSettings(max_shift_px=8, alignment_source="best_score")
    clean_red = _two_channel_chunks(rng, red_noise=0.1, green_noise=2.0)
    assert select_alignment(clean_red, best, channels).selected_source == "structural"

    clean_green = _two_channel_chunks(rng, red_noise=2.0, green_noise=0.1)
    choice = select_alignment(clean_green, best, channels)
    assert choice.selected_source == "functional"
    assert choice.selected_channel_zero_based == 1
    assert set(choice.score_by_source) == {"structural", "functional"}

    huge_margin = best.model_copy(update={"score_switch_margin": 5.0})
    assert select_alignment(clean_green, huge_margin, channels).selected_source == "structural"


def test_fixed_source_scores_only_that_channel(rng):
    chunks = _two_channel_chunks(rng, 0.1, 0.1)
    choice = select_alignment(chunks, PHASE, ChannelSettings())
    assert choice.selected_source == "functional"
    assert list(choice.score_by_source) == ["functional"]


def test_select_alignment_checks_channel_count(rng):
    with pytest.raises(ValueError, match="Expected 2 channel chunks"):
        select_alignment([np.zeros((2, 4, 20))], PHASE, ChannelSettings())

"""Per-frame focus QC on synthetic registered frames with a known out-of-focus stretch."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from mini2p_preprocessing.config import FocusSettings
from mini2p_preprocessing.focus import (
    assess_focus,
    flagged_stretches,
    frame_match,
    relative_to_baseline,
    smooth_frames,
)

SETTINGS = FocusSettings(smooth_px=1.0, window_frames=41, drop_threshold=4.0, merge_gap_frames=3)


def scene(rng, shape=(40, 50)) -> np.ndarray:
    """Blobs on a background, like cells in the structural channel."""
    return 100 + 40 * gaussian_filter(rng.standard_normal(shape), 1.5)


def recording(rng, n_frames=300, defocused=slice(120, 140)):
    """Registered frames: the scene plus noise, blurred and dimmed in one stretch."""
    image = scene(rng)
    frames = np.repeat(image[None], n_frames, axis=0)
    frames[defocused] = 0.8 * gaussian_filter(image, 3)
    return frames + rng.normal(0, 4, frames.shape)


def test_frame_match_is_one_for_scaled_copies_and_lower_for_noise():
    rng = np.random.default_rng(0)
    image = scene(rng)
    frames = np.stack([image, 2 * image + 5, image + rng.normal(0, 20, image.shape)])
    match = frame_match(frames, image)
    np.testing.assert_allclose(match[:2], 1.0, atol=1e-6)
    assert 0 < match[2] < 0.9


def test_frame_match_refuses_flat_images_and_mismatched_shapes():
    rng = np.random.default_rng(1)
    image = scene(rng)
    with pytest.raises(ValueError, match="reference"):
        frame_match(image[None], np.ones_like(image))
    with pytest.raises(ValueError, match=r"frame\(s\) \[1\]"):
        frame_match(np.stack([image, np.ones_like(image)]), image)
    with pytest.raises(ValueError, match="shape"):
        frame_match(image[None], image[:-1])


def test_smoothing_is_spatial_only():
    frames = np.zeros((3, 9, 9), np.float32)
    frames[1, 4, 4] = 1.0
    smoothed = smooth_frames(frames, 1.0)
    assert smoothed[0].max() == 0 and smoothed[2].max() == 0
    assert 0 < smoothed[1, 4, 4] < 1


def test_relative_to_baseline_ignores_short_dips():
    values = np.full(101, 10.0)
    values[50:53] = 5.0
    relative = relative_to_baseline(values, window_frames=21)
    np.testing.assert_allclose(relative[:50], 1.0)
    np.testing.assert_allclose(relative[50:53], 0.5)


def test_flags_the_defocused_stretch_only():
    rng = np.random.default_rng(2)
    frames = recording(rng)
    reference = frames.mean(axis=0)
    smoothed = smooth_frames(frames, SETTINGS.smooth_px)
    match = frame_match(smoothed, smooth_frames(reference[None], SETTINGS.smooth_px)[0])
    result = assess_focus(match, frames.mean(axis=(1, 2)), SETTINGS)
    assert flagged_stretches(result.flagged, SETTINGS.merge_gap_frames) == [(120, 140)]
    assert np.all(result.brightness[120:140] < 0.85)
    assert np.all(result.drop[120:140] > SETTINGS.drop_threshold)


def test_nearby_flagged_frames_merge_into_one_stretch():
    flagged = np.zeros(30, bool)
    flagged[[5, 6, 9, 20]] = True
    assert flagged_stretches(flagged, merge_gap_frames=2) == [(5, 10), (20, 21)]
    assert flagged_stretches(flagged, merge_gap_frames=0) == [(5, 7), (9, 10), (20, 21)]
    assert flagged_stretches(np.zeros(5, bool), merge_gap_frames=2) == []


def test_assess_focus_fails_loudly_on_bad_input():
    with pytest.raises(ValueError, match="same length"):
        assess_focus(np.ones(10), np.ones(9), SETTINGS)
    with pytest.raises(ValueError, match="no spread"):
        assess_focus(np.ones(100), np.ones(100), SETTINGS)
    with pytest.raises(ValueError, match="finite"):
        assess_focus(np.r_[np.nan, np.linspace(0, 1, 99)], np.ones(100), SETTINGS)


def registered(rng, n_frames=300):
    """RegisteredOutputs-like stand-in: structural frames with a defocused stretch, junk in the
    rows frame 10's 6-px shift wraps into, and functional frames that are pure noise."""
    from mini2p_preprocessing.registration import RegisteredOutputs

    structural = recording(rng, n_frames)
    structural[:30, -6:] = rng.normal(0, 200, (30, 6, structural.shape[2]))
    functional = rng.normal(100, 5, structural.shape)
    yoff = np.zeros(n_frames)
    yoff[10] = 6
    ops = {
        "yoff": yoff,
        "xoff": np.zeros(n_frames),
        "Ly": structural.shape[1],
        "Lx": structural.shape[2],
        "meanImg": functional.mean(0),
        "meanImg_chan2": structural.mean(0),
    }
    return RegisteredOutputs(ops=ops, functional=functional, structural=structural)


def test_focus_qc_reads_the_chosen_channel_inside_the_unwrapped_region():
    from mini2p_preprocessing.config import PipelineSettings
    from mini2p_preprocessing.qc import focus_qc

    outputs = registered(np.random.default_rng(3))
    settings = PipelineSettings.model_validate(
        {"distortion": {"mode": "off"}, "focus": SETTINGS.model_dump() | {"chunk_frames": 64}}
    )
    result = focus_qc(outputs, settings)
    assert len(result.match) == 300
    # The junk rows are cropped away, so only the defocused stretch is flagged.
    assert flagged_stretches(result.flagged, SETTINGS.merge_gap_frames) == [(120, 140)]
    functional = settings.model_copy(
        update={"focus": settings.focus.model_copy(update={"channel": "functional"})}
    )
    assert np.all(np.abs(focus_qc(outputs, functional).match) < 0.5)


def test_focus_plot_draws_and_checks_frame_counts():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mini2p_preprocessing.plots import plot_focus_qc

    rng = np.random.default_rng(4)
    frames = recording(rng)
    match = frame_match(smooth_frames(frames, 1.0), smooth_frames(frames.mean(0)[None], 1.0)[0])
    result = assess_focus(match, frames.mean(axis=(1, 2)), SETTINGS)
    ops = {"yoff": np.zeros(300), "xoff": np.zeros(300)}
    axes = plot_focus_qc(result, 7.52, ops, SETTINGS.merge_gap_frames)
    assert len(axes) == 3 and len(axes[0].patches) == 1  # one shaded stretch
    with pytest.raises(ValueError, match="299 frames"):
        plot_focus_qc(result, 7.52, {"yoff": np.zeros(299), "xoff": np.zeros(299)}, 8)
    plt.close("all")

"""Per-frame odd/even scan-phase: measure every frame, then choose a channel stretch by stretch.

The structural channel usually has enough line texture to give each frame's phase, so it
follows jumps exactly. Where it has no tissue (pure noise, which may come and go within a
session) its per-frame phases scatter, and those frames use the functional channel summed
over a longer window instead. Pure functions only; ``correction.py`` streams the TIFF.

Shifts follow Suite2p's convention (``bidiphase.apply_bidiphase``): a positive phase moves
odd rows towards higher column indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .bidiphase import apply_bidiphase
from .config import PhaseSettings


@dataclass(frozen=True)
class FramePhases:
    """The applied phase of every frame and the evidence it was chosen from."""

    phase_px: np.ndarray  # applied odd-row shift per frame
    used_structural: np.ndarray  # True where the structural channel set the phase
    structural_px: np.ndarray  # best shift of each single structural frame (0 if absent)
    functional_px: np.ndarray  # best shift of each single functional frame
    structural_consistency: np.ndarray  # fraction of neighbouring frames that agree
    fallback_px: np.ndarray  # functional phase over phase.fallback_window_frames


# ---- measuring ---------------------------------------------------------------------


def line_correlation_curves(frames: np.ndarray, max_shift_px: int) -> np.ndarray:
    """Even/odd row correlation of each frame for every odd-row shift, (frames, 2*max+1).

    Column ``max_shift_px + p`` is the correlation after shifting odd rows by ``p``. Each
    row's mean is removed first, so line brightness and stripe offsets do not count. A
    frame with no row texture gets NaN throughout.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1] < 2:
        raise ValueError(f"Expected frames x height x width with >= 2 rows, got {frames.shape}")
    width = frames.shape[2]
    if 2 * max_shift_px >= width:
        raise ValueError(f"max_shift_px={max_shift_px} is too large for image width {width}")
    n_pairs = frames.shape[1] // 2
    even, odd = frames[:, 0 : 2 * n_pairs : 2], frames[:, 1 : 2 * n_pairs : 2]
    even = even - even.mean(axis=2, keepdims=True)
    odd = odd - odd.mean(axis=2, keepdims=True)
    # Zero-padded FFT cross-correlation along each row, summed over row pairs:
    # sum_x even[x] * odd[x - p], i.e. the overlap after odd rows move by +p.
    n_fft = 2 * width
    spectrum = np.fft.rfft(even, n_fft) * np.conj(np.fft.rfft(odd, n_fft))
    correlation = np.fft.irfft(spectrum.sum(axis=1), n_fft)
    shifts = np.arange(-max_shift_px, max_shift_px + 1)
    norm = np.sqrt(np.sum(even**2, axis=(1, 2)) * np.sum(odd**2, axis=(1, 2)))
    with np.errstate(invalid="ignore", divide="ignore"):
        curves = correlation[:, shifts % n_fft] / norm[:, None]
    curves[norm == 0] = np.nan
    return curves.astype(np.float32)


def best_shifts(curves: np.ndarray, max_shift_px: int) -> np.ndarray:
    """Shift with the highest correlation in each row of ``curves``.

    A row that is all NaN (no texture) gets 0: with nothing to measure, leave it unshifted.
    """
    curves = np.asarray(curves)
    textured = ~np.all(np.isnan(curves), axis=1)
    best = np.zeros(len(curves), dtype=int)
    best[textured] = np.nanargmax(curves[textured], axis=1) - max_shift_px
    return best


def windowed_best_shifts(curves: np.ndarray, window: int, max_shift_px: int) -> np.ndarray:
    """Best shift of the curves averaged over ``window`` frames centred on each frame."""
    textured = ~np.all(np.isnan(curves), axis=1)
    summed = uniform_filter1d(np.nan_to_num(curves), window, axis=0, mode="nearest")
    has_texture = uniform_filter1d(textured.astype(float), window, mode="nearest") > 0
    summed[~has_texture] = np.nan  # a window with no texture at all has no phase
    return best_shifts(summed, max_shift_px)


def local_consistency(phases: np.ndarray, window: int, tolerance_px: int) -> np.ndarray:
    """Fraction of the ``window`` frames around each frame within ``tolerance_px`` of their
    running median: about 1 where a channel's per-frame phase is real, low where it is noise."""
    phases = np.asarray(phases)
    running_median = median_filter(phases, window, mode="nearest")
    near = (np.abs(phases - running_median) <= tolerance_px).astype(float)
    return uniform_filter1d(near, window, mode="nearest")


# ---- choosing ----------------------------------------------------------------------


def choose_frame_phases(
    structural_curves: np.ndarray | None, functional_curves: np.ndarray, phase: PhaseSettings
) -> FramePhases:
    """Each frame's phase: smoothed structural where it is trustworthy, else functional.

    Structural is trusted where it is self-consistent *and* agrees with the functional
    channel's long-window phase; noise can sit steadily near 0 px, so consistency alone
    is not enough. A majority vote over ``trust_window_frames`` keeps single doubtful
    frames from switching channel. ``structural_curves=None`` uses functional throughout.
    """
    max_shift = phase.max_shift_px
    functional_px = best_shifts(functional_curves, max_shift)
    fallback = windowed_best_shifts(functional_curves, phase.fallback_window_frames, max_shift)
    n_frames = len(functional_px)
    if structural_curves is None:
        zeros = np.zeros(n_frames, dtype=int)
        return FramePhases(
            fallback, zeros.astype(bool), zeros, functional_px, zeros * 0.0, fallback
        )
    if len(structural_curves) != n_frames:
        raise ValueError(f"{len(structural_curves)} structural vs {n_frames} functional frames")
    structural_px = best_shifts(structural_curves, max_shift)
    window = phase.consistency_window_frames
    smoothed = median_filter(structural_px, window, mode="nearest")
    consistency = local_consistency(structural_px, window, phase.consistency_tolerance_px)
    textured = ~np.all(np.isnan(structural_curves), axis=1)
    candidate = (
        (consistency >= phase.consistency_threshold)
        & (np.abs(smoothed - fallback) <= phase.agreement_px)
        & textured
    )
    trusted = (
        uniform_filter1d(candidate.astype(float), phase.trust_window_frames, mode="nearest") > 0.5
    )
    return FramePhases(
        phase_px=np.where(trusted, smoothed, fallback),
        used_structural=trusted,
        structural_px=structural_px,
        functional_px=functional_px,
        structural_consistency=consistency,
        fallback_px=fallback,
    )


# ---- applying and checking ---------------------------------------------------------


def apply_frame_phases(frames: np.ndarray, phases_px: np.ndarray) -> np.ndarray:
    """Float32 copy of ``frames`` with each frame's odd rows shifted by its own phase."""
    phases_px = np.asarray(phases_px, dtype=int)
    if len(phases_px) != len(frames):
        raise ValueError(f"Need one phase per frame: {len(phases_px)} for {len(frames)} frames")
    shifted = np.asarray(frames).astype(np.float32, copy=True)
    for phase in np.unique(phases_px):
        same = phases_px == phase
        shifted[same] = apply_bidiphase(shifted[same], int(phase))
    return shifted


def phase_loss(curves: np.ndarray, applied_px: np.ndarray, max_shift_px: int) -> np.ndarray:
    """Best achievable correlation minus the one at the applied shift (0 = fully fixed)."""
    applied_px = np.asarray(applied_px, dtype=int)
    if np.any(np.abs(applied_px) > max_shift_px):
        raise ValueError(f"Applied phases outside the searched ±{max_shift_px} px")
    at_applied = curves[np.arange(len(curves)), applied_px + max_shift_px]
    with np.errstate(invalid="ignore"):
        return np.nanmax(curves, axis=1, initial=-np.inf, where=~np.isnan(curves)) - at_applied


def fallback_stretches(used_structural: np.ndarray) -> list[tuple[int, int]]:
    """Half-open (start, end) frame ranges where the functional channel set the phase."""
    edges = np.flatnonzero(np.diff(np.r_[0, ~np.asarray(used_structural, bool), 0].astype(int)))
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2], strict=True)]

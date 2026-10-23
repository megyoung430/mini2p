"""Per-frame odd/even scan-phase: measure every frame, then choose a channel stretch by stretch.

The structural channel usually has enough line texture to give each frame's phase, so it
follows jumps exactly. Where it has no tissue (pure noise, which may come and go within a
session) its per-frame phases scatter, and those frames use the functional channel summed
over a longer window instead. Pure functions only; ``correction.py`` streams the TIFF.

Shifts follow Suite2p's convention (``bidiphase.apply_bidiphase``): a positive phase moves
odd rows towards higher column indices.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from .bidiphase import apply_bidiphase
from .config import PhaseSettings


@dataclass(frozen=True)
class FramePhases:
    """The applied phase of every frame and the evidence it was chosen from.

    The ``structural_*`` fields are None when there is no separate structural channel.
    """

    phase_px: np.ndarray  # applied odd-row shift per frame
    used_structural: np.ndarray  # True where the structural channel set the phase
    functional_px: np.ndarray  # best shift of each single functional frame
    fallback_px: np.ndarray  # functional phase over phase.fallback_window_frames
    functional_loss: np.ndarray  # functional correlation lost at the applied phase
    structural_px: np.ndarray | None = None  # best shift of each single structural frame
    structural_consistency: np.ndarray | None = None  # fraction of neighbours that agree
    structural_loss: np.ndarray | None = None  # structural correlation lost at applied phase

    @property
    def has_structural(self) -> bool:
        return self.structural_px is not None

    @property
    def loss(self) -> np.ndarray:
        """Correlation lost in the channel with more texture: structural if there is one."""
        return self.structural_loss if self.has_structural else self.functional_loss


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


def has_texture(curves: np.ndarray) -> np.ndarray:
    """True for each frame whose curve is not all NaN (it had row texture to measure)."""
    return ~np.all(np.isnan(np.asarray(curves)), axis=1)


def windowed_texture(curves: np.ndarray, window: int) -> np.ndarray:
    """True where at least one frame in the ``window`` around a frame had texture."""
    return uniform_filter1d(has_texture(curves).astype(float), window, mode="nearest") > 0


def windowed_best_shifts(curves: np.ndarray, window: int, max_shift_px: int) -> np.ndarray:
    """Best shift of the curves averaged over ``window`` frames centred on each frame.

    A window with no texture at all gets 0 (see ``best_shifts``); ``windowed_texture``
    says where that happened.
    """
    summed = uniform_filter1d(np.nan_to_num(curves), window, axis=0, mode="nearest")
    summed[~windowed_texture(curves, window)] = np.nan
    return best_shifts(summed, max_shift_px)


def running_median(values: np.ndarray, window: int, valid: np.ndarray | None = None) -> np.ndarray:
    """Median of the valid values in the ``window`` around each frame (edges repeat the end
    values); NaN where the window holds no valid value.

    With an even number of valid values the lower middle one is taken, so the result is
    always a value from the window: averaging the two would invent a phase between the
    two sides of a jump.
    """
    values = np.asarray(values, dtype=float)
    if valid is not None:
        values = np.where(valid, values, np.nan)
    half = window // 2
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(values, half, mode="edge"), window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows are expected
        return np.nanpercentile(windows, 50, axis=1, method="lower")


def local_consistency(
    phases: np.ndarray, window: int, tolerance_px: int, valid: np.ndarray | None = None
) -> np.ndarray:
    """Fraction of the ``window`` frames around each frame within ``tolerance_px`` of their
    running median: about 1 where a channel's per-frame phase is real, low where it is noise.

    Frames that are not ``valid`` (no texture) neither count as agreeing nor shift the median.
    """
    phases = np.asarray(phases)
    valid = np.ones(len(phases), bool) if valid is None else np.asarray(valid, bool)
    median = running_median(phases, window, valid)
    with np.errstate(invalid="ignore"):
        near = valid & (np.abs(phases - median) <= tolerance_px)
    return uniform_filter1d(near.astype(float), window, mode="nearest")


# ---- choosing ----------------------------------------------------------------------


def trusted_structural(
    structural_px: np.ndarray,
    structural_textured: np.ndarray,
    fallback_px: np.ndarray,
    fallback_textured: np.ndarray,
    phase: PhaseSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(trusted, smoothed structural phase, consistency) for every frame.

    A frame is a candidate where the structural phase is self-consistent *and* its running
    median agrees with the functional long-window phase; noise can sit steadily near 0 px,
    so consistency alone is not enough. Where the functional channel has no texture at all
    there is nothing to check against, and consistency decides alone. Each frame then
    follows the majority of the ``trust_window_frames`` around it.
    """
    window = phase.consistency_window_frames
    smoothed = running_median(structural_px, window, structural_textured)
    consistency = local_consistency(
        structural_px, window, phase.consistency_tolerance_px, structural_textured
    )
    with np.errstate(invalid="ignore"):
        agrees = np.abs(smoothed - fallback_px) <= phase.agreement_px
    candidate = (
        (consistency >= phase.consistency_threshold)
        & structural_textured
        & ~np.isnan(smoothed)
        & (agrees | ~fallback_textured)
    )
    vote = uniform_filter1d(candidate.astype(float), phase.trust_window_frames, mode="nearest")
    trusted = (vote > 0.5) & ~np.isnan(smoothed)
    return trusted, np.nan_to_num(smoothed).astype(int), consistency


def choose_frame_phases(
    structural_curves: np.ndarray | None, functional_curves: np.ndarray, phase: PhaseSettings
) -> FramePhases:
    """Each frame's phase: smoothed structural where it is trustworthy (``trusted_structural``),
    else the functional channel over ``fallback_window_frames``. ``structural_curves=None``
    (no separate structural channel) uses the functional channel throughout."""
    max_shift = phase.max_shift_px
    functional_px = best_shifts(functional_curves, max_shift)
    window = phase.fallback_window_frames
    fallback = windowed_best_shifts(functional_curves, window, max_shift)
    if structural_curves is None:
        return FramePhases(
            phase_px=fallback,
            used_structural=np.zeros(len(fallback), bool),
            functional_px=functional_px,
            fallback_px=fallback,
            functional_loss=phase_loss(functional_curves, fallback, max_shift),
        )
    if len(structural_curves) != len(functional_curves):
        raise ValueError(
            f"{len(structural_curves)} structural vs {len(functional_curves)} functional frames"
        )
    structural_px = best_shifts(structural_curves, max_shift)
    trusted, smoothed, consistency = trusted_structural(
        structural_px,
        has_texture(structural_curves),
        fallback,
        windowed_texture(functional_curves, window),
        phase,
    )
    applied = np.where(trusted, smoothed, fallback)
    return FramePhases(
        phase_px=applied,
        used_structural=trusted,
        functional_px=functional_px,
        fallback_px=fallback,
        functional_loss=phase_loss(functional_curves, applied, max_shift),
        structural_px=structural_px,
        structural_consistency=consistency,
        structural_loss=phase_loss(structural_curves, applied, max_shift),
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

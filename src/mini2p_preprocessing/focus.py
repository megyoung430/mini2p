"""Per-frame focus QC: how well each registered frame matches the session's mean image.

A frame that matches the mean image worse than its neighbours do was not made to match by
the shift registration applied to it. Three causes give the same low score: the imaged
plane moved in z; the brain moved while the frame was being scanned (at ~7.5 Hz a frame
takes ~130 ms, so a fast movement stretches or shears it); or registration chose the wrong
shift. The score alone cannot tell them apart, and the rigid shifts only partly can: one
shift per frame hides movement during the scan, and z-motion often comes with x/y motion.

The match is a correlation on spatially smoothed frames, so shot noise does not dominate
it, and is compared with its own running median, so changes slower than the window
(bleaching, expression, and also slow z-drift) are not flagged; nor is the middle of a bad
stretch longer than about half the window, which becomes the median itself. The spread is
one robust SD for the whole session, so a noisier half of a session is flagged more often.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import FocusSettings

MAD_TO_SD = 1.4826  # median absolute deviation -> standard deviation for normal data


@dataclass(frozen=True)
class FocusQC:
    match: np.ndarray  # correlation of each frame with the mean image
    baseline: np.ndarray  # running median of ``match``
    # (baseline - match) in robust SDs; large = matches the mean worse than its neighbours
    drop: np.ndarray
    brightness: np.ndarray  # frame mean / its running median
    flagged: np.ndarray  # drop > settings.drop_threshold


def smooth_frames(frames: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian-smooth each (Ly, Lx) frame of a (frames, Ly, Lx) stack, not across time."""
    return gaussian_filter(np.asarray(frames, np.float32), sigma=(0, sigma_px, sigma_px))


def frame_match(frames: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Pearson correlation of each frame with ``reference``, over all pixels.

    Works on one float64 copy of ``frames`` (8 bytes per pixel per frame), centred in place.
    """
    reference = np.asarray(reference, np.float64)
    if np.shape(frames)[1:] != reference.shape:
        raise ValueError(f"Frames of shape {np.shape(frames)[1:]} vs reference {reference.shape}")
    ref = reference - reference.mean()
    if not np.any(ref):
        raise ValueError("The reference image is flat, so nothing can match it")
    centred = np.array(frames, np.float64)
    not_finite = np.flatnonzero(~np.isfinite(centred).all(axis=(1, 2)))
    if not_finite.size:
        raise ValueError(f"Frame(s) {not_finite[:10].tolist()} have values that are not finite")
    centred -= centred.mean(axis=(1, 2), keepdims=True)
    norms = np.sqrt(np.einsum("fij,fij->f", centred, centred))
    flat = np.flatnonzero(norms == 0)
    if flat.size:
        raise ValueError(f"Flat frame(s) {flat[:10].tolist()} cannot be correlated")
    return np.einsum("fij,ij->f", centred, ref) / (norms * np.sqrt(np.sum(ref**2)))


def running_median(values: np.ndarray, window_frames: int) -> np.ndarray:
    """Median of the ``window_frames`` frames centred on each frame (odd window).

    Frames within half a window of either end take the median of the first / last full
    window. Padding the ends with copies of the edge value instead would let a bad first or
    last stretch become its own baseline and never be flagged. A recording shorter than the
    window gets one median over all of it.
    """
    values = np.asarray(values, np.float64)
    if window_frames % 2 == 0:
        raise ValueError(f"window_frames must be odd, got {window_frames}")
    if values.size == 0:
        raise ValueError("No frames to take a running median of")
    if values.size <= window_frames:
        return np.full_like(values, np.median(values))
    half = window_frames // 2
    centred = np.median(np.lib.stride_tricks.sliding_window_view(values, window_frames), axis=1)
    return np.concatenate([np.full(half, centred[0]), centred, np.full(half, centred[-1])])


def relative_to_baseline(values: np.ndarray, window_frames: int) -> np.ndarray:
    """``values`` divided by their running median: 1 = typical for that part of the session."""
    baseline = running_median(values, window_frames)
    not_positive = np.flatnonzero(baseline <= 0)
    if not_positive.size:
        raise ValueError(
            f"The running median brightness is not positive at frame(s) "
            f"{not_positive[:10].tolist()}, so a ratio to it means nothing; is the binary "
            "offset-subtracted?"
        )
    return np.asarray(values, np.float64) / baseline


def assess_focus(match: np.ndarray, brightness: np.ndarray, settings: FocusSettings) -> FocusQC:
    """Flag frames whose match falls far below the running median of their neighbours.

    ``brightness`` is each frame's mean intensity, in the same channel and region.
    """
    match, brightness = np.asarray(match, np.float64), np.asarray(brightness, np.float64)
    if match.shape != brightness.shape:
        raise ValueError(
            f"match and brightness must be the same length: {match.shape} vs {brightness.shape}"
        )
    bad = np.flatnonzero(~(np.isfinite(match) & np.isfinite(brightness)))
    if bad.size:
        raise ValueError(
            f"match and brightness must be finite; not at frame(s) {bad[:10].tolist()}"
        )
    baseline = running_median(match, settings.window_frames)
    residual = baseline - match
    spread = MAD_TO_SD * np.median(np.abs(residual - np.median(residual)))
    if spread == 0:
        raise ValueError("The frame match has no spread around its running median")
    drop = residual / spread
    return FocusQC(
        match=match,
        baseline=baseline,
        drop=drop,
        brightness=relative_to_baseline(brightness, settings.window_frames),
        flagged=drop > settings.drop_threshold,
    )


def flagged_stretches(flagged: np.ndarray, merge_gap_frames: int) -> list[tuple[int, int]]:
    """Half-open (start, end) runs of flagged frames, joining runs at most this far apart."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(flagged, bool), 0].astype(int)))
    runs = [[int(s), int(e)] for s, e in zip(edges[::2], edges[1::2], strict=True)]
    merged: list[list[int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= merge_gap_frames:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]

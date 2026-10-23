"""Per-frame focus QC: how well each registered frame matches the session's mean image.

Rigid registration removes whole-frame x/y shifts only. A frame that still matches the mean
image worse than its neighbours do has changed in a way a shift cannot undo: the imaged
plane moved in z, or the brain moved while the frame was being scanned (at ~7.5 Hz a frame
takes ~130 ms, so a fast movement stretches or shears it). The rigid shifts tell the two
apart only partly: flagged frames during large, fast excursions are often both.

The match is a correlation on spatially smoothed frames, so shot noise does not dominate
it, and is compared with its own running median, so slow changes over the session
(bleaching, drift in expression) are not flagged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from .config import FocusSettings

MAD_TO_SD = 1.4826  # median absolute deviation -> standard deviation for normal data


@dataclass(frozen=True)
class FocusQC:
    match: np.ndarray  # correlation of each frame with the mean image
    baseline: np.ndarray  # running median of ``match``
    drop: np.ndarray  # (baseline - match) in robust SDs; large = out of focus
    brightness: np.ndarray  # frame mean / its running median
    flagged: np.ndarray  # drop > settings.drop_threshold


def smooth_frames(frames: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian-smooth each (Ly, Lx) frame of a (frames, Ly, Lx) stack, not across time."""
    return gaussian_filter(np.asarray(frames, np.float32), sigma=(0, sigma_px, sigma_px))


def frame_match(frames: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Pearson correlation of each frame with ``reference``, over all pixels."""
    frames = np.asarray(frames, np.float64)
    reference = np.asarray(reference, np.float64)
    if frames.shape[1:] != reference.shape:
        raise ValueError(f"Frames of shape {frames.shape[1:]} vs reference {reference.shape}")
    ref = reference - reference.mean()
    if not np.any(ref):
        raise ValueError("The reference image is flat, so nothing can match it")
    centred = frames - frames.mean(axis=(1, 2), keepdims=True)
    norms = np.sqrt(np.sum(centred**2, axis=(1, 2)))
    flat = np.flatnonzero(norms == 0)
    if flat.size:
        raise ValueError(f"Flat frame(s) {flat[:10].tolist()} cannot be correlated")
    return np.tensordot(centred, ref, axes=([1, 2], [0, 1])) / (norms * np.sqrt(np.sum(ref**2)))


def running_median(values: np.ndarray, window_frames: int) -> np.ndarray:
    return median_filter(np.asarray(values, np.float64), size=window_frames, mode="nearest")


def relative_to_baseline(values: np.ndarray, window_frames: int) -> np.ndarray:
    """``values`` divided by their running median: 1 = typical for that part of the session."""
    return np.asarray(values, np.float64) / running_median(values, window_frames)


def assess_focus(match: np.ndarray, brightness: np.ndarray, settings: FocusSettings) -> FocusQC:
    """Flag frames whose match falls far below the running median of their neighbours.

    ``brightness`` is each frame's mean intensity, in the same channel and region.
    """
    match, brightness = np.asarray(match, np.float64), np.asarray(brightness, np.float64)
    if match.shape != brightness.shape:
        raise ValueError(
            f"match and brightness must be the same length: {match.shape} vs {brightness.shape}"
        )
    if not (np.all(np.isfinite(match)) and np.all(np.isfinite(brightness))):
        raise ValueError("match and brightness must be finite")
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

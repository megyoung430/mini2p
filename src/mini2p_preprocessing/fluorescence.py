"""Neuropil correction, F0 baseline and ΔF/F0 (pure functions, no file access).

Suite2p's ``preprocess`` returns the baseline-subtracted trace ΔF = Fcorr - F0, not
ΔF/F0. Here F0 is recovered as Fcorr - ΔF so every step can be inspected, and
ΔF/F0 is formed explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import medfilt
from suite2p.extraction import preprocess as s2p_preprocess

from .config import FluorescenceSettings


@dataclass(frozen=True)
class DffResult:
    """Every intermediate of ΔF/F0, each shaped (n_rois, n_frames)."""

    f_corr: np.ndarray  # neuropil-corrected fluorescence
    f0: np.ndarray  # baseline
    delta_f: np.ndarray  # f_corr - f0
    dff: np.ndarray  # delta_f / f0; NaN where invalid_f0
    invalid_f0: np.ndarray  # True where f0 <= min_f0


def neuropil_correct(f: np.ndarray, fneu: np.ndarray, coefficient: float) -> np.ndarray:
    _check_same_shape(f, fneu)
    _check_finite(f, fneu)
    return np.asarray(f, dtype=np.float32) - coefficient * np.asarray(fneu, dtype=np.float32)


def select_frames(traces: np.ndarray, frames: np.ndarray | None) -> np.ndarray:
    """Keep the frames given as a boolean mask or as strictly increasing indices.

    None keeps every frame. Integer selections must increase strictly, which also
    rejects a 0/1 mask stored as integers (convert it with ``.astype(bool)``) and
    keeps frames in time order for the baseline.
    """
    if frames is None:
        return traces
    frames = np.asarray(frames)
    n_frames = traces.shape[-1]
    if frames.dtype == bool:
        if frames.shape != (n_frames,):
            raise ValueError(f"Frame mask has {frames.shape} entries for {n_frames} frames")
        if not frames.any():
            raise ValueError("Frame mask selects no frames")
        return traces[..., frames]
    if frames.size == 0 or not np.issubdtype(frames.dtype, np.integer):
        raise ValueError(
            "Frame selection must be a non-empty boolean mask or integer frames, "
            f"got {frames.dtype} with {frames.size} entries"
        )
    if frames.ndim != 1 or np.any(np.diff(frames) <= 0):
        raise ValueError(
            "Integer frame selections must be strictly increasing indices; "
            "a 0/1 mask stored as integers needs .astype(bool)"
        )
    if frames.size and (frames[0] < 0 or frames[-1] >= n_frames):
        raise ValueError(f"Frame indices span {frames[0]}..{frames[-1]} for {n_frames} frames")
    return traces[..., frames]


def baseline(f_corr: np.ndarray, fs_hz: float, settings: FluorescenceSettings) -> np.ndarray:
    """Per-ROI F0 from Suite2p's baseline estimate, without modifying ``f_corr``."""
    # Suite2p's prctile baseline subtracts in place, so it always gets a copy.
    delta_f = s2p_preprocess(
        np.array(f_corr, dtype=np.float32, copy=True),
        baseline=settings.baseline,
        win_baseline=settings.win_baseline_s,
        sig_baseline=settings.sig_baseline_frames,
        fs=fs_hz,
        prctile_baseline=settings.prctile_baseline,
        batch_size=settings.batch_size,
        device=torch.device(settings.torch_device),
    )
    return f_corr - delta_f


def dff_from_corrected(
    f_corr: np.ndarray, fs_hz: float, settings: FluorescenceSettings
) -> DffResult:
    f_corr = np.asarray(f_corr, dtype=np.float32)
    f0 = baseline(f_corr, fs_hz, settings)
    delta_f = f_corr - f0
    invalid = f0 <= settings.min_f0
    with np.errstate(divide="ignore", invalid="ignore"):
        dff = np.where(invalid, np.nan, delta_f / f0).astype(np.float32)
    return DffResult(f_corr, f0, delta_f, dff, invalid)


def compute_dff(
    f: np.ndarray, fneu: np.ndarray, fs_hz: float, settings: FluorescenceSettings
) -> DffResult:
    """ΔF/F0 of neuropil-corrected traces, keeping F0 and ΔF for inspection."""
    f_corr = neuropil_correct(f, fneu, settings.neuropil_coefficient)
    return dff_from_corrected(f_corr, fs_hz, settings)


def green_red_dff(
    f_green: np.ndarray,
    fneu_green: np.ndarray,
    f_red: np.ndarray,
    fneu_red: np.ndarray,
    fs_hz: float,
    settings: FluorescenceSettings,
) -> DffResult:
    """Δ(G/R)/(G/R)0: ratio to the red channel removes brightness changes both share.

    Both channels are median-filtered first; red is floored at a low percentile so a
    near-zero red value cannot blow the ratio up.
    """
    _check_same_shape(f_green, f_red)
    _check_finite(f_red, fneu_red)
    green = neuropil_correct(f_green, fneu_green, settings.neuropil_coefficient)
    red = neuropil_correct(f_red, fneu_red, settings.neuropil_coefficient)
    window = int(fs_hz * settings.gr_median_window_s) | 1  # medfilt needs an odd width
    green = medfilt(green, kernel_size=(1, window))
    red = medfilt(red, kernel_size=(1, window))
    floor = np.percentile(red, settings.gr_red_floor_percentile, axis=1, keepdims=True)
    floor = np.maximum(floor, settings.gr_red_floor_min)
    return dff_from_corrected(green / np.maximum(red, floor), fs_hz, settings)


def _check_finite(*arrays: np.ndarray) -> None:
    # One NaN frame would spread across a whole baseline window without being flagged.
    for array in arrays:
        if not np.isfinite(array).all():
            raise ValueError("Traces contain non-finite values (NaN or inf)")


def _check_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    if np.shape(a) != np.shape(b):
        raise ValueError(f"Trace arrays differ in shape: {np.shape(a)} vs {np.shape(b)}")


def audit_table(f: np.ndarray, fneu: np.ndarray, result: DffResult) -> dict[str, np.ndarray]:
    """Per-ROI numbers for spotting traces whose F0 or ΔF/F0 cannot be trusted."""
    finite_rows = np.isfinite(result.dff).any(axis=1)
    dff_p99 = np.full(len(result.dff), np.nan, np.float32)
    if finite_rows.any():
        dff_p99[finite_rows] = np.nanpercentile(result.dff[finite_rows], 99, axis=1)
    return {
        "median_f": np.median(f, axis=1),
        "median_fneu": np.median(fneu, axis=1),
        "median_f0": np.median(result.f0, axis=1),
        "lowest_f0": result.f0.min(axis=1),
        "frac_negative_fcorr": (result.f_corr < 0).mean(axis=1),
        "frac_invalid_f0": result.invalid_f0.mean(axis=1),
        "dff_p99": dff_p99,
    }

"""Background and red-channel regression before ΔF/F0 (computation only; reading and
writing files is in ``background_io``).

Two background sources are compared: a blood-vessel mask, and every pixel outside the
cell ROIs (``non_roi_mask``). The first holds no labelled cells but is small; the second is
large but includes unsegmented cells and neuropil, so it carries more neural signal.

A blood vessel holds no labelled cells, so its fluorescence over time is taken as
background: out-of-focus and scattered light, and brightness changes of the whole field.
Its trace is regressed out of each ROI's F and of its neuropil Fneu, each with its own
coefficient, before the usual neuropil correction; regressing it out of F alone would let
``F - c * Fneu`` add part of it back. Least squares is linear, so this equals regressing it
out of Fcorr; the two coefficients are kept as diagnostics. Caveats: light from vessels in
GCaMP tissue includes out-of-focus neural fluorescence (population activity) and changes
with vessel diameter, so the regression can also remove shared, slow neural signal such as
arousal-linked modulation; ROI pixels are left out of the vessel mask so no cell's own
activity enters the regressor.

The red (structural) label is meant not to follow activity, so what a ROI's red trace
shares with its green trace is brightness change from motion or focus, and with a usable
red channel each ROI's red trace (background removed and neuropil corrected the same way)
is regressed out of its green trace. Caveats: green fluorescence bleeding into the red
channel puts activity into red, and the regression then removes part of the real
transients (most when red is clean and motion small); a noisy red trace gives too small a
coefficient, under-correcting motion. Large or negative red coefficients are worth a look.

Regression keeps each trace's mean: only the fluctuations explained by the regressor are
removed, so F0 stays on the original scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FluorescenceSettings
from .fluorescence import DffResult, compute_dff, dff_from_corrected, neuropil_correct
from .tiff_io import chunk_bounds


@dataclass(frozen=True)
class Regression:
    cleaned: np.ndarray  # (n_rois, n_frames), mean of each row unchanged
    coef: np.ndarray  # (n_rois,) regressor coefficient per row


@dataclass(frozen=True)
class DffVersions:
    standard: DffResult  # neuropil correction only
    background: DffResult  # vessel background regressed out of F and Fneu first
    red: DffResult | None  # and each ROI's red trace regressed out; None when not usable
    background_coef: dict[str, np.ndarray]  # "F", "Fneu" (and "F_chan2", "Fneu_chan2")
    red_coef: np.ndarray | None


def vessel_mask(
    labels: np.ndarray, frame_shape: tuple[int, int], exclude: np.ndarray | None = None
) -> np.ndarray:
    """Every labelled pixel of a Cellpose segmentation of blood vessels, as one mask,
    without the pixels in *exclude* (e.g. ROI pixels, whose own activity must not enter
    the background)."""
    labels = np.asarray(labels)
    if labels.shape != tuple(frame_shape):
        raise ValueError(
            f"Vessel mask shape {labels.shape} differs from the registered frames "
            f"{tuple(frame_shape)}; segment this plane's mean image"
        )
    mask = labels > 0
    if not mask.any():
        raise ValueError("The vessel segmentation has no labelled pixels")
    if exclude is not None:
        if np.shape(exclude) != mask.shape:
            raise ValueError(f"Excluded pixels of shape {np.shape(exclude)} vs {mask.shape}")
        mask &= ~np.asarray(exclude, bool)
        if not mask.any():
            raise ValueError("Every vessel pixel is also a ROI pixel; no background is left")
    return mask


def non_roi_mask(roi_labels: np.ndarray, region: tuple[slice, slice]) -> np.ndarray:
    """Every pixel outside the cell ROIs, within *region* (rows, columns): the region no
    frame's registration shifts wrap into, so rolled-in edge pixels are not background."""
    labels = np.asarray(roi_labels)
    mask = np.zeros(labels.shape, bool)
    mask[region] = labels[region] == 0
    if not mask.any():
        raise ValueError("No pixel outside the ROIs is left in the region")
    return mask


def rowwise_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of each row of *a* with the same row of *b* (NaN frames, e.g.
    an undefined ΔF/F0, left out pairwise)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shapes differ: {a.shape} vs {b.shape}")
    both = np.isfinite(a) & np.isfinite(b)
    out = np.full(len(a), np.nan)
    for i, (x, y, keep) in enumerate(zip(a, b, both, strict=True)):
        if keep.sum() > 2 and np.std(x[keep]) > 0 and np.std(y[keep]) > 0:
            out[i] = np.corrcoef(x[keep], y[keep])[0, 1]
    return out


def centred_regressor(regressor: np.ndarray, n_frames: int) -> np.ndarray:
    regressor = np.asarray(regressor, np.float64)
    if regressor.shape[-1] != n_frames:
        raise ValueError(f"Regressor has {regressor.shape[-1]} frames, traces {n_frames}")
    if not np.isfinite(regressor).all():
        raise ValueError("Regressor has values that are not finite")
    return regressor - regressor.mean(axis=-1, keepdims=True)


def regress_out(traces: np.ndarray, regressor: np.ndarray) -> Regression:
    """Least-squares removal of one regressor (n_frames,) from every row of *traces*."""
    traces = np.asarray(traces, np.float64)
    _check_traces(traces)
    centred = centred_regressor(regressor, traces.shape[1])
    power = centred @ centred
    if power == 0:
        raise ValueError("The regressor is flat, so nothing can be regressed out")
    coef = (traces - traces.mean(axis=1, keepdims=True)) @ centred / power
    return Regression((traces - coef[:, None] * centred).astype(np.float32), coef)


def regress_out_each(traces: np.ndarray, regressors: np.ndarray) -> Regression:
    """Least-squares removal of row i of *regressors* from row i of *traces*."""
    traces = np.asarray(traces, np.float64)
    _check_traces(traces)
    if np.shape(regressors) != traces.shape:
        raise ValueError(f"Regressors of shape {np.shape(regressors)} vs traces {traces.shape}")
    centred = centred_regressor(regressors, traces.shape[1])
    power = np.einsum("rt,rt->r", centred, centred)
    flat = np.flatnonzero(power == 0)
    if flat.size:
        raise ValueError(f"The regressor is flat for row(s) {flat[:10].tolist()}")
    traces_centred = traces - traces.mean(axis=1, keepdims=True)
    coef = np.einsum("rt,rt->r", traces_centred, centred) / power
    return Regression((traces - coef[:, None] * centred).astype(np.float32), coef)


def _check_traces(traces: np.ndarray) -> None:
    if traces.ndim != 2:
        raise ValueError(f"Traces must be (n_rois, n_frames), got shape {traces.shape}")
    if not np.isfinite(traces).all():
        raise ValueError("Traces have values that are not finite")


def dff_versions(
    f: np.ndarray,
    fneu: np.ndarray,
    f_chan2: np.ndarray | None,
    fneu_chan2: np.ndarray | None,
    background_green: np.ndarray,
    background_red: np.ndarray | None,
    fs_hz: float,
    settings: FluorescenceSettings,
    red_usable: bool,
    standard: DffResult | None = None,
) -> DffVersions:
    """ΔF/F0 three ways: standard, vessel background removed, and red regressed out too.

    The red version needs red traces, a red background trace and *red_usable*. *standard*
    is the already computed ``compute_dff(f, fneu, ...)``, if there is one.
    """
    if standard is None:
        standard = compute_dff(f, fneu, fs_hz, settings)
    f_bg, fneu_bg = regress_out(f, background_green), regress_out(fneu, background_green)
    background = compute_dff(f_bg.cleaned, fneu_bg.cleaned, fs_hz, settings)
    coefs = {"F": f_bg.coef, "Fneu": fneu_bg.coef}
    have_red = f_chan2 is not None and fneu_chan2 is not None and background_red is not None
    if not (red_usable and have_red):
        return DffVersions(standard, background, None, coefs, None)
    f2_bg, fneu2_bg = regress_out(f_chan2, background_red), regress_out(fneu_chan2, background_red)
    coefs |= {"F_chan2": f2_bg.coef, "Fneu_chan2": fneu2_bg.coef}
    red_corrected = neuropil_correct(f2_bg.cleaned, fneu2_bg.cleaned, settings.neuropil_coefficient)
    without_red = regress_out_each(background.f_corr, red_corrected)
    red = dff_from_corrected(without_red.cleaned, fs_hz, settings)
    return DffVersions(standard, background, red, coefs, without_red.coef)


# ---- reading the registered binaries ------------------------------------------------


def mask_mean_trace(stack: np.ndarray, mask: np.ndarray, chunk_frames: int) -> np.ndarray:
    """Mean of the *mask* pixels in every frame of a (frames, Ly, Lx) stack, read in chunks."""
    if np.shape(stack)[1:] != np.shape(mask):
        raise ValueError(f"Mask shape {np.shape(mask)} vs frames {np.shape(stack)[1:]}")
    return np.concatenate(
        [
            np.asarray(stack[start:end])[:, mask].astype(np.float64).mean(axis=1)
            for start, end in chunk_bounds(len(stack), chunk_frames)
        ]
    )


def pixel_coefficients(stack: np.ndarray, regressor: np.ndarray, chunk_frames: int) -> np.ndarray:
    """Least-squares coefficient of *regressor* (n_frames,) for every pixel of the stack."""
    centred = centred_regressor(regressor, len(stack))
    power = centred @ centred
    if power == 0:
        raise ValueError("The regressor is flat, so nothing can be regressed out")
    total = np.zeros(np.shape(stack)[1:], np.float64)
    for start, end in chunk_bounds(len(stack), chunk_frames):
        # sum(pixel * centred regressor) needs no centred pixel: the regressor sums to zero
        total += np.tensordot(centred[start:end], np.asarray(stack[start:end], np.float64), 1)
    return total / power


def corrected_chunk(
    stack: np.ndarray, start: int, end: int, coef: np.ndarray, centred: np.ndarray
) -> np.ndarray:
    """Frames start..end with the regressor removed from each pixel (means unchanged)."""
    frames = np.asarray(stack[start:end], np.float32)
    return frames - centred[start:end, None, None].astype(np.float32) * coef.astype(np.float32)


def top_rois(dff: np.ndarray, n: int) -> np.ndarray:
    """The *n* ROIs with the largest maximum ΔF/F0, largest first (all-NaN rows last)."""
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    peaks = np.full(len(dff), -np.inf)
    defined = np.isfinite(dff).any(axis=1)
    peaks[defined] = np.nanmax(dff[defined], axis=1)
    return np.argsort(-peaks, kind="stable")[:n]

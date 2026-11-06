"""Ways to correct a ROI's raw F before ΔF/F0, and numbers to compare them by (computation
only; the extraction comparison notebook loads the run and draws).

Suite2p's correction ``F - 0.7 * Fneu`` assumes the ring of neuropil around a ROI carries
contaminating calcium signal about as bright as the halo inside it. With an indicator kept in
the soma (ribosome-tethered GCaMP), the neuropil holds little indicator: its signal is mostly
out-of-focus somata, autofluorescence and any detector offset, so subtracting 0.7 of it can
take most of a dim cell's own baseline and leave F0 at or below ``min_f0``. The corrections:

- ``Neuropil <c>``: the fixed coefficient (``[fluorescence] neuropil_coefficient``);
- ``Neuropil Fitted``: each ROI's own coefficient (``fitted_neuropil_coefficients``);
- ``No Neuropil``: raw F;
- ``No Neuropil, <source> Bg Regressed``: raw F with a background trace (vessel, or every
  pixel outside the ROIs) regressed out, each ROI with its own coefficient, mean kept;
- ``Neuropil <c>, <source> Bg Regressed``: the background removed from F and Fneu before the
  fixed neuropil correction, as in the QC notebook (``background.dff_versions``).

Slow changes (bleaching, drift) dominate the variance of raw traces, so fits and correlations
here use high-passed traces: each minus its running mean over ``detrend_s``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from .background import regress_out
from .fluorescence import DffResult, neuropil_correct

MAD_TO_SD = 1.4826  # scales a median absolute deviation to a normal standard deviation
FIT_ITERATIONS = 3  # fits without the frames above the previous fit
PAIRWISE_MAX_ROIS = 300  # mean pairwise correlation on at most this many ROIs (evenly spaced)
DETREND_S = 60.0


def high_pass(traces: np.ndarray, fs_hz: float, detrend_s: float = DETREND_S) -> np.ndarray:
    """Each trace (last axis: frames) minus its running mean over *detrend_s* seconds."""
    window = max(3, int(round(detrend_s * fs_hz)) | 1)  # odd: centred on each frame
    traces = np.asarray(traces, np.float64)
    return traces - uniform_filter1d(traces, window, axis=-1, mode="nearest")


def _check_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} has values that are not finite")


# ---------------------------------------------------------------- neuropil coefficient


def _robust_sd(values: np.ndarray, axis=None) -> np.ndarray:
    centre = np.nanmedian(values, axis=axis, keepdims=True)
    return MAD_TO_SD * np.nanmedian(np.abs(values - centre), axis=axis)


def _fit_one(f: np.ndarray, regressors: np.ndarray, transient_sd: float) -> float:
    """Least-squares coefficients of F on *regressors* (columns; the first is the neuropil),
    each fit after the first without the frames more than *transient_sd* robust SDs above
    the previous fit (the cell's transients). Returns the neuropil's coefficient."""
    design = np.column_stack([regressors, np.ones(len(f))])
    keep = np.ones(len(f), bool)
    for _ in range(FIT_ITERATIONS):
        coef = np.linalg.lstsq(design[keep], f[keep], rcond=None)[0]
        residual = f - design @ coef
        keep = residual <= np.median(residual[keep]) + transient_sd * _robust_sd(residual[keep])
    return float(np.linalg.lstsq(design[keep], f[keep], rcond=None)[0][0])


def fitted_neuropil_coefficients(
    f: np.ndarray,
    fneu: np.ndarray,
    fs_hz: float,
    background: np.ndarray | None = None,
    detrend_s: float = DETREND_S,
    transient_sd: float = 2.0,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Each ROI's neuropil coefficient: the c that best removes its neuropil's fast
    fluctuations from it, i.e. the least-squares slope of high-passed F on high-passed Fneu
    (with the high-passed *background* as a second regressor, so a field-wide change is not
    taken as neuropil; a noisy background trace, e.g. from a small mask, lets part of that
    change through and biases the coefficient high), kept in *bounds*. The cell's own
    transients (frames far above the fit) are left out of the later fits, which steadies the
    slope of an active cell.

    Not a measurement of the contamination: noise and local signal in Fneu that never reach
    the ROI pull a least-squares slope towards 0, and activity that appears in both the ROI
    and its neuropil (population events, the cell's own light spilling into the ring) pushes
    it up.

    Raises:
        ValueError: If *transient_sd* is not positive, the traces are not finite or do not
            match, or a ROI's neuropil has no fluctuation left after high-passing.
    """
    if transient_sd <= 0:
        raise ValueError(f"transient_sd must be positive, got {transient_sd}")
    f, fneu = np.asarray(f, np.float64), np.asarray(fneu, np.float64)
    if f.shape != fneu.shape or f.ndim != 2:
        raise ValueError(f"F {f.shape} and Fneu {fneu.shape} must be the same (ROIs, frames)")
    _check_finite("F", f)
    _check_finite("Fneu", fneu)
    f_hp, fneu_hp = high_pass(f, fs_hz, detrend_s), high_pass(fneu, fs_hz, detrend_s)
    flat = np.flatnonzero(np.ptp(fneu_hp, axis=1) == 0)
    if flat.size:
        raise ValueError(f"Neuropil is flat for ROI(s) {flat[:10].tolist()}; no slope to fit")
    extra = []
    if background is not None:
        background = np.asarray(background, np.float64)
        if background.shape != (f.shape[1],):
            raise ValueError(f"Background of shape {background.shape} for {f.shape[1]} frames")
        _check_finite("Background", background)
        extra = [high_pass(background, fs_hz, detrend_s)]
    slopes = [
        _fit_one(row, np.column_stack([neu, *extra]), transient_sd)
        for row, neu in zip(f_hp, fneu_hp, strict=True)
    ]
    return np.clip(slopes, *bounds)


def contrast_ratio(f: np.ndarray, fneu: np.ndarray) -> np.ndarray:
    """Median F over median Fneu per ROI: how far each ROI stands above its surroundings.
    NaN where the neuropil's median is not positive (no ratio to give; a detector offset at
    or below zero shows up this way)."""
    f_med = np.median(np.asarray(f, np.float64), axis=1)
    n_med = np.median(np.asarray(fneu, np.float64), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n_med > 0, f_med / n_med, np.nan)


# ---------------------------------------------------------------- corrections


def corrected_traces(
    f: np.ndarray,
    fneu: np.ndarray,
    backgrounds: dict[str, np.ndarray],
    neuropil_coefficient: float,
    fitted: np.ndarray,
) -> dict[str, np.ndarray]:
    """The corrected F of every method (see the module docstring), by method name.

    Args:
        backgrounds: background trace (n_frames,) of each source, by label (e.g. "Vessel").
        fitted: each ROI's coefficient, from ``fitted_neuropil_coefficients``.
    """
    f = np.asarray(f, np.float32)
    fitted = np.asarray(fitted, np.float32)
    if fitted.shape != (len(f),):
        raise ValueError(f"{len(fitted)} fitted coefficients for {len(f)} ROIs")
    fixed = f"Neuropil {neuropil_coefficient:g}"
    traces = {
        fixed: neuropil_correct(f, fneu, neuropil_coefficient),
        "Neuropil Fitted": f - fitted[:, None] * np.asarray(fneu, np.float32),
        "No Neuropil": f,
    }
    for label, background in backgrounds.items():
        traces[f"No Neuropil, {label} Bg Regressed"] = regress_out(f, background).cleaned
    for label, background in backgrounds.items():
        f_bg = regress_out(f, background).cleaned
        fneu_bg = regress_out(fneu, background).cleaned
        traces[f"{fixed}, {label} Bg Regressed"] = neuropil_correct(
            f_bg, fneu_bg, neuropil_coefficient
        )
    return traces


# ---------------------------------------------------------------- comparing them


def valid_everywhere(results: dict[str, DffResult]) -> np.ndarray:
    """ROIs whose F0 is valid in every frame under every correction: the common set that
    noise and skew are compared on."""
    return np.logical_and.reduce([~np.asarray(r.invalid_f0).any(axis=1) for r in results.values()])


def _noise(dff: np.ndarray) -> np.ndarray:
    """Frame-to-frame SD of each ROI's ΔF/F0 (robust SD of the differences / sqrt 2)."""
    return _robust_sd(np.diff(dff, axis=1), axis=1) / np.sqrt(2)


def _skew(dff: np.ndarray) -> np.ndarray:
    centred = dff - np.mean(dff, axis=1, keepdims=True)
    sd = np.std(dff, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, np.mean(centred**3, axis=1) / sd**3, np.nan)


def _mean_pairwise_r(traces: np.ndarray) -> float:
    if len(traces) < 2:
        raise ValueError("Pairwise correlation needs at least two ROIs")
    rows = np.unique(np.linspace(0, len(traces) - 1, PAIRWISE_MAX_ROIS).round().astype(int))
    r = np.corrcoef(traces[rows])
    return float(np.nanmean(r[~np.eye(len(rows), dtype=bool)]))


def method_metrics(
    result: DffResult,
    background: np.ndarray,
    fs_hz: float,
    common: np.ndarray,
    detrend_s: float = DETREND_S,
) -> dict[str, float]:
    """Numbers to compare one correction by.

    - ``rois_invalid_f0_anywhere`` / ``rois_invalid_f0_mostly``: ROIs whose F0 is at or below
      ``min_f0`` in any frame / in at least half of the frames;
    - ``fraction_frames_invalid_f0``: of all ROI-frames, those without a ΔF/F0;
    - ``median_f0``: median over all ROIs of each ROI's median F0;
    - ``median_noise``, ``median_skew``: frame-to-frame SD and skewness of ΔF/F0, median over
      the *common* ROIs (valid under every correction; ``valid_everywhere``), so every
      correction is summarised over the same cells;
    - ``median_abs_r_background``: median |r| of the high-passed corrected F with the
      high-passed *background* (the fast background left in the traces; zero by construction
      for a correction that regressed out this background or one nearly identical to it);
    - ``mean_pairwise_r``: mean correlation between the ROIs' high-passed corrected F, over up
      to ``PAIRWISE_MAX_ROIS`` ROIs (raised by any signal shared across the field).

    With no common ROI, noise and skew are NaN (the counts are still given).
    """
    common = np.asarray(common, bool)
    f_hp = high_pass(result.f_corr, fs_hz, detrend_s)
    bg_hp = high_pass(background, fs_hz, detrend_s)
    f_c = f_hp - f_hp.mean(axis=1, keepdims=True)
    b_c = bg_hp - bg_hp.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        r_background = f_c @ b_c / (np.linalg.norm(f_c, axis=1) * np.linalg.norm(b_c))
    invalid = np.asarray(result.invalid_f0)
    dff = np.asarray(result.dff, np.float64)[common]
    spread = (
        (float(np.median(_noise(dff))), float(np.nanmedian(_skew(dff))))
        if common.any()
        else (np.nan, np.nan)
    )
    return {
        "rois_invalid_f0_anywhere": int(invalid.any(axis=1).sum()),
        "rois_invalid_f0_mostly": int((invalid.mean(axis=1) >= 0.5).sum()),
        "fraction_frames_invalid_f0": float(invalid.mean()),
        "median_f0": float(np.median(np.median(result.f0, axis=1))),
        "median_noise": spread[0],
        "median_skew": spread[1],
        "median_abs_r_background": float(np.nanmedian(np.abs(r_background))),
        "mean_pairwise_r": _mean_pairwise_r(f_hp),
    }


def channel_fraction(
    trace: np.ndarray, fs_hz: float, dark: float = 0.0, detrend_s: float = DETREND_S, name: str = ""
) -> np.ndarray:
    """A background trace high-passed, as a fraction of its mean above the dark level.

    Raises:
        ValueError: If the trace is not finite or its mean is not above *dark*.
    """
    trace = np.asarray(trace, np.float64)
    _check_finite(f"The {name} background".replace("  ", " "), trace)
    level = trace.mean() - dark
    if level <= 0:
        raise ValueError(f"The {name} background's mean is not above its dark level ({dark:g})")
    return high_pass(trace, fs_hz, detrend_s) / level


def background_channel_check(
    green: np.ndarray,
    red: np.ndarray,
    fs_hz: float,
    dark_green: float = 0.0,
    dark_red: float = 0.0,
    detrend_s: float = DETREND_S,
) -> dict[str, float]:
    """How the green background's fast fluctuation relates to the red one's, each as a
    fraction of its mean above the detector's dark level, after high-passing (so bleaching
    and slow drift, which differ between channels, do not count).

    - ``green_on_red_slope``: least squares of green's fractional change on red's. Whatever
      moves both channels by the same fraction (laser power, shared detection, focus or
      motion) gives 1; blood volume dims green more than red (hemoglobin absorbs green
      more), above 1. Signal in green alone (indicator) does not bias it, it only lowers r.
      Noise in red pulls it towards 0.
    - ``r_squared``: the fraction of green's fluctuation shared with red; the rest is in
      green alone (indicator signal, or green's noise).
    - ``slope_low``, ``slope_high``: the slope's range, from green on red to the reverse
      regression (slope / r²). Noise in red pulls the first down, signal or noise in green
      alone pushes the second up; the shared part's true slope lies between them, so a
      reading of "near 1" needs a narrow range.

    Raises:
        ValueError: If a channel's mean is not above its dark level.
    """
    g = channel_fraction(green, fs_hz, dark_green, detrend_s, "green")
    r = channel_fraction(red, fs_hz, dark_red, detrend_s, "red")
    g, r = g - g.mean(), r - r.mean()
    correlation = float(np.corrcoef(g, r)[0, 1])
    slope = float(g @ r / (r @ r))
    reverse = slope / correlation**2 if correlation != 0 else np.inf
    return {
        "green_on_red_slope": slope,
        "slope_low": min(slope, reverse),
        "slope_high": max(slope, reverse),
        "r": correlation,
        "r_squared": correlation**2,
        "green_sd_fraction": float(g.std()),
        "red_sd_fraction": float(r.std()),
    }
